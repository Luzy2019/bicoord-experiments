from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import reduce

from flow_matching_policy.common.pytorch_util import dict_apply
from flow_matching_policy.model.common.normalizer import LinearNormalizer
from flow_matching_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from flow_matching_policy.model.vision.multi_image_obs_encoder import MultiImageObsEncoder
from flow_matching_policy.policy.base_image_policy import BaseImagePolicy


def _slice_from_start_dim(start: int, dim: int) -> slice:
    return slice(int(start), int(start) + int(dim))


class LatentActionWorldModel(nn.Module):
    """Latent verifier for action-only compact scheduling."""

    def __init__(
        self,
        obs_feature_dim: int,
        action_dim: int,
        latent_dim: int = 128,
        hidden_dim: int = 256,
        detach_target: bool = True,
    ):
        super().__init__()
        self.detach_target = bool(detach_target)
        self.projector = nn.Sequential(
            nn.LayerNorm(obs_feature_dim),
            nn.Linear(obs_feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.dynamics = nn.Sequential(
            nn.LayerNorm(latent_dim + action_dim),
            nn.Linear(latent_dim + action_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def encode(self, obs_features: torch.Tensor) -> torch.Tensor:
        return self.projector(obs_features)

    def compute_loss(
        self,
        obs_features: torch.Tensor,
        actions: torch.Tensor,
        gaussian_weight: float = 0.01,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if obs_features.shape[1] < 2:
            zero = obs_features.new_zeros(())
            return zero, {
                "world_pred_loss": zero,
                "world_gaussian_loss": zero,
                "world_latent_std": zero,
            }

        z = self.encode(obs_features)
        z_now = z[:, :-1]
        z_next = z[:, 1:]
        transition_actions = actions[:, : z_now.shape[1]]
        pred_next = z_now + self.dynamics(torch.cat([z_now, transition_actions], dim=-1))
        target_next = z_next.detach() if self.detach_target else z_next
        pred_loss = F.mse_loss(pred_next, target_next)

        flat_z = z.reshape(-1, z.shape[-1])
        latent_mean = flat_z.mean(dim=0)
        latent_std = flat_z.std(dim=0, unbiased=False)
        gaussian_loss = latent_mean.pow(2).mean() + (latent_std - 1.0).pow(2).mean()
        total = pred_loss + float(gaussian_weight) * gaussian_loss
        return total, {
            "world_pred_loss": pred_loss,
            "world_gaussian_loss": gaussian_loss,
            "world_latent_std": latent_std.mean(),
        }

    def score(self, obs_features: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        z = self.encode(obs_features)[:, -1]
        z_repeat = z.unsqueeze(1).expand(-1, actions.shape[1], -1)
        delta = self.dynamics(torch.cat([z_repeat, actions], dim=-1))
        return delta.pow(2).mean(dim=(1, 2))


class ActionEnergyHead(nn.Module):
    """Scalar demo-like feasibility energy over action chunks."""

    def __init__(
        self,
        action_dim: int,
        cond_dim: int,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(action_dim * 2 + cond_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, actions: torch.Tensor, global_cond: torch.Tensor) -> torch.Tensor:
        pooled = torch.cat(
            [
                actions.mean(dim=1),
                actions.std(dim=1, unbiased=False),
            ],
            dim=-1,
        )
        return self.net(torch.cat([pooled, global_cond], dim=-1)).squeeze(-1)


class BimanualCouplingEstimator(nn.Module):
    """Auxiliary marginal/conditional vector fields for learned arm dependency."""

    def __init__(
        self,
        left_action_dim: int,
        right_action_dim: int,
        global_cond_dim: int,
        diffusion_step_embed_dim=256,
        down_dims=(256, 512, 1024),
        kernel_size=5,
        n_groups=8,
        cond_predict_scale=True,
    ):
        super().__init__()
        self.left_action_dim = int(left_action_dim)
        self.right_action_dim = int(right_action_dim)
        left_cond_dim = global_cond_dim + self.right_action_dim * 2 + 2
        right_cond_dim = global_cond_dim + self.left_action_dim * 2 + 2
        self.left_model = ConditionalUnet1D(
            input_dim=self.left_action_dim,
            local_cond_dim=None,
            global_cond_dim=left_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale,
        )
        self.right_model = ConditionalUnet1D(
            input_dim=self.right_action_dim,
            local_cond_dim=None,
            global_cond_dim=right_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale,
        )

    def _flag(self, batch_size: int, value: float, ref: torch.Tensor) -> torch.Tensor:
        return torch.full((batch_size, 1), float(value), device=ref.device, dtype=ref.dtype)

    def _context(self, actions: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [
                actions.mean(dim=1),
                actions.std(dim=1, unbiased=False),
            ],
            dim=-1,
        )

    def _branch_cond(
        self,
        global_cond: torch.Tensor,
        other_context: torch.Tensor,
        arm_id: float,
        cond_mask: float,
    ) -> torch.Tensor:
        batch_size = global_cond.shape[0]
        return torch.cat(
            [
                global_cond,
                other_context,
                self._flag(batch_size, arm_id, global_cond),
                self._flag(batch_size, cond_mask, global_cond),
            ],
            dim=-1,
        )

    def forward(
        self,
        left_input: torch.Tensor,
        right_input: torch.Tensor,
        timesteps: torch.Tensor,
        global_cond: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        left_context = self._context(left_input)
        right_context = self._context(right_input)
        zero_left_context = torch.zeros_like(left_context)
        zero_right_context = torch.zeros_like(right_context)

        left_marginal = self.left_model(
            left_input,
            timesteps,
            global_cond=self._branch_cond(global_cond, zero_right_context, arm_id=0.0, cond_mask=0.0),
        )
        right_marginal = self.right_model(
            right_input,
            timesteps,
            global_cond=self._branch_cond(global_cond, zero_left_context, arm_id=1.0, cond_mask=0.0),
        )
        left_cond = self.left_model(
            left_input,
            timesteps,
            global_cond=self._branch_cond(global_cond, right_context, arm_id=0.0, cond_mask=1.0),
        )
        right_cond = self.right_model(
            right_input,
            timesteps,
            global_cond=self._branch_cond(global_cond, left_context, arm_id=1.0, cond_mask=1.0),
        )
        return {
            "left_marginal": left_marginal,
            "right_marginal": right_marginal,
            "left_cond": left_cond,
            "right_cond": right_cond,
        }

    def compute_aux_loss(
        self,
        left_input: torch.Tensor,
        right_input: torch.Tensor,
        timesteps: torch.Tensor,
        global_cond: torch.Tensor,
        left_target: torch.Tensor,
        right_target: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        pred = self(left_input, right_input, timesteps, global_cond)
        left_marginal_per = reduce(
            F.mse_loss(pred["left_marginal"], left_target, reduction="none"),
            "b ... -> b (...)",
            "mean",
        )
        right_marginal_per = reduce(
            F.mse_loss(pred["right_marginal"], right_target, reduction="none"),
            "b ... -> b (...)",
            "mean",
        )
        left_cond_per = reduce(
            F.mse_loss(pred["left_cond"], left_target, reduction="none"),
            "b ... -> b (...)",
            "mean",
        )
        right_cond_per = reduce(
            F.mse_loss(pred["right_cond"], right_target, reduction="none"),
            "b ... -> b (...)",
            "mean",
        )
        aux_loss = 0.25 * (
            left_marginal_per.mean()
            + right_marginal_per.mean()
            + left_cond_per.mean()
            + right_cond_per.mean()
        )
        coupling_l_to_r = F.relu(right_marginal_per - right_cond_per)
        coupling_r_to_l = F.relu(left_marginal_per - left_cond_per)
        logs = {
            "coupling_aux_loss": aux_loss,
            "left_marginal_aux_loss": left_marginal_per.mean(),
            "right_marginal_aux_loss": right_marginal_per.mean(),
            "left_cond_aux_loss": left_cond_per.mean(),
            "right_cond_aux_loss": right_cond_per.mean(),
            "coupling_l_to_r": coupling_l_to_r.mean(),
            "coupling_r_to_l": coupling_r_to_l.mean(),
        }
        return aux_loss, logs

    def estimate(
        self,
        normalized_actions: torch.Tensor,
        timesteps: torch.Tensor,
        global_cond: torch.Tensor,
        left_slice: slice,
        right_slice: slice,
    ) -> torch.Tensor:
        left = normalized_actions[:, :, left_slice]
        right = normalized_actions[:, :, right_slice]
        pred = self(left, right, timesteps, global_cond)
        left_depends_on_right = (pred["left_cond"] - pred["left_marginal"]).pow(2).mean(dim=-1)
        right_depends_on_left = (pred["right_cond"] - pred["right_marginal"]).pow(2).mean(dim=-1)
        return torch.sqrt(
            torch.stack([right_depends_on_left, left_depends_on_right], dim=-1).clamp_min(1.0e-12)
        )


class CompactScheduler:
    """Generic coupling-aware schedule search without alpha, gates, or action warping."""

    def __init__(
        self,
        enabled: bool = True,
        num_samples: int = 16,
        lambda_time: float = 0.1,
        dependency_weight: float = 1.0,
        dynamics_weight: float = 0.25,
        risk_weight: float = 1.0,
        min_duration_scale: float = 0.5,
        max_duration_scale: float = 1.0,
        max_offset_scale: float = 0.25,
    ):
        self.enabled = bool(enabled)
        self.num_samples = max(int(num_samples), 1)
        self.lambda_time = float(lambda_time)
        self.dependency_weight = float(dependency_weight)
        self.dynamics_weight = float(dynamics_weight)
        self.risk_weight = float(risk_weight)
        self.min_duration_scale = float(min_duration_scale)
        self.max_duration_scale = float(max_duration_scale)
        self.max_offset_scale = float(max_offset_scale)

    def _linear_schedule(self, batch_size: int, horizon: int, device, dtype) -> torch.Tensor:
        base = torch.arange(horizon, device=device, dtype=dtype) / max(horizon, 1)
        return base.view(1, horizon, 1).expand(batch_size, horizon, 2)

    def search(
        self,
        actions: torch.Tensor,
        coupling_scores: torch.Tensor,
        energy_scores: torch.Tensor,
        wm_scores: torch.Tensor,
        left_slice: slice,
        right_slice: slice,
    ) -> Dict[str, torch.Tensor]:
        batch_size, horizon = actions.shape[:2]
        device = actions.device
        dtype = actions.dtype
        risk = F.softplus(energy_scores) + F.softplus(wm_scores)
        if (not self.enabled) or horizon <= 1:
            schedule = self._linear_schedule(batch_size, horizon, device, dtype)
            makespan = torch.ones(batch_size, device=device, dtype=dtype)
            zero = torch.zeros(batch_size, device=device, dtype=dtype)
            total = self.risk_weight * risk + self.lambda_time * makespan
            return {
                "schedule": schedule,
                "makespan": makespan,
                "dependency_cost": zero,
                "dynamics_cost": zero,
                "risk_cost": risk,
                "total_cost": total,
            }

        num = self.num_samples
        base_dt = actions.new_tensor(1.0 / horizon)
        scales = torch.empty(num, batch_size, horizon, 2, device=device, dtype=dtype).uniform_(
            self.min_duration_scale,
            self.max_duration_scale,
        )
        scales[0].fill_(1.0)
        durations = base_dt * scales
        offsets = torch.empty(num, batch_size, 1, 2, device=device, dtype=dtype).uniform_(
            0.0,
            self.max_offset_scale / horizon,
        )
        offsets[0].zero_()
        starts = offsets + torch.cumsum(
            torch.cat([torch.zeros_like(durations[:, :, :1]), durations[:, :, :-1]], dim=2),
            dim=2,
        )
        left_start = starts[..., 0]
        right_start = starts[..., 1]
        left_end = left_start + durations[..., 0]
        right_end = right_start + durations[..., 1]
        makespan = torch.maximum(left_end[:, :, -1], right_end[:, :, -1])

        l_to_r = coupling_scores[..., 0].unsqueeze(0)
        r_to_l = coupling_scores[..., 1].unsqueeze(0)
        violation_l_to_r = F.relu(left_start - right_start)
        violation_r_to_l = F.relu(right_start - left_start)
        sync_weight = torch.minimum(l_to_r, r_to_l)
        dependency_cost = (
            l_to_r * violation_l_to_r
            + r_to_l * violation_r_to_l
            + sync_weight * (left_start - right_start).abs()
        ).mean(dim=-1)

        left_actions = actions[:, :, left_slice]
        right_actions = actions[:, :, right_slice]
        left_delta = torch.cat([left_actions[:, :1] * 0.0, left_actions[:, 1:] - left_actions[:, :-1]], dim=1)
        right_delta = torch.cat([right_actions[:, :1] * 0.0, right_actions[:, 1:] - right_actions[:, :-1]], dim=1)
        left_motion = left_delta.pow(2).mean(dim=-1).unsqueeze(0)
        right_motion = right_delta.pow(2).mean(dim=-1).unsqueeze(0)
        dynamics_cost = (
            left_motion / durations[..., 0].clamp_min(1.0e-4)
            + right_motion / durations[..., 1].clamp_min(1.0e-4)
        ).mean(dim=-1)

        risk_cost = risk.unsqueeze(0).expand(num, -1)
        total = (
            self.lambda_time * makespan
            + self.dependency_weight * dependency_cost
            + self.dynamics_weight * dynamics_cost
            + self.risk_weight * risk_cost
        )
        best_idx = total.argmin(dim=0)
        batch_idx = torch.arange(batch_size, device=device)
        return {
            "schedule": starts[best_idx, batch_idx],
            "makespan": makespan[best_idx, batch_idx],
            "dependency_cost": dependency_cost[best_idx, batch_idx],
            "dynamics_cost": dynamics_cost[best_idx, batch_idx],
            "risk_cost": risk,
            "total_cost": total[best_idx, batch_idx],
        }


class CouplingScheduledFlowUnetImagePolicy(BaseImagePolicy):
    """Action-only Flow Matching with learned coupling-aware compact scheduling.

    The class name is kept for config compatibility, but the generated object is
    now A=(A_L,A_R). Timing is an external schedule optimized for compactness.
    """

    def __init__(
        self,
        shape_meta: dict,
        noise_scheduler,
        obs_encoder: MultiImageObsEncoder,
        horizon,
        n_action_steps,
        n_obs_steps,
        num_inference_steps=None,
        obs_as_global_cond=True,
        diffusion_step_embed_dim=256,
        down_dims=(256, 512, 1024),
        kernel_size=5,
        n_groups=8,
        cond_predict_scale=True,
        left_action_dim=None,
        right_action_dim=None,
        left_action_start=0,
        right_action_start=None,
        coupling_enabled=True,
        coupling_aux_loss_weight=0.25,
        scheduler_enabled=True,
        scheduler_num_samples=16,
        scheduler_lambda_time=0.1,
        scheduler_dependency_weight=1.0,
        scheduler_dynamics_weight=0.25,
        scheduler_risk_weight=1.0,
        scheduler_min_duration_scale=0.5,
        scheduler_max_duration_scale=1.0,
        scheduler_max_offset_scale=0.25,
        action_rerank_samples=4,
        energy_loss_weight=0.01,
        energy_hidden_dim=256,
        energy_margin=1.0,
        world_model_enabled=True,
        world_model_loss_weight=0.05,
        world_model_gaussian_weight=0.01,
        world_model_latent_dim=128,
        world_model_hidden_dim=256,
        **kwargs,
    ):
        super().__init__()
        if not obs_as_global_cond:
            raise ValueError("CouplingScheduledFlowUnetImagePolicy requires obs_as_global_cond=True")

        action_shape = shape_meta["action"]["shape"]
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        obs_feature_dim = obs_encoder.output_shape()[0]
        global_cond_dim = obs_feature_dim * n_obs_steps

        if left_action_dim is None:
            left_action_dim = action_dim // 2
        if right_action_dim is None:
            right_action_dim = action_dim - int(left_action_dim)
        if right_action_start is None:
            right_action_start = int(left_action_start) + int(left_action_dim)

        self.left_action_dim = int(left_action_dim)
        self.right_action_dim = int(right_action_dim)
        self.left_slice = _slice_from_start_dim(int(left_action_start), self.left_action_dim)
        self.right_slice = _slice_from_start_dim(int(right_action_start), self.right_action_dim)

        self.model = ConditionalUnet1D(
            input_dim=action_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale,
        )
        self.obs_encoder = obs_encoder
        self.noise_scheduler = noise_scheduler
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.kwargs = kwargs
        self.coupling_enabled = bool(coupling_enabled)
        self.coupling_aux_loss_weight = float(coupling_aux_loss_weight)
        self.energy_loss_weight = float(energy_loss_weight)
        self.energy_margin = float(energy_margin)
        self.world_model_enabled = bool(world_model_enabled)
        self.world_model_loss_weight = float(world_model_loss_weight)
        self.world_model_gaussian_weight = float(world_model_gaussian_weight)
        self.action_rerank_samples = int(action_rerank_samples)
        self.last_loss_dict = {}
        self.last_candidate_scores = None
        self.last_compact_schedule = None
        self.last_coupling_scores = None

        if self.coupling_enabled:
            self.coupling_estimator = BimanualCouplingEstimator(
                left_action_dim=self.left_action_dim,
                right_action_dim=self.right_action_dim,
                global_cond_dim=global_cond_dim,
                diffusion_step_embed_dim=diffusion_step_embed_dim,
                down_dims=down_dims,
                kernel_size=kernel_size,
                n_groups=n_groups,
                cond_predict_scale=cond_predict_scale,
            )
        else:
            self.coupling_estimator = None

        self.action_energy = ActionEnergyHead(
            action_dim=action_dim,
            cond_dim=global_cond_dim,
            hidden_dim=int(energy_hidden_dim),
        )
        if self.world_model_enabled:
            self.action_world_model = LatentActionWorldModel(
                obs_feature_dim=obs_feature_dim,
                action_dim=action_dim,
                latent_dim=int(world_model_latent_dim),
                hidden_dim=int(world_model_hidden_dim),
            )
        else:
            self.action_world_model = None

        self.compact_scheduler = CompactScheduler(
            enabled=scheduler_enabled,
            num_samples=scheduler_num_samples,
            lambda_time=scheduler_lambda_time,
            dependency_weight=scheduler_dependency_weight,
            dynamics_weight=scheduler_dynamics_weight,
            risk_weight=scheduler_risk_weight,
            min_duration_scale=scheduler_min_duration_scale,
            max_duration_scale=scheduler_max_duration_scale,
            max_offset_scale=scheduler_max_offset_scale,
        )

        if num_inference_steps is None:
            num_inference_steps = getattr(getattr(noise_scheduler, "config", None), "num_train_timesteps", 100)
        self.num_inference_steps = num_inference_steps

    def _encode_obs_sequence(self, nobs: Dict[str, torch.Tensor]) -> torch.Tensor:
        value = next(iter(nobs.values()))
        batch_size = value.shape[0]
        this_nobs = dict_apply(nobs, lambda x: x[:, : self.n_obs_steps, ...].reshape(-1, *x.shape[2:]))
        obs_features = self.obs_encoder(this_nobs)
        return obs_features.reshape(batch_size, self.n_obs_steps, -1)

    def _encode_global_cond(self, obs_features: torch.Tensor) -> torch.Tensor:
        return obs_features.reshape(obs_features.shape[0], -1)

    def _sample_action(self, batch_size: int, global_cond: torch.Tensor, generator=None) -> torch.Tensor:
        action = torch.randn(
            size=(batch_size, self.horizon, self.action_dim),
            dtype=self.dtype,
            device=self.device,
            generator=generator,
        )
        num_steps = max(int(self.num_inference_steps), 1)
        dt = 1.0 / num_steps
        for step_idx in range(num_steps):
            t = torch.full(
                (batch_size,),
                step_idx / num_steps,
                dtype=action.dtype,
                device=action.device,
            )
            velocity = self.model(action, t, global_cond=global_cond)
            action = action + dt * velocity
        return action

    def _energy_contrastive_loss(
        self,
        positive_actions: torch.Tensor,
        global_cond: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        positive_energy = self.action_energy(positive_actions, global_cond)
        negative = positive_actions.clone()
        if positive_actions.shape[0] > 1:
            perm = torch.randperm(positive_actions.shape[0], device=positive_actions.device)
            negative[:, :, self.right_slice] = positive_actions[perm, :, self.right_slice]
        else:
            negative = negative + 0.25 * torch.randn_like(negative)
        negative_energy = self.action_energy(negative, global_cond)
        loss = F.softplus(positive_energy - negative_energy + self.energy_margin).mean()
        return loss, {
            "action_energy_positive": positive_energy.mean(),
            "action_energy_negative": negative_energy.mean(),
        }

    def _estimate_coupling(self, normalized_actions: torch.Tensor, global_cond: torch.Tensor) -> torch.Tensor:
        if self.coupling_estimator is None:
            return torch.zeros(
                normalized_actions.shape[0],
                normalized_actions.shape[1],
                2,
                device=normalized_actions.device,
                dtype=normalized_actions.dtype,
            )
        timesteps = torch.ones(normalized_actions.shape[0], device=normalized_actions.device, dtype=normalized_actions.dtype)
        return self.coupling_estimator.estimate(
            normalized_actions,
            timesteps,
            global_cond,
            self.left_slice,
            self.right_slice,
        )

    def _score_action(
        self,
        normalized_actions: torch.Tensor,
        global_cond: torch.Tensor,
        obs_features: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        energy_scores = self.action_energy(normalized_actions, global_cond)
        if self.action_world_model is None:
            wm_scores = torch.zeros_like(energy_scores)
        else:
            wm_scores = self.action_world_model.score(obs_features, normalized_actions)
        coupling_scores = self._estimate_coupling(normalized_actions, global_cond)
        schedule = self.compact_scheduler.search(
            normalized_actions,
            coupling_scores,
            energy_scores,
            wm_scores,
            self.left_slice,
            self.right_slice,
        )
        schedule["energy_scores"] = energy_scores
        schedule["wm_scores"] = wm_scores
        schedule["coupling_scores"] = coupling_scores
        return schedule

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        assert "past_action" not in obs_dict
        nobs = self.normalizer.normalize(obs_dict)
        value = next(iter(nobs.values()))
        batch_size = value.shape[0]
        obs_features = self._encode_obs_sequence(nobs)
        global_cond = self._encode_global_cond(obs_features)

        samples = []
        scored = []
        num_candidates = max(self.action_rerank_samples, 1)
        for _ in range(num_candidates):
            sample = self._sample_action(batch_size, global_cond)
            samples.append(sample)
            scored.append(self._score_action(sample, global_cond, obs_features))

        action_stack = torch.stack(samples, dim=0)
        score_stack = torch.stack([item["total_cost"] for item in scored], dim=0)
        best_idx = score_stack.argmin(dim=0)
        batch_idx = torch.arange(batch_size, device=action_stack.device)
        best_normalized_action = action_stack[best_idx, batch_idx]
        self.last_candidate_scores = score_stack.detach()

        compact_schedule = torch.stack([item["schedule"] for item in scored], dim=0)[best_idx, batch_idx]
        makespan = torch.stack([item["makespan"] for item in scored], dim=0)[best_idx, batch_idx]
        coupling_scores = torch.stack([item["coupling_scores"] for item in scored], dim=0)[best_idx, batch_idx]
        wm_scores = torch.stack([item["wm_scores"] for item in scored], dim=0)[best_idx, batch_idx]
        energy_scores = torch.stack([item["energy_scores"] for item in scored], dim=0)[best_idx, batch_idx]
        dependency_cost = torch.stack([item["dependency_cost"] for item in scored], dim=0)[best_idx, batch_idx]
        dynamics_cost = torch.stack([item["dynamics_cost"] for item in scored], dim=0)[best_idx, batch_idx]
        collision_or_risk_cost = torch.stack([item["risk_cost"] for item in scored], dim=0)[best_idx, batch_idx]
        self.last_compact_schedule = compact_schedule.detach()
        self.last_coupling_scores = coupling_scores.detach()

        action_pred = self.normalizer["action"].unnormalize(best_normalized_action)
        start = self.n_obs_steps - 1
        end = start + self.n_action_steps
        return {
            "action": action_pred[:, start:end],
            "action_pred": action_pred,
            "compact_schedule": compact_schedule,
            "makespan": makespan,
            "coupling_scores": coupling_scores,
            "candidate_scores": score_stack.detach(),
            "wm_scores": wm_scores,
            "energy_scores": energy_scores,
            "dependency_cost": dependency_cost,
            "dynamics_cost": dynamics_cost,
            "collision_or_risk_cost": collision_or_risk_cost,
        }

    def compute_loss(self, batch):
        assert "valid_mask" not in batch
        nobs = self.normalizer.normalize(batch["obs"])
        nactions = self.normalizer["action"].normalize(batch["action"])
        batch_size = nactions.shape[0]

        obs_features = self._encode_obs_sequence(nobs)
        global_cond = self._encode_global_cond(obs_features)

        noise = torch.randn_like(nactions)
        timesteps = torch.rand((batch_size,), device=nactions.device, dtype=nactions.dtype)
        t_view = timesteps.view(batch_size, 1, 1)
        noisy_actions = (1.0 - t_view) * noise + t_view * nactions
        target_velocity = nactions - noise
        pred_velocity = self.model(noisy_actions, timesteps, global_cond=global_cond)
        flow_loss = reduce(
            F.mse_loss(pred_velocity, target_velocity, reduction="none"),
            "b ... -> b (...)",
            "mean",
        ).mean()
        total_loss = flow_loss

        extra_logs = {"flow_matching_loss": flow_loss}
        if self.coupling_estimator is not None and self.coupling_aux_loss_weight > 0:
            coupling_loss, coupling_logs = self.coupling_estimator.compute_aux_loss(
                noisy_actions[:, :, self.left_slice],
                noisy_actions[:, :, self.right_slice],
                timesteps,
                global_cond,
                target_velocity[:, :, self.left_slice],
                target_velocity[:, :, self.right_slice],
            )
            total_loss = total_loss + self.coupling_aux_loss_weight * coupling_loss
            extra_logs.update(coupling_logs)

        energy_loss, energy_logs = self._energy_contrastive_loss(nactions, global_cond)
        total_loss = total_loss + self.energy_loss_weight * energy_loss
        extra_logs["energy_contrastive_loss"] = energy_loss
        extra_logs.update(energy_logs)

        if self.action_world_model is not None and self.world_model_loss_weight > 0:
            world_loss, world_logs = self.action_world_model.compute_loss(
                obs_features,
                nactions,
                gaussian_weight=self.world_model_gaussian_weight,
            )
            total_loss = total_loss + self.world_model_loss_weight * world_loss
            extra_logs["world_model_loss"] = world_loss
            extra_logs.update(world_logs)

        self.last_loss_dict = {
            key: float(value.detach().cpu())
            for key, value in extra_logs.items()
        }
        return total_loss
