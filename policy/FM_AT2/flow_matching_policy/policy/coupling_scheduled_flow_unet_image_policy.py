import math
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


class PreferenceEnergyHead(nn.Module):
    """Decomposed preference energy over action chunks and compact schedules."""

    component_names = (
        "demo_energy",
        "compactness_energy",
        "dag_energy",
        "dynamics_energy",
        "phase_energy",
    )

    def __init__(
        self,
        action_dim: int,
        cond_dim: int,
        hidden_dim: int = 256,
        demo_weight: float = 1.0,
        compactness_weight: float = 1.0,
        dag_weight: float = 1.0,
        dynamics_weight: float = 1.0,
        phase_weight: float = 1.0,
    ):
        super().__init__()
        self.weights = {
            "demo_energy": float(demo_weight),
            "compactness_energy": float(compactness_weight),
            "dag_energy": float(dag_weight),
            "dynamics_energy": float(dynamics_weight),
            "phase_energy": float(phase_weight),
        }
        # action mean/std + global cond + schedule/coupling/scalar summary features
        input_dim = int(action_dim) * 2 + int(cond_dim) + 36
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, len(self.component_names)),
        )

    @staticmethod
    def _stats(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        if x is None:
            return ref.new_zeros(ref.shape[0], 8)
        return torch.cat(
            [
                x.mean(dim=1),
                x.std(dim=1, unbiased=False),
                x.max(dim=1).values,
                x.min(dim=1).values,
            ],
            dim=-1,
        )

    @staticmethod
    def _scalar_feature(
        dag_features: Dict[str, torch.Tensor],
        key: str,
        ref: torch.Tensor,
    ) -> torch.Tensor:
        if dag_features is None or key not in dag_features:
            return ref.new_zeros(ref.shape[0], 1)
        value = dag_features[key]
        if value.dim() == 0:
            value = value.view(1).expand(ref.shape[0])
        if value.dim() > 1:
            value = value.reshape(value.shape[0], -1).mean(dim=-1)
        return value.view(ref.shape[0], 1).to(device=ref.device, dtype=ref.dtype)

    def forward(
        self,
        actions: torch.Tensor,
        global_cond: torch.Tensor,
        schedule: torch.Tensor = None,
        durations: torch.Tensor = None,
        speed_scale: torch.Tensor = None,
        coupling_scores: torch.Tensor = None,
        wm_scores: torch.Tensor = None,
        dag_features: Dict[str, torch.Tensor] = None,
        phase_scores: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        batch_size = actions.shape[0]
        if schedule is None:
            schedule = actions.new_zeros(batch_size, actions.shape[1], 2)
        if durations is None:
            durations = actions.new_zeros(batch_size, actions.shape[1], 2)
        if speed_scale is None:
            speed_scale = actions.new_zeros(batch_size, actions.shape[1], 2)
        if coupling_scores is None:
            coupling_scores = actions.new_zeros(batch_size, actions.shape[1], 2)

        action_features = torch.cat(
            [
                actions.mean(dim=1),
                actions.std(dim=1, unbiased=False),
            ],
            dim=-1,
        )
        coupling_features = torch.cat(
            [
                coupling_scores.mean(dim=1),
                coupling_scores.max(dim=1).values,
            ],
            dim=-1,
        )
        scalar_keys = (
            "makespan",
            "dependency_cost",
            "dynamics_cost",
            "dag_precedence_cost",
            "dag_sync_cost",
            "dag_critical_cost",
        )
        scalar_features = [self._scalar_feature(dag_features, key, actions) for key in scalar_keys]
        if wm_scores is None:
            scalar_features.append(actions.new_zeros(batch_size, 1))
        else:
            scalar_features.append(wm_scores.reshape(batch_size, -1).mean(dim=-1, keepdim=True))
        if phase_scores is None:
            scalar_features.append(actions.new_zeros(batch_size, 1))
        else:
            scalar_features.append(phase_scores.reshape(batch_size, -1).mean(dim=-1, keepdim=True))

        features = torch.cat(
            [
                action_features,
                global_cond,
                self._stats(schedule, actions),
                self._stats(durations, actions),
                self._stats(speed_scale, actions),
                coupling_features,
                torch.cat(scalar_features, dim=-1),
            ],
            dim=-1,
        )
        raw = self.net(features)
        result = {
            name: raw[:, idx]
            for idx, name in enumerate(self.component_names)
        }
        total = actions.new_zeros(batch_size)
        for name in self.component_names:
            total = total + self.weights[name] * result[name]
        result["total_energy"] = total
        return result


class DiagonalGaussianActionHead(nn.Module):
    """Per-step diagonal Gaussian action distribution head."""

    def __init__(
        self,
        global_cond_dim: int,
        action_dim: int,
        other_action_dim: int = 0,
        hidden_dim: int = 256,
        logvar_min: float = -10.0,
        logvar_max: float = 5.0,
    ):
        super().__init__()
        self.action_dim = int(action_dim)
        self.other_action_dim = int(other_action_dim)
        self.logvar_min = float(logvar_min)
        self.logvar_max = float(logvar_max)
        input_dim = int(global_cond_dim) + self.other_action_dim + 2
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.action_dim * 2),
        )

    def _phase_features(self, batch_size: int, horizon: int, ref: torch.Tensor) -> torch.Tensor:
        if horizon <= 1:
            phase = torch.zeros(1, horizon, 1, device=ref.device, dtype=ref.dtype)
        else:
            phase = torch.arange(horizon, device=ref.device, dtype=ref.dtype).view(1, horizon, 1)
            phase = phase / float(horizon - 1)
        return torch.cat([phase, 1.0 - phase], dim=-1).expand(batch_size, horizon, 2)

    def forward(
        self,
        global_cond: torch.Tensor,
        horizon: int,
        other_actions: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        batch_size = global_cond.shape[0]
        global_context = global_cond.unsqueeze(1).expand(batch_size, horizon, global_cond.shape[-1])
        phase = self._phase_features(batch_size, horizon, global_cond)
        if self.other_action_dim > 0:
            if other_actions is None:
                other_actions = global_cond.new_zeros(batch_size, horizon, self.other_action_dim)
            other_context = other_actions
        else:
            other_context = global_cond.new_zeros(batch_size, horizon, 0)
        params = self.net(torch.cat([global_context, phase, other_context], dim=-1))
        mu, logvar = params.chunk(2, dim=-1)
        logvar = logvar.clamp(self.logvar_min, self.logvar_max)
        return {"mu": mu, "logvar": logvar}


class BimanualCouplingEstimator(nn.Module):
    """Auxiliary marginal/conditional action distributions for learned arm dependency."""

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
        hidden_dim = int(down_dims[0]) if len(down_dims) > 0 else 256
        self.left_marginal_head = DiagonalGaussianActionHead(
            global_cond_dim=global_cond_dim,
            action_dim=self.left_action_dim,
            hidden_dim=hidden_dim,
        )
        self.right_marginal_head = DiagonalGaussianActionHead(
            global_cond_dim=global_cond_dim,
            action_dim=self.right_action_dim,
            hidden_dim=hidden_dim,
        )
        self.left_conditional_head = DiagonalGaussianActionHead(
            global_cond_dim=global_cond_dim,
            action_dim=self.left_action_dim,
            other_action_dim=self.right_action_dim,
            hidden_dim=hidden_dim,
        )
        self.right_conditional_head = DiagonalGaussianActionHead(
            global_cond_dim=global_cond_dim,
            action_dim=self.right_action_dim,
            other_action_dim=self.left_action_dim,
            hidden_dim=hidden_dim,
        )

    @staticmethod
    def _gaussian_nll(params: Dict[str, torch.Tensor], target: torch.Tensor) -> torch.Tensor:
        mu = params["mu"]
        logvar = params["logvar"]
        inv_var = torch.exp(-logvar)
        return 0.5 * (math.log(2.0 * math.pi) + logvar + (target - mu).pow(2) * inv_var).mean(dim=-1)

    @staticmethod
    def _gaussian_kl(q_params: Dict[str, torch.Tensor], p_params: Dict[str, torch.Tensor]) -> torch.Tensor:
        q_mu = q_params["mu"]
        q_logvar = q_params["logvar"]
        p_mu = p_params["mu"]
        p_logvar = p_params["logvar"]
        kl = 0.5 * (
            p_logvar
            - q_logvar
            + (torch.exp(q_logvar) + (q_mu - p_mu).pow(2)) * torch.exp(-p_logvar)
            - 1.0
        )
        return kl.mean(dim=-1).clamp_min(0.0)

    def forward(
        self,
        left_actions: torch.Tensor,
        right_actions: torch.Tensor,
        global_cond: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        horizon = left_actions.shape[1]
        left_marginal = self.left_marginal_head(global_cond, horizon)
        right_marginal = self.right_marginal_head(global_cond, horizon)
        left_cond = self.left_conditional_head(global_cond, horizon, other_actions=right_actions)
        right_cond = self.right_conditional_head(global_cond, horizon, other_actions=left_actions)
        return {
            "left_marginal": left_marginal,
            "right_marginal": right_marginal,
            "left_cond": left_cond,
            "right_cond": right_cond,
        }

    def compute_aux_loss(
        self,
        left_actions: torch.Tensor,
        right_actions: torch.Tensor,
        global_cond: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        pred = self(left_actions, right_actions, global_cond)
        left_marginal_per = self._gaussian_nll(pred["left_marginal"], left_actions)
        right_marginal_per = self._gaussian_nll(pred["right_marginal"], right_actions)
        left_cond_per = self._gaussian_nll(pred["left_cond"], left_actions)
        right_cond_per = self._gaussian_nll(pred["right_cond"], right_actions)
        aux_loss = 0.25 * (
            left_marginal_per.mean()
            + right_marginal_per.mean()
            + left_cond_per.mean()
            + right_cond_per.mean()
        )
        coupling_l_to_r = self._gaussian_kl(pred["right_cond"], pred["right_marginal"])
        coupling_r_to_l = self._gaussian_kl(pred["left_cond"], pred["left_marginal"])
        logs = {
            "coupling_aux_loss": aux_loss,
            "left_marginal_nll": left_marginal_per.mean(),
            "right_marginal_nll": right_marginal_per.mean(),
            "left_conditional_nll": left_cond_per.mean(),
            "right_conditional_nll": right_cond_per.mean(),
            "coupling_kl_l_to_r": coupling_l_to_r.mean(),
            "coupling_kl_r_to_l": coupling_r_to_l.mean(),
        }
        return aux_loss, logs

    def estimate(
        self,
        normalized_actions: torch.Tensor,
        global_cond: torch.Tensor,
        left_slice: slice,
        right_slice: slice,
    ) -> torch.Tensor:
        left = normalized_actions[:, :, left_slice]
        right = normalized_actions[:, :, right_slice]
        pred = self(left, right, global_cond)
        coupling_l_to_r = self._gaussian_kl(pred["right_cond"], pred["right_marginal"])
        coupling_r_to_l = self._gaussian_kl(pred["left_cond"], pred["left_marginal"])
        return torch.stack([coupling_l_to_r, coupling_r_to_l], dim=-1)


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

    # 修改：serach可能也要该，目前是软惩罚搜索，可能要改成利用DAG进行搜索
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
            base_dt = actions.new_tensor(1.0 / max(horizon, 1))
            durations = torch.ones((batch_size, horizon, 2), device=device, dtype=dtype) * base_dt
            ends = schedule + durations
            speed_scale = torch.ones_like(durations)
            makespan = torch.ones(batch_size, device=device, dtype=dtype)
            zero = torch.zeros(batch_size, device=device, dtype=dtype)
            total = self.risk_weight * risk + self.lambda_time * makespan
            return {
                "schedule": schedule,
                "durations": durations,
                "ends": ends,
                "speed_scale": speed_scale,
                "makespan": makespan,
                "dependency_cost": zero,
                "dynamics_cost": zero,
                "risk_cost": risk,
                "total_cost": total,
                "dag_precedence_cost": zero,
                "dag_sync_cost": zero,
                "dag_critical_cost": zero,
                "dag_dependency_cost": zero,
                "dag_edge_count": zero,
                "dag_slack": torch.zeros(batch_size, horizon, 2, device=device, dtype=dtype),
                "dag_criticality": torch.zeros(batch_size, horizon, 2, device=device, dtype=dtype),
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
        best_starts = starts[best_idx, batch_idx]
        best_durations = durations[best_idx, batch_idx]
        best_ends = best_starts + best_durations
        best_speed_scale = base_dt / best_durations.clamp_min(1.0e-4)
        return {
            "schedule": best_starts,
            "durations": best_durations,
            "ends": best_ends,
            "speed_scale": best_speed_scale,
            "makespan": makespan[best_idx, batch_idx],
            "dependency_cost": dependency_cost[best_idx, batch_idx],
            "dynamics_cost": dynamics_cost[best_idx, batch_idx],
            "risk_cost": risk,
            "total_cost": total[best_idx, batch_idx],
            "dag_precedence_cost": dependency_cost[best_idx, batch_idx],
            "dag_sync_cost": torch.zeros(batch_size, device=device, dtype=dtype),
            "dag_critical_cost": torch.zeros(batch_size, device=device, dtype=dtype),
            "dag_dependency_cost": dependency_cost[best_idx, batch_idx],
            "dag_edge_count": torch.zeros(batch_size, device=device, dtype=dtype),
            "dag_slack": torch.zeros(batch_size, horizon, 2, device=device, dtype=dtype),
            "dag_criticality": torch.zeros(batch_size, horizon, 2, device=device, dtype=dtype),
        }


class DAGCompactScheduler(CompactScheduler):
    """DAG-style compact schedule search with precedence and sync costs."""

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
        coupling_threshold: float = 0.05,
        direction_margin: float = 0.02,
        precedence_fraction: float = 1.0,
        sync_weight: float = 1.0,
        critical_weight: float = 0.5,
        critical_tau: float = 0.05,
    ):
        super().__init__(
            enabled=enabled,
            num_samples=num_samples,
            lambda_time=lambda_time,
            dependency_weight=dependency_weight,
            dynamics_weight=dynamics_weight,
            risk_weight=risk_weight,
            min_duration_scale=min_duration_scale,
            max_duration_scale=max_duration_scale,
            max_offset_scale=max_offset_scale,
        )
        self.coupling_threshold = float(coupling_threshold)
        self.direction_margin = float(direction_margin)
        self.precedence_fraction = float(precedence_fraction)
        self.sync_weight = float(sync_weight)
        self.critical_weight = float(critical_weight)
        self.critical_tau = max(float(critical_tau), 1.0e-6)

    def _edge_masks(self, coupling_scores: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        l_to_r = coupling_scores[..., 0]
        r_to_l = coupling_scores[..., 1]
        lr_edge = (l_to_r > self.coupling_threshold) & (l_to_r > r_to_l + self.direction_margin)
        rl_edge = (r_to_l > self.coupling_threshold) & (r_to_l > l_to_r + self.direction_margin)
        sync_edge = (
            (l_to_r > self.coupling_threshold)
            & (r_to_l > self.coupling_threshold)
            & ((l_to_r - r_to_l).abs() <= self.direction_margin)
        )
        return lr_edge, rl_edge, sync_edge

    def _linear_fallback(
        self,
        actions: torch.Tensor,
        risk: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        batch_size, horizon = actions.shape[:2]
        device = actions.device
        dtype = actions.dtype
        schedule = self._linear_schedule(batch_size, horizon, device, dtype)
        base_dt = actions.new_tensor(1.0 / max(horizon, 1))
        durations = torch.ones((batch_size, horizon, 2), device=device, dtype=dtype) * base_dt
        ends = schedule + durations
        speed_scale = torch.ones_like(durations)
        makespan = torch.ones(batch_size, device=device, dtype=dtype)
        zero = torch.zeros(batch_size, device=device, dtype=dtype)
        total = self.risk_weight * risk + self.lambda_time * makespan
        return {
            "schedule": schedule,
            "durations": durations,
            "ends": ends,
            "speed_scale": speed_scale,
            "makespan": makespan,
            "dependency_cost": zero,
            "dynamics_cost": zero,
            "risk_cost": risk,
            "total_cost": total,
            "dag_precedence_cost": zero,
            "dag_sync_cost": zero,
            "dag_critical_cost": zero,
            "dag_dependency_cost": zero,
            "dag_edge_count": torch.full_like(zero, max(horizon - 1, 0) * 2.0),
            "dag_slack": torch.zeros(batch_size, horizon, 2, device=device, dtype=dtype),
            "dag_criticality": torch.zeros(batch_size, horizon, 2, device=device, dtype=dtype),
        }

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
            return self._linear_fallback(actions, risk)

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
        left_duration = durations[..., 0]
        right_duration = durations[..., 1]
        left_end = left_start + left_duration
        right_end = right_start + right_duration
        makespan = torch.maximum(left_end[:, :, -1], right_end[:, :, -1])

        lr_edge, rl_edge, sync_edge = self._edge_masks(coupling_scores)
        lr_weight = (coupling_scores[..., 0] * lr_edge.to(dtype)).unsqueeze(0)
        rl_weight = (coupling_scores[..., 1] * rl_edge.to(dtype)).unsqueeze(0)
        sync_weight = (
            torch.minimum(coupling_scores[..., 0], coupling_scores[..., 1])
            * sync_edge.to(dtype)
            * self.sync_weight
        ).unsqueeze(0)

        lr_slack = right_start - (left_start + self.precedence_fraction * left_duration)
        rl_slack = left_start - (right_start + self.precedence_fraction * right_duration)
        lr_violation = F.relu(-lr_slack)
        rl_violation = F.relu(-rl_slack)
        precedence_per = lr_weight * lr_violation + rl_weight * rl_violation
        precedence_cost = precedence_per.mean(dim=-1)

        sync_per = sync_weight * (left_start - right_start).abs()
        sync_cost = sync_per.mean(dim=-1)

        lr_criticality = torch.exp(-F.relu(lr_slack) / self.critical_tau) * lr_edge.to(dtype).unsqueeze(0)
        rl_criticality = torch.exp(-F.relu(rl_slack) / self.critical_tau) * rl_edge.to(dtype).unsqueeze(0)
        critical_per = lr_weight * lr_criticality * lr_violation + rl_weight * rl_criticality * rl_violation
        critical_cost = critical_per.mean(dim=-1) * self.critical_weight
        dag_dependency_cost = precedence_cost + sync_cost + critical_cost

        left_actions = actions[:, :, left_slice]
        right_actions = actions[:, :, right_slice]
        left_delta = torch.cat([left_actions[:, :1] * 0.0, left_actions[:, 1:] - left_actions[:, :-1]], dim=1)
        right_delta = torch.cat([right_actions[:, :1] * 0.0, right_actions[:, 1:] - right_actions[:, :-1]], dim=1)
        left_motion = left_delta.pow(2).mean(dim=-1).unsqueeze(0)
        right_motion = right_delta.pow(2).mean(dim=-1).unsqueeze(0)
        dynamics_cost = (
            left_motion / left_duration.clamp_min(1.0e-4)
            + right_motion / right_duration.clamp_min(1.0e-4)
        ).mean(dim=-1)

        risk_cost = risk.unsqueeze(0).expand(num, -1)
        total = (
            self.lambda_time * makespan
            + self.dependency_weight * dag_dependency_cost
            + self.dynamics_weight * dynamics_cost
            + self.risk_weight * risk_cost
        )
        best_idx = total.argmin(dim=0)
        batch_idx = torch.arange(batch_size, device=device)
        best_starts = starts[best_idx, batch_idx]
        best_durations = durations[best_idx, batch_idx]
        best_ends = best_starts + best_durations
        best_speed_scale = base_dt / best_durations.clamp_min(1.0e-4)
        best_lr_slack = lr_slack[best_idx, batch_idx]
        best_rl_slack = rl_slack[best_idx, batch_idx]
        best_lr_crit = lr_criticality[best_idx, batch_idx]
        best_rl_crit = rl_criticality[best_idx, batch_idx]
        edge_count = (
            lr_edge.to(dtype).sum(dim=-1)
            + rl_edge.to(dtype).sum(dim=-1)
            + sync_edge.to(dtype).sum(dim=-1)
            + float(max(horizon - 1, 0) * 2)
        )
        return {
            "schedule": best_starts,
            "durations": best_durations,
            "ends": best_ends,
            "speed_scale": best_speed_scale,
            "makespan": makespan[best_idx, batch_idx],
            "dependency_cost": dag_dependency_cost[best_idx, batch_idx],
            "dynamics_cost": dynamics_cost[best_idx, batch_idx],
            "risk_cost": risk,
            "total_cost": total[best_idx, batch_idx],
            "dag_precedence_cost": precedence_cost[best_idx, batch_idx],
            "dag_sync_cost": sync_cost[best_idx, batch_idx],
            "dag_critical_cost": critical_cost[best_idx, batch_idx],
            "dag_dependency_cost": dag_dependency_cost[best_idx, batch_idx],
            "dag_edge_count": edge_count,
            "dag_slack": torch.stack([F.relu(best_lr_slack), F.relu(best_rl_slack)], dim=-1),
            "dag_criticality": torch.stack([best_lr_crit, best_rl_crit], dim=-1),
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
        scheduler_mode="dag",
        dag_coupling_threshold=0.05,
        dag_direction_margin=0.02,
        dag_precedence_fraction=1.0,
        dag_sync_weight=1.0,
        dag_critical_weight=0.5,
        dag_critical_tau=0.05,
        action_rerank_samples=4,
        energy_loss_weight=0.01,
        energy_hidden_dim=256,
        energy_margin=1.0,
        preference_improve_margin=0.1,
        preference_bad_margin=1.0,
        preference_demo_weight=1.0,
        preference_compactness_weight=1.0,
        preference_dag_weight=1.0,
        preference_dynamics_weight=1.0,
        preference_phase_weight=1.0,
        reward_weight_temperature=1.0,
        rl_self_imitation_weight=0.1,
        rl_ppo_reranker_weight=0.05,
        rl_bc_anchor_weight=1.0,
        rl_ppo_clip=0.2,
        rl_candidate_temperature=1.0,
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

        # flow matching policy model
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
        self.preference_improve_margin = float(preference_improve_margin)
        self.preference_bad_margin = float(preference_bad_margin)
        self.world_model_enabled = bool(world_model_enabled)
        self.world_model_loss_weight = float(world_model_loss_weight)
        self.world_model_gaussian_weight = float(world_model_gaussian_weight)
        self.action_rerank_samples = int(action_rerank_samples)
        self.reward_weight_temperature = max(float(reward_weight_temperature), 1.0e-6)
        self.rl_self_imitation_weight = float(rl_self_imitation_weight)
        self.rl_ppo_reranker_weight = float(rl_ppo_reranker_weight)
        self.rl_bc_anchor_weight = float(rl_bc_anchor_weight)
        self.rl_ppo_clip = float(rl_ppo_clip)
        self.rl_candidate_temperature = max(float(rl_candidate_temperature), 1.0e-6)
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

        self.preference_energy = PreferenceEnergyHead(
            action_dim=action_dim,
            cond_dim=global_cond_dim,
            hidden_dim=int(energy_hidden_dim),
            demo_weight=preference_demo_weight,
            compactness_weight=preference_compactness_weight,
            dag_weight=preference_dag_weight,
            dynamics_weight=preference_dynamics_weight,
            phase_weight=preference_phase_weight,
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

        scheduler_cls = DAGCompactScheduler if str(scheduler_mode).lower() == "dag" else CompactScheduler
        scheduler_kwargs = dict(
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
        if scheduler_cls is DAGCompactScheduler:
            scheduler_kwargs.update(
                coupling_threshold=dag_coupling_threshold,
                direction_margin=dag_direction_margin,
                precedence_fraction=dag_precedence_fraction,
                sync_weight=dag_sync_weight,
                critical_weight=dag_critical_weight,
                critical_tau=dag_critical_tau,
            )
        self.compact_scheduler = scheduler_cls(**scheduler_kwargs)

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

    def _linear_schedule_result(
        self,
        actions: torch.Tensor,
        risk: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        batch_size, horizon = actions.shape[:2]
        base_dt = actions.new_tensor(1.0 / max(horizon, 1))
        base = torch.arange(horizon, device=actions.device, dtype=actions.dtype) / max(horizon, 1)
        schedule = base.view(1, horizon, 1).expand(batch_size, horizon, 2)
        durations = torch.ones((batch_size, horizon, 2), device=actions.device, dtype=actions.dtype) * base_dt
        zero = torch.zeros(batch_size, device=actions.device, dtype=actions.dtype)
        if risk is None:
            risk = zero
        return {
            "schedule": schedule,
            "durations": durations,
            "ends": schedule + durations,
            "speed_scale": torch.ones_like(durations),
            "makespan": torch.ones(batch_size, device=actions.device, dtype=actions.dtype),
            "dependency_cost": zero,
            "dynamics_cost": zero,
            "risk_cost": risk,
            "total_cost": risk,
            "dag_precedence_cost": zero,
            "dag_sync_cost": zero,
            "dag_critical_cost": zero,
            "dag_dependency_cost": zero,
            "dag_edge_count": torch.full_like(zero, max(horizon - 1, 0) * 2.0),
            "dag_slack": torch.zeros(batch_size, horizon, 2, device=actions.device, dtype=actions.dtype),
            "dag_criticality": torch.zeros(batch_size, horizon, 2, device=actions.device, dtype=actions.dtype),
        }

    def _run_preference_energy(
        self,
        actions: torch.Tensor,
        global_cond: torch.Tensor,
        schedule: Dict[str, torch.Tensor],
        coupling_scores: torch.Tensor,
        wm_scores: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        return self.preference_energy(
            actions,
            global_cond,
            schedule=schedule.get("schedule"),
            durations=schedule.get("durations"),
            speed_scale=schedule.get("speed_scale"),
            coupling_scores=coupling_scores,
            wm_scores=wm_scores,
            dag_features=schedule,
        )

    def _make_bad_schedule(self, schedule: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        bad = {}
        for key, value in schedule.items():
            bad[key] = value.clone() if torch.is_tensor(value) else value
        if "durations" in bad:
            bad["durations"] = bad["durations"].clamp_min(1.0e-4) * 0.5
        if "speed_scale" in bad:
            bad["speed_scale"] = bad["speed_scale"] * 2.0
        if "makespan" in bad:
            bad["makespan"] = bad["makespan"] * 0.5
        for key in ("dependency_cost", "dag_dependency_cost", "dag_precedence_cost", "dag_sync_cost"):
            if key in bad:
                bad[key] = bad[key] + 1.0
        return bad

    def _preference_energy_loss(
        self,
        expert_actions: torch.Tensor,
        global_cond: torch.Tensor,
        obs_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        with torch.no_grad():
            coupling_scores = self._estimate_coupling(expert_actions, global_cond)
            if self.action_world_model is None:
                wm_scores = torch.zeros(expert_actions.shape[0], device=expert_actions.device, dtype=expert_actions.dtype)
            else:
                wm_scores = self.action_world_model.score(obs_features, expert_actions)
            zero_energy = torch.zeros_like(wm_scores)
            expert_schedule = self._linear_schedule_result(expert_actions)
            improved_schedule = self.compact_scheduler.search(
                expert_actions,
                coupling_scores,
                zero_energy,
                wm_scores,
                self.left_slice,
                self.right_slice,
            )
            bad_actions = expert_actions.clone()
            if expert_actions.shape[0] > 1:
                perm = torch.randperm(expert_actions.shape[0], device=expert_actions.device)
                bad_actions[:, :, self.right_slice] = expert_actions[perm, :, self.right_slice]
            else:
                bad_actions = bad_actions + 0.25 * torch.randn_like(bad_actions)
            bad_coupling = self._estimate_coupling(bad_actions, global_cond)
            bad_wm_scores = wm_scores
            bad_schedule = self._make_bad_schedule(improved_schedule)

        expert_energy = self._run_preference_energy(
            expert_actions,
            global_cond,
            expert_schedule,
            coupling_scores,
            wm_scores,
        )
        improved_energy = self._run_preference_energy(
            expert_actions,
            global_cond,
            improved_schedule,
            coupling_scores,
            wm_scores,
        )
        bad_energy = self._run_preference_energy(
            bad_actions,
            global_cond,
            bad_schedule,
            bad_coupling,
            bad_wm_scores,
        )
        improve_loss = F.softplus(
            improved_energy["total_energy"] - expert_energy["total_energy"] + self.preference_improve_margin
        ).mean()
        bad_loss = F.softplus(
            expert_energy["total_energy"] - bad_energy["total_energy"] + self.preference_bad_margin
        ).mean()
        loss = improve_loss + bad_loss
        logs = {
            "preference_energy_loss": loss,
            "preference_improve_loss": improve_loss,
            "preference_bad_loss": bad_loss,
            "preference_expert_energy": expert_energy["total_energy"].mean(),
            "preference_improved_energy": improved_energy["total_energy"].mean(),
            "preference_bad_energy": bad_energy["total_energy"].mean(),
            "preference_demo_energy": expert_energy["demo_energy"].mean(),
            "preference_compactness_energy": expert_energy["compactness_energy"].mean(),
            "preference_dag_energy": expert_energy["dag_energy"].mean(),
            "preference_dynamics_energy": expert_energy["dynamics_energy"].mean(),
            "preference_phase_energy": expert_energy["phase_energy"].mean(),
            "dag_precedence_cost": improved_schedule["dag_precedence_cost"].mean(),
            "dag_sync_cost": improved_schedule["dag_sync_cost"].mean(),
            "dag_critical_cost": improved_schedule["dag_critical_cost"].mean(),
            "dag_dependency_cost": improved_schedule["dag_dependency_cost"].mean(),
            "dag_edge_count": improved_schedule["dag_edge_count"].mean(),
        }
        return loss, logs

    def _estimate_coupling(self, normalized_actions: torch.Tensor, global_cond: torch.Tensor) -> torch.Tensor:
        if self.coupling_estimator is None:
            return torch.zeros(
                normalized_actions.shape[0],
                normalized_actions.shape[1],
                2,
                device=normalized_actions.device,
                dtype=normalized_actions.dtype,
            )
        return self.coupling_estimator.estimate(
            normalized_actions,
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
        coupling_scores = self._estimate_coupling(normalized_actions, global_cond)
        if self.action_world_model is None:
            wm_scores = torch.zeros(normalized_actions.shape[0], device=normalized_actions.device, dtype=normalized_actions.dtype)
        else:
            wm_scores = self.action_world_model.score(obs_features, normalized_actions)
        initial_energy = torch.zeros_like(wm_scores)
        schedule = self.compact_scheduler.search(
            normalized_actions,
            coupling_scores,
            initial_energy,
            wm_scores,
            self.left_slice,
            self.right_slice,
        )
        preference = self._run_preference_energy(
            normalized_actions,
            global_cond,
            schedule,
            coupling_scores,
            wm_scores,
        )
        preference_total = preference["total_energy"]
        total_cost = (
            preference_total
            + self.compact_scheduler.risk_weight * F.softplus(wm_scores)
            + self.compact_scheduler.lambda_time * schedule["makespan"]
            + self.compact_scheduler.dependency_weight * schedule["dag_dependency_cost"]
            + self.compact_scheduler.dynamics_weight * schedule["dynamics_cost"]
        )
        schedule["energy_scores"] = preference_total
        schedule["preference_energy_components"] = preference
        schedule["wm_scores"] = wm_scores
        schedule["coupling_scores"] = coupling_scores
        schedule["risk_cost"] = F.softplus(preference_total) + F.softplus(wm_scores)
        schedule["total_cost"] = total_cost
        return schedule

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def predict_action(
        self,
        obs_dict: Dict[str, torch.Tensor],
        stochastic_select: bool = False,
        return_candidate_batch: bool = False,
        rl_mode: bool = False,
    ) -> Dict[str, torch.Tensor]:
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
        candidate_logits = (-score_stack.transpose(0, 1)) / self.rl_candidate_temperature
        candidate_log_probs = F.log_softmax(candidate_logits, dim=-1)
        if stochastic_select:
            selected_idx = torch.distributions.Categorical(logits=candidate_logits).sample()
            selected_logprob = candidate_log_probs.gather(1, selected_idx.view(-1, 1)).squeeze(1)
            best_idx = selected_idx
        else:
            best_idx = score_stack.argmin(dim=0)
            selected_logprob = candidate_log_probs.gather(1, best_idx.view(-1, 1)).squeeze(1)
        batch_idx = torch.arange(batch_size, device=action_stack.device)
        best_normalized_action = action_stack[best_idx, batch_idx]
        self.last_candidate_scores = score_stack.detach()

        def select_stacked(key: str) -> torch.Tensor:
            return torch.stack([item[key] for item in scored], dim=0)[best_idx, batch_idx]

        compact_schedule = select_stacked("schedule")
        compact_schedule_durations = select_stacked("durations")
        compact_schedule_ends = select_stacked("ends")
        compact_schedule_speed_scale = select_stacked("speed_scale")
        makespan = select_stacked("makespan")
        coupling_scores = select_stacked("coupling_scores")
        wm_scores = select_stacked("wm_scores")
        energy_scores = select_stacked("energy_scores")
        dependency_cost = select_stacked("dependency_cost")
        dynamics_cost = select_stacked("dynamics_cost")
        collision_or_risk_cost = select_stacked("risk_cost")
        dag_precedence_cost = select_stacked("dag_precedence_cost")
        dag_sync_cost = select_stacked("dag_sync_cost")
        dag_critical_cost = select_stacked("dag_critical_cost")
        dag_dependency_cost = select_stacked("dag_dependency_cost")
        dag_edge_count = select_stacked("dag_edge_count")
        dag_slack = select_stacked("dag_slack")
        dag_criticality = select_stacked("dag_criticality")
        preference_energy_components = {}
        for name in list(PreferenceEnergyHead.component_names) + ["total_energy"]:
            preference_energy_components[name] = torch.stack(
                [item["preference_energy_components"][name] for item in scored],
                dim=0,
            )[best_idx, batch_idx]
        self.last_compact_schedule = compact_schedule.detach()
        self.last_coupling_scores = coupling_scores.detach()

        action_pred = self.normalizer["action"].unnormalize(best_normalized_action)
        start = self.n_obs_steps - 1
        end = start + self.n_action_steps
        result = {
            "action": action_pred[:, start:end],
            "action_pred": action_pred,
            "compact_schedule": compact_schedule,
            "compact_schedule_durations": compact_schedule_durations,
            "compact_schedule_ends": compact_schedule_ends,
            "compact_schedule_speed_scale": compact_schedule_speed_scale,
            "makespan": makespan,
            "coupling_scores": coupling_scores,
            "candidate_scores": score_stack.detach(),
            "wm_scores": wm_scores,
            "energy_scores": energy_scores,
            "dependency_cost": dependency_cost,
            "dynamics_cost": dynamics_cost,
            "collision_or_risk_cost": collision_or_risk_cost,
            "dag_precedence_cost": dag_precedence_cost,
            "dag_sync_cost": dag_sync_cost,
            "dag_critical_cost": dag_critical_cost,
            "dag_dependency_cost": dag_dependency_cost,
            "dag_edge_count": dag_edge_count,
            "dag_slack": dag_slack,
            "dag_criticality": dag_criticality,
            "preference_energy_components": preference_energy_components,
        }
        if rl_mode or return_candidate_batch:
            candidate_action_stack = action_stack.permute(1, 0, 2, 3).contiguous()
            flat_candidates = candidate_action_stack.reshape(batch_size * num_candidates, self.horizon, self.action_dim)
            candidate_actions = self.normalizer["action"].unnormalize(flat_candidates).reshape(
                batch_size,
                num_candidates,
                self.horizon,
                self.action_dim,
            )
            result.update({
                "candidate_actions": candidate_actions,
                "candidate_normalized_actions": candidate_action_stack,
                "candidate_schedules": torch.stack([item["schedule"] for item in scored], dim=0).permute(1, 0, 2, 3),
                "candidate_schedule_durations": torch.stack([item["durations"] for item in scored], dim=0).permute(1, 0, 2, 3),
                "candidate_logits": candidate_logits.detach(),
                "selected_candidate_idx": best_idx.detach(),
                "selected_logprob": selected_logprob.detach(),
                "reward_features": {
                    "makespan": makespan.detach(),
                    "dag_dependency_cost": dag_dependency_cost.detach(),
                    "dynamics_cost": dynamics_cost.detach(),
                    "wm_scores": wm_scores.detach(),
                    "speed_scale_max": compact_schedule_speed_scale.reshape(batch_size, -1).max(dim=-1).values.detach(),
                },
            })
        return result

    def _flow_matching_loss(
        self,
        nactions: torch.Tensor,
        global_cond: torch.Tensor,
        sample_weights: torch.Tensor = None,
    ) -> torch.Tensor:
        batch_size = nactions.shape[0]
        noise = torch.randn_like(nactions)
        timesteps = torch.rand((batch_size,), device=nactions.device, dtype=nactions.dtype)
        t_view = timesteps.view(batch_size, 1, 1)
        noisy_actions = (1.0 - t_view) * noise + t_view * nactions
        target_velocity = nactions - noise
        pred_velocity = self.model(noisy_actions, timesteps, global_cond=global_cond)
        per_sample = reduce(
            F.mse_loss(pred_velocity, target_velocity, reduction="none"),
            "b ... -> b (...)",
            "mean",
        )
        if sample_weights is None:
            return per_sample.mean()
        sample_weights = sample_weights.to(device=nactions.device, dtype=nactions.dtype).view(-1)
        sample_weights = sample_weights / sample_weights.mean().clamp_min(1.0e-6)
        return (per_sample * sample_weights.detach()).mean()

    def _reward_weights(self, returns: torch.Tensor) -> torch.Tensor:
        returns = returns.view(-1).to(dtype=self.dtype, device=self.device)
        centered = returns - returns.mean()
        scaled = centered / returns.std(unbiased=False).clamp_min(1.0e-6)
        return F.softplus(scaled / self.reward_weight_temperature) + 1.0e-3

    def _candidate_ppo_loss(
        self,
        nobs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
        returns: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        required = ("candidate_actions", "selected_candidate_idx", "selected_logprob")
        if not all(key in batch for key in required):
            zero = returns.new_zeros(())
            return zero, {"rl_ppo_reranker_loss": zero}

        candidate_actions = batch["candidate_actions"]
        if candidate_actions.dim() != 4:
            zero = returns.new_zeros(())
            return zero, {"rl_ppo_reranker_loss": zero}

        batch_size, num_candidates, horizon, action_dim = candidate_actions.shape
        obs_features = self._encode_obs_sequence(nobs)
        global_cond = self._encode_global_cond(obs_features)
        flat_actions = self.normalizer["action"].normalize(
            candidate_actions.reshape(batch_size * num_candidates, horizon, action_dim)
        )
        flat_global = global_cond.repeat_interleave(num_candidates, dim=0)
        flat_obs_features = obs_features.repeat_interleave(num_candidates, dim=0)
        flat_scores = self._score_action(flat_actions, flat_global, flat_obs_features)["total_cost"]
        logits = (-flat_scores.reshape(batch_size, num_candidates)) / self.rl_candidate_temperature
        log_probs = F.log_softmax(logits, dim=-1)
        selected_idx = batch["selected_candidate_idx"].long().view(-1, 1)
        new_logprob = log_probs.gather(1, selected_idx).squeeze(1)
        old_logprob = batch["selected_logprob"].to(device=new_logprob.device, dtype=new_logprob.dtype).view(-1)
        advantages = returns.to(device=new_logprob.device, dtype=new_logprob.dtype).view(-1)
        advantages = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(1.0e-6)
        ratio = torch.exp(new_logprob - old_logprob)
        clipped = torch.clamp(ratio, 1.0 - self.rl_ppo_clip, 1.0 + self.rl_ppo_clip)
        loss = -torch.minimum(ratio * advantages.detach(), clipped * advantages.detach()).mean()
        return loss, {
            "rl_ppo_reranker_loss": loss,
            "rl_candidate_entropy": -(log_probs.exp() * log_probs).sum(dim=-1).mean(),
            "rl_logprob_ratio": ratio.mean(),
        }

    def compute_rl_loss(self, batch: Dict[str, torch.Tensor], bc_batch: Dict[str, torch.Tensor] = None):
        nobs = self.normalizer.normalize(batch["obs"])
        nactions = self.normalizer["action"].normalize(batch["action"])
        returns = batch.get("return", batch.get("reward"))
        if returns is None:
            returns = torch.ones(nactions.shape[0], device=nactions.device, dtype=nactions.dtype)
        returns = returns.to(device=nactions.device, dtype=nactions.dtype).view(-1)

        obs_features = self._encode_obs_sequence(nobs)
        global_cond = self._encode_global_cond(obs_features)
        reward_weights = self._reward_weights(returns)
        self_imitation_loss = self._flow_matching_loss(
            nactions,
            global_cond,
            sample_weights=reward_weights,
        )
        total_loss = self.rl_self_imitation_weight * self_imitation_loss
        logs = {
            "rl_self_imitation_loss": self_imitation_loss,
            "rl_return_mean": returns.mean(),
            "rl_return_max": returns.max(),
            "rl_reward_weight_mean": reward_weights.mean(),
        }

        preference_loss, preference_logs = self._preference_energy_loss(nactions, global_cond, obs_features)
        total_loss = total_loss + self.energy_loss_weight * preference_loss
        logs.update({f"rl_{key}": value for key, value in preference_logs.items()})

        ppo_loss, ppo_logs = self._candidate_ppo_loss(nobs, batch, returns)
        total_loss = total_loss + self.rl_ppo_reranker_weight * ppo_loss
        logs.update(ppo_logs)

        if bc_batch is not None and self.rl_bc_anchor_weight > 0:
            bc_nobs = self.normalizer.normalize(bc_batch["obs"])
            bc_actions = self.normalizer["action"].normalize(bc_batch["action"])
            bc_obs_features = self._encode_obs_sequence(bc_nobs)
            bc_global_cond = self._encode_global_cond(bc_obs_features)
            bc_anchor_loss = self._flow_matching_loss(bc_actions, bc_global_cond)
            total_loss = total_loss + self.rl_bc_anchor_weight * bc_anchor_loss
            logs["rl_bc_anchor_loss"] = bc_anchor_loss

        self.last_loss_dict = {
            key: float(value.detach().cpu())
            for key, value in logs.items()
        }
        return total_loss

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
                nactions[:, :, self.left_slice],
                nactions[:, :, self.right_slice],
                global_cond,
            )
            total_loss = total_loss + self.coupling_aux_loss_weight * coupling_loss
            extra_logs.update(coupling_logs)

        preference_loss, preference_logs = self._preference_energy_loss(nactions, global_cond, obs_features)
        total_loss = total_loss + self.energy_loss_weight * preference_loss
        extra_logs.update(preference_logs)

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
