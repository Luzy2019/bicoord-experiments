from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy.model.vision.multi_image_obs_encoder import MultiImageObsEncoder
from diffusion_policy.policy.base_image_policy import BaseImagePolicy


class LatentTraceWorldModel(nn.Module):
    """Latent verifier for action-time traces.

    This head is deliberately small: it learns whether a local action-time trace
    predicts the next observation embedding. It supplies a world-model signal for
    search/reranking without requiring compact demonstrations.
    """

    def __init__(
        self,
        obs_feature_dim: int,
        trace_dim: int,
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
            nn.LayerNorm(latent_dim + trace_dim),
            nn.Linear(latent_dim + trace_dim, hidden_dim),
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
        trace: torch.Tensor,
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
        transition_trace = trace[:, : z_now.shape[1]]
        pred_next = z_now + self.dynamics(torch.cat([z_now, transition_trace], dim=-1))
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


class TraceEnergyHead(nn.Module):
    """Energy over an execution trace Y=(A_L,A_R,t_L,t_R)."""

    def __init__(
        self,
        trace_dim: int,
        cond_dim: int,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(trace_dim * 2 + cond_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, trace: torch.Tensor, global_cond: torch.Tensor) -> torch.Tensor:
        pooled = torch.cat(
            [
                trace.mean(dim=1),
                trace.std(dim=1, unbiased=False),
            ],
            dim=-1,
        )
        return self.net(torch.cat([pooled, global_cond], dim=-1)).squeeze(-1)


class ActionTimeDiffusionUnetImagePolicy(BaseImagePolicy):
    """Diffusion Policy over action-time execution traces.

    Instead of post-processing a generated action chunk with alpha/gates, this
    policy treats timing as part of the generated object:

        Y = [A_L, A_R, t_L, t_R]

    The action channels are executed as usual. The time channels can be used by
    a scheduler, WM/energy reranker, or later distillation pipeline to learn
    compact parallel execution traces.
    """

    def __init__(
        self,
        shape_meta: dict,
        noise_scheduler: DDPMScheduler,
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
        time_dim=2,
        compactness_weight=0.1,
        monotonic_loss_weight=0.01,
        energy_loss_weight=0.01,
        energy_hidden_dim=256,
        energy_margin=1.0,
        world_model_enabled=True,
        world_model_loss_weight=0.05,
        world_model_gaussian_weight=0.01,
        world_model_latent_dim=128,
        world_model_hidden_dim=256,
        action_time_rerank_samples=1,
        **kwargs,
    ):
        super().__init__()
        if not obs_as_global_cond:
            raise ValueError("ActionTimeDiffusionUnetImagePolicy requires obs_as_global_cond=True")

        action_shape = shape_meta["action"]["shape"]
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        obs_feature_dim = obs_encoder.output_shape()[0]
        trace_dim = action_dim + int(time_dim)
        global_cond_dim = obs_feature_dim * n_obs_steps

        self.model = ConditionalUnet1D(
            input_dim=trace_dim,
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
        self.mask_generator = LowdimMaskGenerator(
            action_dim=trace_dim,
            obs_dim=0,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False,
        )
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.time_dim = int(time_dim)
        self.trace_dim = trace_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.kwargs = kwargs
        self.compactness_weight = float(compactness_weight)
        self.monotonic_loss_weight = float(monotonic_loss_weight)
        self.energy_loss_weight = float(energy_loss_weight)
        self.energy_margin = float(energy_margin)
        self.world_model_enabled = bool(world_model_enabled)
        self.world_model_loss_weight = float(world_model_loss_weight)
        self.world_model_gaussian_weight = float(world_model_gaussian_weight)
        self.action_time_rerank_samples = int(action_time_rerank_samples)
        self.last_loss_dict = {}
        self.last_time_trace = None
        self.last_candidate_scores = None

        self.trace_energy = TraceEnergyHead(
            trace_dim=trace_dim,
            cond_dim=global_cond_dim,
            hidden_dim=int(energy_hidden_dim),
        )
        if self.world_model_enabled:
            self.trace_world_model = LatentTraceWorldModel(
                obs_feature_dim=obs_feature_dim,
                trace_dim=trace_dim,
                latent_dim=int(world_model_latent_dim),
                hidden_dim=int(world_model_hidden_dim),
            )
        else:
            self.trace_world_model = None

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

    def _encode_obs_sequence(self, nobs: Dict[str, torch.Tensor]) -> torch.Tensor:
        value = next(iter(nobs.values()))
        batch_size = value.shape[0]
        this_nobs = dict_apply(nobs, lambda x: x[:, : self.n_obs_steps, ...].reshape(-1, *x.shape[2:]))
        obs_features = self.obs_encoder(this_nobs)
        return obs_features.reshape(batch_size, self.n_obs_steps, -1)

    def _encode_global_cond(self, nobs: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self._encode_obs_sequence(nobs).reshape(next(iter(nobs.values())).shape[0], -1)

    def _default_time_trace(self, batch_size: int, horizon: int, device, dtype) -> torch.Tensor:
        base = torch.linspace(0.0, 1.0, horizon, device=device, dtype=dtype)
        return base.view(1, horizon, 1).expand(batch_size, horizon, self.time_dim)

    def _target_time_trace(self, batch: dict, action_reference: torch.Tensor) -> torch.Tensor:
        if "time_trace" in batch:
            time_trace = batch["time_trace"].to(device=action_reference.device, dtype=action_reference.dtype)
            return time_trace[..., : self.time_dim]
        return self._default_time_trace(
            action_reference.shape[0],
            action_reference.shape[1],
            action_reference.device,
            action_reference.dtype,
        )

    def _make_trace(self, normalized_actions: torch.Tensor, time_trace: torch.Tensor) -> torch.Tensor:
        return torch.cat([normalized_actions, time_trace], dim=-1)

    def _decode_time_trace(self, trace: torch.Tensor) -> torch.Tensor:
        raw_time = trace[..., self.action_dim : self.action_dim + self.time_dim]
        time_trace = torch.sigmoid(raw_time)
        time_trace, _ = torch.sort(time_trace, dim=1)
        return time_trace

    def _decode_action_trace(self, trace: torch.Tensor) -> torch.Tensor:
        return trace[..., : self.action_dim]

    def _canonical_trace_for_energy(self, trace: torch.Tensor) -> torch.Tensor:
        return torch.cat([self._decode_action_trace(trace), self._decode_time_trace(trace)], dim=-1)

    def _makespan(self, trace: torch.Tensor) -> torch.Tensor:
        time_trace = self._decode_time_trace(trace)
        return time_trace[:, -1].amax(dim=-1)

    def _monotonic_penalty(self, time_trace: torch.Tensor) -> torch.Tensor:
        if time_trace.shape[1] <= 1:
            return time_trace.new_zeros(())
        return F.relu(time_trace[:, :-1] - time_trace[:, 1:]).pow(2).mean()

    def _energy_contrastive_loss(
        self,
        positive_trace: torch.Tensor,
        global_cond: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        positive_energy = self.trace_energy(positive_trace, global_cond)
        negative = positive_trace.clone()
        if positive_trace.shape[0] > 1:
            perm = torch.randperm(positive_trace.shape[0], device=positive_trace.device)
            negative[..., : self.action_dim] = positive_trace[perm, :, : self.action_dim]
        else:
            negative[..., : self.action_dim] = negative[..., : self.action_dim] + 0.25 * torch.randn_like(
                negative[..., : self.action_dim]
            )
        random_time = torch.rand_like(negative[..., self.action_dim :])
        negative[..., self.action_dim :] = random_time
        negative_energy = self.trace_energy(negative, global_cond)
        loss = F.softplus(positive_energy - negative_energy + self.energy_margin).mean()
        return loss, {
            "trace_energy_positive": positive_energy.mean(),
            "trace_energy_negative": negative_energy.mean(),
        }

    def conditional_sample(
        self,
        condition_data,
        condition_mask,
        local_cond=None,
        global_cond=None,
        generator=None,
        **kwargs,
    ):
        trajectory = torch.randn(
            size=condition_data.shape,
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator,
        )

        self.noise_scheduler.set_timesteps(self.num_inference_steps)
        for t in self.noise_scheduler.timesteps:
            trajectory[condition_mask] = condition_data[condition_mask]
            model_output = self.model(trajectory, t, local_cond=local_cond, global_cond=global_cond)
            trajectory = self.noise_scheduler.step(model_output, t, trajectory, generator=generator, **kwargs).prev_sample
        trajectory[condition_mask] = condition_data[condition_mask]
        return trajectory

    def _sample_trace(self, batch_size: int, global_cond: torch.Tensor, generator=None) -> torch.Tensor:
        cond_data = torch.zeros(
            size=(batch_size, self.horizon, self.trace_dim),
            device=self.device,
            dtype=self.dtype,
        )
        cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        return self.conditional_sample(cond_data, cond_mask, global_cond=global_cond, generator=generator, **self.kwargs)

    def _score_trace(self, trace: torch.Tensor, global_cond: torch.Tensor) -> torch.Tensor:
        canonical = self._canonical_trace_for_energy(trace)
        energy = self.trace_energy(canonical, global_cond)
        compactness = self._makespan(trace)
        monotonic = self._monotonic_penalty(self._decode_time_trace(trace))
        return energy + self.compactness_weight * compactness + self.monotonic_loss_weight * monotonic

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        assert "past_action" not in obs_dict
        nobs = self.normalizer.normalize(obs_dict)
        value = next(iter(nobs.values()))
        batch_size = value.shape[0]
        global_cond = self._encode_global_cond(nobs)

        traces = []
        scores = []
        num_candidates = max(self.action_time_rerank_samples, 1)
        for _ in range(num_candidates):
            trace = self._sample_trace(batch_size, global_cond)
            traces.append(trace)
            scores.append(self._score_trace(trace, global_cond))

        trace_stack = torch.stack(traces, dim=0)
        score_stack = torch.stack(scores, dim=0)
        best_idx = score_stack.argmin(dim=0)
        batch_idx = torch.arange(batch_size, device=trace_stack.device)
        best_trace = trace_stack[best_idx, batch_idx]
        self.last_candidate_scores = score_stack.detach()

        normalized_action = self._decode_action_trace(best_trace)
        action_pred = self.normalizer["action"].unnormalize(normalized_action)
        time_trace = self._decode_time_trace(best_trace)
        self.last_time_trace = time_trace.detach()

        start = self.n_obs_steps - 1
        end = start + self.n_action_steps
        return {
            "action": action_pred[:, start:end],
            "action_pred": action_pred,
            "time_trace": time_trace,
            "candidate_scores": score_stack.detach(),
        }

    def compute_loss(self, batch):
        assert "valid_mask" not in batch
        nobs = self.normalizer.normalize(batch["obs"])
        nactions = self.normalizer["action"].normalize(batch["action"])
        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]

        obs_features = self._encode_obs_sequence(nobs)
        global_cond = obs_features.reshape(batch_size, -1)
        target_time = self._target_time_trace(batch, nactions)
        target_trace = self._make_trace(nactions, target_time)

        condition_mask = self.mask_generator(target_trace.shape)
        noise = torch.randn(target_trace.shape, device=target_trace.device)
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (batch_size,),
            device=target_trace.device,
        ).long()
        noisy_trace = self.noise_scheduler.add_noise(target_trace, noise, timesteps)
        noisy_trace[condition_mask] = target_trace[condition_mask]
        pred = self.model(noisy_trace, timesteps, global_cond=global_cond)

        pred_type = self.noise_scheduler.config.prediction_type
        if pred_type == "epsilon":
            target = noise
        elif pred_type == "sample":
            target = target_trace
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss_mask = ~condition_mask
        diffusion_loss = F.mse_loss(pred, target, reduction="none")
        diffusion_loss = diffusion_loss * loss_mask.type(diffusion_loss.dtype)
        diffusion_loss = reduce(diffusion_loss, "b ... -> b (...)", "mean").mean()

        mono_loss = self._monotonic_penalty(target_time)
        energy_loss, energy_logs = self._energy_contrastive_loss(target_trace, global_cond)
        total_loss = diffusion_loss + self.monotonic_loss_weight * mono_loss + self.energy_loss_weight * energy_loss

        extra_logs = {
            "diffusion_loss": diffusion_loss,
            "time_monotonic_loss": mono_loss,
            "energy_contrastive_loss": energy_loss,
            **energy_logs,
        }
        if self.trace_world_model is not None and self.world_model_loss_weight > 0:
            world_loss, world_logs = self.trace_world_model.compute_loss(
                obs_features,
                target_trace,
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
