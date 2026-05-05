from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from einops import reduce

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.model.vision.multi_image_obs_encoder import MultiImageObsEncoder
from diffusion_policy.policy.base_image_policy import BaseImagePolicy


def _slice_from_start_dim(start: int, dim: int) -> slice:
    return slice(start, start + dim)


def _compressed_len(horizon: int, stride: int) -> int:
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")
    return (horizon + stride - 1) // stride


def _gather_stride(actions: torch.Tensor, arm_slice: slice, stride: int) -> torch.Tensor:
    indices = torch.arange(0, actions.shape[1], stride, device=actions.device)
    return actions.index_select(1, indices)[:, :, arm_slice]


def _expand_stride(actions: torch.Tensor, horizon: int, stride: int) -> torch.Tensor:
    step_ids = torch.arange(horizon, device=actions.device)
    compressed_ids = torch.clamp(step_ids // stride, max=actions.shape[1] - 1)
    return actions.index_select(1, compressed_ids)


def _update_mask(horizon: int, stride: int, batch_size: int, device, dtype) -> torch.Tensor:
    step_ids = torch.arange(horizon, device=device)
    mask = (step_ids % stride == 0).to(dtype).view(1, horizon, 1)
    return mask.expand(batch_size, -1, -1)


class StrictAsymGate(nn.Module):
    def __init__(self, action_dim: int, cond_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(action_dim * 2 + cond_dim + 1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.expert_head = nn.Linear(hidden_dim, 4)
        self.support_head = nn.Linear(hidden_dim, 2)
        nn.init.zeros_(self.expert_head.weight)
        nn.init.constant_(self.expert_head.bias, 0.0)
        nn.init.zeros_(self.support_head.weight)
        nn.init.constant_(self.support_head.bias, 0.0)

    def forward(
        self,
        full_noisy: torch.Tensor,
        timesteps: torch.Tensor,
        global_cond: Optional[torch.Tensor],
        num_train_timesteps: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=full_noisy.device)
        elif timesteps.ndim == 0:
            timesteps = timesteps[None].to(full_noisy.device)
        timesteps = timesteps.expand(full_noisy.shape[0]).float()
        time_feat = (timesteps / max(num_train_timesteps - 1, 1)).view(-1, 1)

        pooled = torch.cat(
            [
                full_noisy.mean(dim=1),
                full_noisy.std(dim=1, unbiased=False),
            ],
            dim=-1,
        )
        if global_cond is None:
            global_cond = torch.zeros(full_noisy.shape[0], 0, device=full_noisy.device, dtype=full_noisy.dtype)
        hidden = self.net(torch.cat([pooled, global_cond, time_feat], dim=-1))
        expert_prob = torch.softmax(self.expert_head(hidden), dim=-1)
        support_prob = torch.softmax(self.support_head(hidden), dim=-1)
        return expert_prob, support_prob


class StrictAsymDiffusionUnetImagePolicy(BaseImagePolicy):
    """
    Role-adaptive asymmetric bimanual diffusion policy.

    The policy keeps the bimanual expert structure but lets the gate decide
    which arm should act as the smoother support arm for the current state.
    Frequency strides remain available for ablations, but the default path uses
    full-rate left/right actions and dynamic role assignment.
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
        left_action_dim=None,
        right_action_dim=None,
        left_action_start=0,
        right_action_start=None,
        left_stride=1,
        right_stride=4,
        asym_hidden_dim=256,
        asym_cost_weights=(1.0, 0.5, 0.5, 0.0),
        asym_lambda_cost=1e-3,
        asym_lambda_entropy=1e-4,
        asym_lambda_update_sparse=1e-3,
        asym_lambda_smooth=1e-3,
        asym_lambda_role_smooth=1e-3,
        **kwargs,
    ):
        super().__init__()
        if not obs_as_global_cond:
            raise ValueError("StrictAsymDiffusionUnetImagePolicy currently requires obs_as_global_cond=True")

        action_shape = shape_meta["action"]["shape"]
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        obs_feature_dim = obs_encoder.output_shape()[0]

        if left_action_dim is None:
            left_action_dim = action_dim // 2
        if right_action_dim is None:
            right_action_dim = action_dim - left_action_dim
        if right_action_start is None:
            right_action_start = left_action_start + left_action_dim

        self.left_action_dim = int(left_action_dim)
        self.right_action_dim = int(right_action_dim)
        self.left_slice = _slice_from_start_dim(int(left_action_start), self.left_action_dim)
        self.right_slice = _slice_from_start_dim(int(right_action_start), self.right_action_dim)
        self.left_stride = int(left_stride)
        self.right_stride = int(right_stride)
        self.left_horizon = _compressed_len(horizon, self.left_stride)
        self.right_horizon = _compressed_len(horizon, self.right_stride)

        global_cond_dim = obs_feature_dim * n_obs_steps

        def build_arm_unet(input_dim: int, cond_dim: int):
            return ConditionalUnet1D(
                input_dim=input_dim,
                local_cond_dim=None,
                global_cond_dim=cond_dim,
                diffusion_step_embed_dim=diffusion_step_embed_dim,
                down_dims=down_dims,
                kernel_size=kernel_size,
                n_groups=n_groups,
                cond_predict_scale=cond_predict_scale,
            )

        self.left_local_model = build_arm_unet(self.left_action_dim, global_cond_dim)
        self.right_local_model = build_arm_unet(self.right_action_dim, global_cond_dim)
        self.left_cond_model = build_arm_unet(self.left_action_dim, global_cond_dim + self.right_action_dim * 2)
        self.right_cond_model = build_arm_unet(self.right_action_dim, global_cond_dim + self.left_action_dim * 2)
        self.gating = StrictAsymGate(action_dim=action_dim, cond_dim=global_cond_dim, hidden_dim=asym_hidden_dim)

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

        self.asym_lambda_cost = float(asym_lambda_cost)
        self.asym_lambda_entropy = float(asym_lambda_entropy)
        self.asym_lambda_update_sparse = float(asym_lambda_update_sparse)
        self.asym_lambda_smooth = float(asym_lambda_smooth)
        self.asym_lambda_role_smooth = float(asym_lambda_role_smooth)
        self.register_buffer("asym_cost_weights", torch.tensor(asym_cost_weights, dtype=torch.float32))

        self.last_gate_info = None
        self.last_loss_dict = {}
        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

    def _encode_obs(self, obs_dict):
        nobs = self.normalizer.normalize(obs_dict)
        value = next(iter(nobs.values()))
        batch_size = value.shape[0]
        this_nobs = dict_apply(nobs, lambda x: x[:, :self.n_obs_steps, ...].reshape(-1, *x.shape[2:]))
        nobs_features = self.obs_encoder(this_nobs)
        return nobs_features.reshape(batch_size, -1)

    def _combine_full(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        full = torch.zeros(left.shape[0], self.horizon, self.action_dim, device=left.device, dtype=left.dtype)
        full[:, :, self.left_slice] = _expand_stride(left, self.horizon, self.left_stride)
        full[:, :, self.right_slice] = _expand_stride(right, self.horizon, self.right_stride)
        return full

    @staticmethod
    def _context(traj: torch.Tensor) -> torch.Tensor:
        return torch.cat([traj.mean(dim=1), traj.std(dim=1, unbiased=False)], dim=-1)

    def _strict_update_prob(self, batch_size: int, device, dtype) -> torch.Tensor:
        left = _update_mask(self.horizon, self.left_stride, batch_size, device, dtype)
        right = _update_mask(self.horizon, self.right_stride, batch_size, device, dtype)
        return torch.cat([left, right], dim=-1)

    def _predict_strict(
        self,
        left_noisy: torch.Tensor,
        right_noisy: torch.Tensor,
        timesteps: torch.Tensor,
        global_cond: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        full_noisy = self._combine_full(left_noisy, right_noisy)
        expert_prob, support_prob = self.gating(
            full_noisy=full_noisy,
            timesteps=timesteps,
            global_cond=global_cond,
            num_train_timesteps=self.noise_scheduler.config.num_train_timesteps,
        )

        left_local = self.left_local_model(left_noisy, timesteps, global_cond=global_cond)
        right_local = self.right_local_model(right_noisy, timesteps, global_cond=global_cond)

        left_cond = self.left_cond_model(
            left_noisy,
            timesteps,
            global_cond=torch.cat([global_cond, self._context(right_noisy)], dim=-1),
        )
        right_cond = self.right_cond_model(
            right_noisy,
            timesteps,
            global_cond=torch.cat([global_cond, self._context(left_noisy)], dim=-1),
        )

        # A uses both conditional arms; LR uses local left + conditioned right;
        # RL uses conditioned left + local right; C uses both local arms.
        left_experts = torch.stack([left_cond, left_local, left_cond, left_local], dim=1)
        right_experts = torch.stack([right_cond, right_cond, right_local, right_local], dim=1)
        left_pred = (expert_prob[:, :, None, None] * left_experts).sum(dim=1)
        right_pred = (expert_prob[:, :, None, None] * right_experts).sum(dim=1)

        update_prob = self._strict_update_prob(
            batch_size=left_noisy.shape[0],
            device=left_noisy.device,
            dtype=left_noisy.dtype,
        )
        gate_info = {
            "expert_prob": expert_prob,
            "support_prob": support_prob,
            "update_prob": update_prob,
        }
        self.last_gate_info = gate_info
        return left_pred, right_pred, gate_info

    def _regularization_loss(self, pred_full: torch.Tensor, gate_info: Dict[str, torch.Tensor]):
        expert_prob = gate_info["expert_prob"]
        support_prob = gate_info["support_prob"]
        update_prob = gate_info["update_prob"]
        losses = {}
        losses["asym_cost"] = (expert_prob * self.asym_cost_weights.to(expert_prob.device)).sum(dim=-1).mean()
        losses["asym_entropy"] = -(expert_prob.clamp(min=1e-6).log() * expert_prob).sum(dim=-1).mean()
        losses["asym_update_sparse"] = update_prob.mean()

        if pred_full.shape[1] > 1:
            smooth = pred_full[:, 1:] - pred_full[:, :-1]
            losses["asym_smooth"] = smooth.pow(2).mean()
            left_vel = pred_full[:, 1:, self.left_slice] - pred_full[:, :-1, self.left_slice]
            right_vel = pred_full[:, 1:, self.right_slice] - pred_full[:, :-1, self.right_slice]
            left_smooth = reduce(left_vel.pow(2), "b ... -> b", "mean")
            right_smooth = reduce(right_vel.pow(2), "b ... -> b", "mean")
            losses["asym_role_smooth"] = (
                support_prob[:, 0] * left_smooth + support_prob[:, 1] * right_smooth
            ).mean()
        else:
            losses["asym_smooth"] = pred_full.sum() * 0.0
            losses["asym_role_smooth"] = pred_full.sum() * 0.0

        total = (
            self.asym_lambda_cost * losses["asym_cost"]
            + self.asym_lambda_entropy * losses["asym_entropy"]
            + self.asym_lambda_update_sparse * losses["asym_update_sparse"]
            + self.asym_lambda_smooth * losses["asym_smooth"]
            + self.asym_lambda_role_smooth * losses["asym_role_smooth"]
        )
        return total, losses

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        assert "valid_mask" not in batch
        nobs = self.normalizer.normalize(batch["obs"])
        nactions = self.normalizer["action"].normalize(batch["action"])
        batch_size = nactions.shape[0]

        this_nobs = dict_apply(nobs, lambda x: x[:, :self.n_obs_steps, ...].reshape(-1, *x.shape[2:]))
        nobs_features = self.obs_encoder(this_nobs)
        global_cond = nobs_features.reshape(batch_size, -1)

        left_target = _gather_stride(nactions, self.left_slice, self.left_stride)
        right_target = _gather_stride(nactions, self.right_slice, self.right_stride)
        left_noise = torch.randn_like(left_target)
        right_noise = torch.randn_like(right_target)
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (batch_size,),
            device=nactions.device,
        ).long()

        left_noisy = self.noise_scheduler.add_noise(left_target, left_noise, timesteps)
        right_noisy = self.noise_scheduler.add_noise(right_target, right_noise, timesteps)
        left_pred, right_pred, gate_info = self._predict_strict(left_noisy, right_noisy, timesteps, global_cond)

        pred_type = self.noise_scheduler.config.prediction_type
        if pred_type == "epsilon":
            left_label = left_noise
            right_label = right_noise
        elif pred_type == "sample":
            left_label = left_target
            right_label = right_target
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        left_loss = reduce(F.mse_loss(left_pred, left_label, reduction="none"), "b ... -> b (...)", "mean").mean()
        right_loss = reduce(F.mse_loss(right_pred, right_label, reduction="none"), "b ... -> b (...)", "mean").mean()
        diffusion_loss = 0.5 * (left_loss + right_loss)
        pred_full = self._combine_full(left_pred, right_pred)
        reg_loss, reg_losses = self._regularization_loss(pred_full, gate_info)
        total_loss = diffusion_loss + reg_loss

        self.last_loss_dict = {
            "diffusion_loss": float(diffusion_loss.detach().cpu()),
            "left_diffusion_loss": float(left_loss.detach().cpu()),
            "right_diffusion_loss": float(right_loss.detach().cpu()),
            "asym_cost": float(reg_losses["asym_cost"].detach().cpu()),
            "asym_entropy": float(reg_losses["asym_entropy"].detach().cpu()),
            "asym_update_sparse": float(reg_losses["asym_update_sparse"].detach().cpu()),
            "asym_smooth": float(reg_losses["asym_smooth"].detach().cpu()),
            "asym_role_smooth": float(reg_losses["asym_role_smooth"].detach().cpu()),
            "left_support_prob": float(gate_info["support_prob"][:, 0].mean().detach().cpu()),
            "right_support_prob": float(gate_info["support_prob"][:, 1].mean().detach().cpu()),
            "left_stride": float(self.left_stride),
            "right_stride": float(self.right_stride),
        }
        return total_loss

    def _strict_conditional_sample(self, batch_size: int, global_cond: torch.Tensor, generator=None):
        device = self.device
        dtype = self.dtype
        left = torch.randn(
            size=(batch_size, self.left_horizon, self.left_action_dim),
            dtype=dtype,
            device=device,
            generator=generator,
        )
        right = torch.randn(
            size=(batch_size, self.right_horizon, self.right_action_dim),
            dtype=dtype,
            device=device,
            generator=generator,
        )

        scheduler = self.noise_scheduler
        scheduler.set_timesteps(self.num_inference_steps)
        for t in scheduler.timesteps:
            left_pred, right_pred, _ = self._predict_strict(left, right, t, global_cond)
            left = scheduler.step(left_pred, t, left, generator=generator, **self.kwargs).prev_sample
            right = scheduler.step(right_pred, t, right, generator=generator, **self.kwargs).prev_sample
        return left, right

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        assert "past_action" not in obs_dict
        value = next(iter(obs_dict.values()))
        batch_size = value.shape[0]
        global_cond = self._encode_obs(obs_dict)

        left_sample, right_sample = self._strict_conditional_sample(batch_size, global_cond)
        normalized_full = self._combine_full(left_sample, right_sample)
        action_pred = self.normalizer["action"].unnormalize(normalized_full)

        start = self.n_obs_steps - 1
        end = start + self.n_action_steps
        result = {
            "action": action_pred[:, start:end],
            "action_pred": action_pred,
        }
        if self.last_gate_info is not None:
            result["expert_prob"] = self.last_gate_info["expert_prob"]
            result["support_prob"] = self.last_gate_info["support_prob"]
            result["update_prob"] = self.last_gate_info["update_prob"]
        return result
