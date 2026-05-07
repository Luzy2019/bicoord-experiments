from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import reduce

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.model.vision.multi_image_obs_encoder import MultiImageObsEncoder
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.policy.speed_modulation import (
    SpeedModulationHead,
    compute_action_risk,
    compute_branch_mse_risk,
    normalize_risk,
    speed_modulation_loss,
    warp_action_sequence,
)


def _slice_from_start_dim(start: int, dim: int) -> slice:
    return slice(start, start + dim)


class FactorizedBimanualFlowGate(nn.Module):
    """Predicts w(t), u(t) for conditional-vs-marginal velocity correction."""

    def __init__(self, action_dim: int, cond_dim: int, hidden_dim: int = 256, init_bias: float = -2.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(action_dim * 2 + cond_dim + 1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, init_bias)

    def forward(self, full_state: torch.Tensor, timesteps: torch.Tensor, global_cond: Optional[torch.Tensor], num_train_timesteps: int):
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.float32, device=full_state.device)
        elif timesteps.ndim == 0:
            timesteps = timesteps[None].to(full_state.device)
        timesteps = timesteps.expand(full_state.shape[0]).float()
        time_feat = (timesteps / max(num_train_timesteps - 1, 1)).view(-1, 1)
        pooled = torch.cat([full_state.mean(dim=1), full_state.std(dim=1, unbiased=False)], dim=-1)
        if global_cond is None:
            global_cond = torch.zeros(full_state.shape[0], 0, device=full_state.device, dtype=full_state.dtype)
        return torch.sigmoid(self.net(torch.cat([pooled, global_cond, time_feat], dim=-1)))


class FactorizedBimanualFlowUnetImagePolicy(BaseImagePolicy):
    """Flow-matching factorized bimanual policy with learned asymmetric speed."""

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
        factorized_hidden_dim=256,
        factorized_gate_init_bias=-2.0,
        factorized_aux_loss_weight=0.25,
        speed_modulation_enabled=False,
        speed_modulation_min=0.5,
        speed_modulation_max=2.0,
        speed_modulation_learned=True,
        speed_modulation_hidden_dim=128,
        speed_modulation_loss_weight=0.01,
        speed_modulation_target_weight=1.0,
        speed_modulation_smooth_weight=0.1,
        speed_modulation_fast_weight=0.01,
        speed_modulation_risk_weight=0.1,
        speed_modulation_detach_signal=True,
        speed_modulation_train_only=False,
        speed_modulation_coupling_risk_weight=1.0,
        speed_modulation_geometry_risk_weight=0.25,
        **kwargs,
    ):
        super().__init__()
        if not obs_as_global_cond:
            raise ValueError("FactorizedBimanualFlowUnetImagePolicy requires obs_as_global_cond=True")

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
        if self.left_action_dim != self.right_action_dim:
            raise ValueError("Shared factorized flow model requires equal left/right action dimensions")
        self.left_slice = _slice_from_start_dim(int(left_action_start), self.left_action_dim)
        self.right_slice = _slice_from_start_dim(int(right_action_start), self.right_action_dim)

        global_cond_dim = obs_feature_dim * n_obs_steps
        arm_action_dim = self.left_action_dim
        branch_cond_dim = global_cond_dim + arm_action_dim * 2 + 2
        self.factorized_model = ConditionalUnet1D(
            input_dim=arm_action_dim,
            local_cond_dim=None,
            global_cond_dim=branch_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale,
        )
        self.factorized_gate = FactorizedBimanualFlowGate(
            action_dim=action_dim,
            cond_dim=global_cond_dim,
            hidden_dim=factorized_hidden_dim,
            init_bias=factorized_gate_init_bias,
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
        self.factorized_aux_loss_weight = float(factorized_aux_loss_weight)

        num_train_timesteps = getattr(noise_scheduler.config, "num_train_timesteps", 100)
        self.num_train_timesteps = int(num_train_timesteps)
        self.num_inference_steps = int(num_inference_steps or 20)

        self.speed_modulation_enabled = bool(speed_modulation_enabled)
        self.speed_modulation_min = float(speed_modulation_min)
        self.speed_modulation_max = float(speed_modulation_max)
        self.speed_modulation_learned = bool(speed_modulation_learned)
        self.speed_modulation_loss_weight = float(speed_modulation_loss_weight)
        self.speed_modulation_target_weight = float(speed_modulation_target_weight)
        self.speed_modulation_smooth_weight = float(speed_modulation_smooth_weight)
        self.speed_modulation_fast_weight = float(speed_modulation_fast_weight)
        self.speed_modulation_risk_weight = float(speed_modulation_risk_weight)
        self.speed_modulation_detach_signal = bool(speed_modulation_detach_signal)
        self.speed_modulation_train_only = bool(speed_modulation_train_only)
        self.speed_modulation_coupling_risk_weight = float(speed_modulation_coupling_risk_weight)
        self.speed_modulation_geometry_risk_weight = float(speed_modulation_geometry_risk_weight)
        self.left_speed_head = SpeedModulationHead(
            action_dim=self.left_action_dim,
            cond_dim=global_cond_dim,
            hidden_dim=int(speed_modulation_hidden_dim),
            alpha_min=self.speed_modulation_min,
            alpha_max=self.speed_modulation_max,
            init_alpha=1.0,
        )
        self.right_speed_head = SpeedModulationHead(
            action_dim=self.right_action_dim,
            cond_dim=global_cond_dim,
            hidden_dim=int(speed_modulation_hidden_dim),
            alpha_min=self.speed_modulation_min,
            alpha_max=self.speed_modulation_max,
            init_alpha=1.0,
        )

        self.last_gate_info = None
        self.last_loss_dict = {}
        self.last_left_velocity = None
        self.last_right_velocity = None
        self.last_left_speed_alpha = None
        self.last_right_speed_alpha = None

    def _flow_timestep(self, t: torch.Tensor) -> torch.Tensor:
        return t * float(max(self.num_train_timesteps - 1, 1))

    def _encode_obs(self, obs_dict):
        nobs = self.normalizer.normalize(obs_dict)
        value = next(iter(nobs.values()))
        batch_size = value.shape[0]
        this_nobs = dict_apply(nobs, lambda x: x[:, :self.n_obs_steps, ...].reshape(-1, *x.shape[2:]))
        nobs_features = self.obs_encoder(this_nobs)
        return nobs_features.reshape(batch_size, -1)

    def _combine_full(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        full = torch.zeros(left.shape[0], self.horizon, self.action_dim, device=left.device, dtype=left.dtype)
        full[:, :, self.left_slice] = left
        full[:, :, self.right_slice] = right
        return full

    @staticmethod
    def _context(traj: torch.Tensor) -> torch.Tensor:
        return torch.cat([traj.mean(dim=1), traj.std(dim=1, unbiased=False)], dim=-1)

    @staticmethod
    def _flag(batch_size: int, value: float, reference: torch.Tensor) -> torch.Tensor:
        return torch.full((batch_size, 1), fill_value=value, device=reference.device, dtype=reference.dtype)

    def _branch_cond(self, global_cond: torch.Tensor, other_context: torch.Tensor, arm_id: float, cond_mask: float) -> torch.Tensor:
        batch_size = global_cond.shape[0]
        return torch.cat(
            [global_cond, other_context, self._flag(batch_size, arm_id, global_cond), self._flag(batch_size, cond_mask, global_cond)],
            dim=-1,
        )

    def _predict_factorized(self, left_state: torch.Tensor, right_state: torch.Tensor, timesteps: torch.Tensor, global_cond: torch.Tensor):
        full_state = self._combine_full(left_state, right_state)
        gates = self.factorized_gate(full_state, timesteps, global_cond, self.num_train_timesteps)
        left_context = self._context(left_state)
        right_context = self._context(right_state)
        zero_left_context = torch.zeros_like(left_context)
        zero_right_context = torch.zeros_like(right_context)

        left_marginal = self.factorized_model(
            left_state,
            timesteps,
            global_cond=self._branch_cond(global_cond, zero_right_context, arm_id=0.0, cond_mask=0.0),
        )
        right_marginal = self.factorized_model(
            right_state,
            timesteps,
            global_cond=self._branch_cond(global_cond, zero_left_context, arm_id=1.0, cond_mask=0.0),
        )
        left_cond = self.factorized_model(
            left_state,
            timesteps,
            global_cond=self._branch_cond(global_cond, right_context, arm_id=0.0, cond_mask=1.0),
        )
        right_cond = self.factorized_model(
            right_state,
            timesteps,
            global_cond=self._branch_cond(global_cond, left_context, arm_id=1.0, cond_mask=1.0),
        )

        w = gates[:, 0].view(-1, 1, 1)
        u = gates[:, 1].view(-1, 1, 1)
        left_velocity = left_marginal + w * (left_cond - left_marginal)
        right_velocity = right_marginal + u * (right_cond - right_marginal)
        gate_info = {
            "factorized_gates": gates,
            "left_marginal": left_marginal,
            "right_marginal": right_marginal,
            "left_cond": left_cond,
            "right_cond": right_cond,
        }
        self.last_gate_info = gate_info
        return left_velocity, right_velocity, gate_info

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
        if self.speed_modulation_train_only:
            global_cond = global_cond.detach()

        left_x1 = nactions[:, :, self.left_slice]
        right_x1 = nactions[:, :, self.right_slice]
        left_x0 = torch.randn_like(left_x1)
        right_x0 = torch.randn_like(right_x1)
        t = torch.rand((batch_size,), device=nactions.device, dtype=nactions.dtype)
        t_view = t.view(batch_size, 1, 1)
        timesteps = self._flow_timestep(t)
        left_xt = (1.0 - t_view) * left_x0 + t_view * left_x1
        right_xt = (1.0 - t_view) * right_x0 + t_view * right_x1
        left_target_v = left_x1 - left_x0
        right_target_v = right_x1 - right_x0

        if self.speed_modulation_train_only:
            with torch.no_grad():
                left_v, right_v, gate_info = self._predict_factorized(left_xt, right_xt, timesteps, global_cond)
        else:
            left_v, right_v, gate_info = self._predict_factorized(left_xt, right_xt, timesteps, global_cond)

        left_loss = reduce(F.mse_loss(left_v, left_target_v, reduction="none"), "b ... -> b (...)", "mean").mean()
        right_loss = reduce(F.mse_loss(right_v, right_target_v, reduction="none"), "b ... -> b (...)", "mean").mean()
        flow_loss = 0.5 * (left_loss + right_loss)
        aux_loss_dict = {
            "left_marginal_aux_loss": F.mse_loss(gate_info["left_marginal"], left_target_v),
            "right_marginal_aux_loss": F.mse_loss(gate_info["right_marginal"], right_target_v),
            "left_cond_aux_loss": F.mse_loss(gate_info["left_cond"], left_target_v),
            "right_cond_aux_loss": F.mse_loss(gate_info["right_cond"], right_target_v),
        }
        aux_loss = sum(aux_loss_dict.values()) / len(aux_loss_dict)
        total_loss = flow_loss * 0.0 if self.speed_modulation_train_only else flow_loss + self.factorized_aux_loss_weight * aux_loss

        speed_loss_dict = {}
        if self.speed_modulation_enabled and self.speed_modulation_learned and self.speed_modulation_loss_weight > 0:
            left_signal = left_v.detach() if self.speed_modulation_detach_signal else left_v
            right_signal = right_v.detach() if self.speed_modulation_detach_signal else right_v
            left_alpha = self.left_speed_head(left_signal, timesteps, global_cond=global_cond, num_train_timesteps=self.num_train_timesteps)
            right_alpha = self.right_speed_head(right_signal, timesteps, global_cond=global_cond, num_train_timesteps=self.num_train_timesteps)
            left_risk = normalize_risk(
                self.speed_modulation_coupling_risk_weight * compute_branch_mse_risk(gate_info["left_cond"], gate_info["left_marginal"])
                + self.speed_modulation_geometry_risk_weight * compute_action_risk(left_x1)
            )
            right_risk = normalize_risk(
                self.speed_modulation_coupling_risk_weight * compute_branch_mse_risk(gate_info["right_cond"], gate_info["right_marginal"])
                + self.speed_modulation_geometry_risk_weight * compute_action_risk(right_x1)
            )
            left_speed_loss, left_speed_dict = speed_modulation_loss(
                left_alpha,
                left_x1,
                self.speed_modulation_min,
                self.speed_modulation_max,
                target_weight=self.speed_modulation_target_weight,
                smooth_weight=self.speed_modulation_smooth_weight,
                fast_weight=self.speed_modulation_fast_weight,
                risk_weight=self.speed_modulation_risk_weight,
                risk=left_risk,
            )
            right_speed_loss, right_speed_dict = speed_modulation_loss(
                right_alpha,
                right_x1,
                self.speed_modulation_min,
                self.speed_modulation_max,
                target_weight=self.speed_modulation_target_weight,
                smooth_weight=self.speed_modulation_smooth_weight,
                fast_weight=self.speed_modulation_fast_weight,
                risk_weight=self.speed_modulation_risk_weight,
                risk=right_risk,
            )
            speed_loss = 0.5 * (left_speed_loss + right_speed_loss)
            total_loss = total_loss + self.speed_modulation_loss_weight * speed_loss
            speed_loss_dict = {
                "speed_modulation_loss": speed_loss,
                "left_coupling_mse_risk": left_risk.mean(),
                "right_coupling_mse_risk": right_risk.mean(),
                **{f"left_{key}": value for key, value in left_speed_dict.items()},
                **{f"right_{key}": value for key, value in right_speed_dict.items()},
            }

        gates = gate_info["factorized_gates"]
        self.last_loss_dict = {
            "flow_loss": float(flow_loss.detach().cpu()),
            "left_flow_loss": float(left_loss.detach().cpu()),
            "right_flow_loss": float(right_loss.detach().cpu()),
            "factorized_aux_loss": float(aux_loss.detach().cpu()),
            **{key: float(value.detach().cpu()) for key, value in aux_loss_dict.items()},
            "factorized_w": float(gates[:, 0].mean().detach().cpu()),
            "factorized_u": float(gates[:, 1].mean().detach().cpu()),
            **{key: float(value.detach().cpu()) for key, value in speed_loss_dict.items()},
        }
        return total_loss

    def _make_initial_latents(self, batch_size: int, generator=None):
        device = self.device
        dtype = self.dtype
        left = torch.randn((batch_size, self.horizon, self.left_action_dim), dtype=dtype, device=device, generator=generator)
        right = torch.randn((batch_size, self.horizon, self.right_action_dim), dtype=dtype, device=device, generator=generator)
        return left, right

    def _conditional_sample(self, batch_size: int, global_cond: torch.Tensor, generator=None):
        left, right = self._make_initial_latents(batch_size, generator=generator)
        dt = 1.0 / float(self.num_inference_steps)
        for step_idx in range(self.num_inference_steps):
            t = torch.full((batch_size,), step_idx / float(self.num_inference_steps), device=left.device, dtype=left.dtype)
            timesteps = self._flow_timestep(t)
            left_v, right_v, _ = self._predict_factorized(left, right, timesteps, global_cond)
            self.last_left_velocity = left_v.detach()
            self.last_right_velocity = right_v.detach()
            left = left + dt * left_v
            right = right + dt * right_v
        return left, right

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        assert "past_action" not in obs_dict
        value = next(iter(obs_dict.values()))
        batch_size = value.shape[0]
        global_cond = self._encode_obs(obs_dict)
        left_sample, right_sample = self._conditional_sample(batch_size, global_cond)
        normalized_full = self._combine_full(left_sample, right_sample)
        action_pred = self.normalizer["action"].unnormalize(normalized_full)
        action_pred_raw = action_pred.detach()

        if (
            self.speed_modulation_enabled
            and self.speed_modulation_learned
            and self.last_left_velocity is not None
            and self.last_right_velocity is not None
        ):
            zero_t = torch.zeros(batch_size, dtype=torch.long, device=global_cond.device)
            left_alpha = self.left_speed_head(self.last_left_velocity, zero_t, global_cond=global_cond, num_train_timesteps=self.num_train_timesteps)
            right_alpha = self.right_speed_head(self.last_right_velocity, zero_t, global_cond=global_cond, num_train_timesteps=self.num_train_timesteps)
            left_action = warp_action_sequence(action_pred[:, :, self.left_slice], left_alpha)
            right_action = warp_action_sequence(action_pred[:, :, self.right_slice], right_alpha)
            action_pred = self._combine_full(left_action, right_action)
            self.last_left_speed_alpha = left_alpha.detach()
            self.last_right_speed_alpha = right_alpha.detach()

        start = self.n_obs_steps - 1
        end = start + self.n_action_steps
        result = {"action": action_pred[:, start:end], "action_pred": action_pred}
        if self.last_gate_info is not None:
            result["factorized_gates"] = self.last_gate_info["factorized_gates"]
        if self.last_left_speed_alpha is not None and self.last_right_speed_alpha is not None:
            result["left_speed_alpha"] = self.last_left_speed_alpha
            result["right_speed_alpha"] = self.last_right_speed_alpha
            result["action_pred_raw"] = action_pred_raw
        return result
