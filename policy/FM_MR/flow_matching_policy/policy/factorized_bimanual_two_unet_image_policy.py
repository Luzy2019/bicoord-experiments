from typing import Dict, Optional, Tuple

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
    return slice(start, start + dim)


class FactorizedBimanualGate(nn.Module):
    """Predicts w(t), u(t) for conditional-vs-marginal bimanual correction."""

    def __init__(
        self,
        action_dim: int,
        cond_dim: int,
        hidden_dim: int = 256,
        init_bias: float = -2.0,
    ):
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

    def forward(
        self,
        full_noisy: torch.Tensor,
        timesteps: torch.Tensor,
        global_cond: Optional[torch.Tensor],
        num_train_timesteps: int,
    ) -> torch.Tensor:
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
            global_cond = torch.zeros(
                full_noisy.shape[0],
                0,
                device=full_noisy.device,
                dtype=full_noisy.dtype,
            )
        return torch.sigmoid(self.net(torch.cat([pooled, global_cond, time_feat], dim=-1)))


class FactorizedBimanualTwoUnetImagePolicy(BaseImagePolicy):
    """
    Factorized bimanual diffusion policy with two arm-specific denoisers.

    The left denoiser handles:
    - p(a_left | obs)
    - p(a_left | obs, a_right)

    The right denoiser handles:
    - p(a_right | obs)
    - p(a_right | obs, a_left)

    A cond_mask flag tells each arm denoiser whether the other-arm context is
    available. Marginal mode uses zero context and cond_mask=0. Conditional
    mode uses the other-arm context and cond_mask=1.
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
        factorized_hidden_dim=256,
        factorized_gate_init_bias=-2.0,
        factorized_aux_loss_weight=0.25,
        **kwargs,
    ):
        super().__init__()
        if not obs_as_global_cond:
            raise ValueError("FactorizedBimanualTwoUnetImagePolicy requires obs_as_global_cond=True")

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

        global_cond_dim = obs_feature_dim * n_obs_steps
        left_cond_dim = global_cond_dim + self.right_action_dim * 2 + 1
        right_cond_dim = global_cond_dim + self.left_action_dim * 2 + 1

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
        self.factorized_gate = FactorizedBimanualGate(
            action_dim=action_dim,
            cond_dim=global_cond_dim,
            hidden_dim=factorized_hidden_dim,
            init_bias=factorized_gate_init_bias,
        )

        self.obs_encoder = obs_encoder
        # Kept for config/checkpoint compatibility; flow matching does not use
        # DDPM scheduler transitions.
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
        self.last_gate_info = None
        self.last_loss_dict = {}
        self.last_debug_info = None

        if num_inference_steps is None:
            num_inference_steps = getattr(getattr(noise_scheduler, "config", None), "num_train_timesteps", 100)
        self.num_inference_steps = num_inference_steps

    def _encode_obs(self, obs_dict):
        nobs = self.normalizer.normalize(obs_dict)
        value = next(iter(nobs.values()))
        batch_size = value.shape[0]
        this_nobs = dict_apply(nobs, lambda x: x[:, :self.n_obs_steps, ...].reshape(-1, *x.shape[2:]))
        nobs_features = self.obs_encoder(this_nobs)
        return nobs_features.reshape(batch_size, -1)

    def _combine_full(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        full = torch.zeros(
            left.shape[0],
            self.horizon,
            self.action_dim,
            device=left.device,
            dtype=left.dtype,
        )
        full[:, :, self.left_slice] = left
        full[:, :, self.right_slice] = right
        return full

    @staticmethod
    def _context(traj: torch.Tensor) -> torch.Tensor:
        return torch.cat([traj.mean(dim=1), traj.std(dim=1, unbiased=False)], dim=-1)

    @staticmethod
    def _flag(batch_size: int, value: float, reference: torch.Tensor) -> torch.Tensor:
        return torch.full(
            (batch_size, 1),
            fill_value=value,
            device=reference.device,
            dtype=reference.dtype,
        )

    @staticmethod
    def _expand_timesteps(timesteps, batch_size: int, device: torch.device) -> torch.Tensor:
        if not torch.is_tensor(timesteps):
            return torch.full((batch_size,), int(timesteps), dtype=torch.long, device=device)
        timesteps = timesteps.to(device)
        if timesteps.ndim == 0:
            return timesteps.expand(batch_size)
        if timesteps.shape[0] == 1:
            return timesteps.expand(batch_size)
        return timesteps

    def _branch_cond(
        self,
        global_cond: torch.Tensor,
        other_context: torch.Tensor,
        cond_mask: float,
    ) -> torch.Tensor:
        batch_size = global_cond.shape[0]
        return torch.cat(
            [
                global_cond,
                other_context,
                self._flag(batch_size, cond_mask, global_cond),
            ],
            dim=-1,
        )

    def _predict_factorized(
        self,
        left_noisy: torch.Tensor,
        right_noisy: torch.Tensor,
        timesteps: torch.Tensor,
        global_cond: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        full_noisy = self._combine_full(left_noisy, right_noisy)
        gates = self.factorized_gate(
            full_noisy=full_noisy,
            timesteps=timesteps,
            global_cond=global_cond,
            num_train_timesteps=1,
        )

        left_context = self._context(left_noisy)
        right_context = self._context(right_noisy)
        zero_left_context = torch.zeros_like(left_context)
        zero_right_context = torch.zeros_like(right_context)

        batch_size = left_noisy.shape[0]
        branch_timesteps = self._expand_timesteps(timesteps, batch_size, left_noisy.device)
        branch_timesteps = torch.cat([branch_timesteps, branch_timesteps], dim=0)

        left_outputs = self.left_model(
            torch.cat([left_noisy, left_noisy], dim=0),
            branch_timesteps,
            global_cond=torch.cat(
                [
                    self._branch_cond(global_cond, zero_right_context, cond_mask=0.0),
                    self._branch_cond(global_cond, right_context, cond_mask=1.0),
                ],
                dim=0,
            ),
        )
        right_outputs = self.right_model(
            torch.cat([right_noisy, right_noisy], dim=0),
            branch_timesteps,
            global_cond=torch.cat(
                [
                    self._branch_cond(global_cond, zero_left_context, cond_mask=0.0),
                    self._branch_cond(global_cond, left_context, cond_mask=1.0),
                ],
                dim=0,
            ),
        )
        left_marginal, left_cond = left_outputs.chunk(2, dim=0)
        right_marginal, right_cond = right_outputs.chunk(2, dim=0)

        w = gates[:, 0].view(-1, 1, 1)
        u = gates[:, 1].view(-1, 1, 1)
        left_pred = left_marginal + w * (left_cond - left_marginal)
        right_pred = right_marginal + u * (right_cond - right_marginal)

        gate_info = {
            "factorized_gates": gates,
            "left_marginal": left_marginal,
            "right_marginal": right_marginal,
            "left_cond": left_cond,
            "right_cond": right_cond,
        }
        self.last_gate_info = gate_info
        return left_pred, right_pred, gate_info

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

        left_target = nactions[:, :, self.left_slice]
        right_target = nactions[:, :, self.right_slice]
        left_noise = torch.randn_like(left_target)
        right_noise = torch.randn_like(right_target)
        timesteps = torch.rand((batch_size,), device=nactions.device, dtype=nactions.dtype)
        t_view = timesteps.view(batch_size, 1, 1)

        left_noisy = (1.0 - t_view) * left_noise + t_view * left_target
        right_noisy = (1.0 - t_view) * right_noise + t_view * right_target
        left_pred, right_pred, gate_info = self._predict_factorized(left_noisy, right_noisy, timesteps, global_cond)

        left_label = left_target - left_noise
        right_label = right_target - right_noise

        left_loss = reduce(F.mse_loss(left_pred, left_label, reduction="none"), "b ... -> b (...)", "mean").mean()
        right_loss = reduce(F.mse_loss(right_pred, right_label, reduction="none"), "b ... -> b (...)", "mean").mean()
        flow_matching_loss = 0.5 * (left_loss + right_loss)

        aux_loss_dict = {
            "left_marginal_aux_loss": F.mse_loss(gate_info["left_marginal"], left_label),
            "right_marginal_aux_loss": F.mse_loss(gate_info["right_marginal"], right_label),
            "left_cond_aux_loss": F.mse_loss(gate_info["left_cond"], left_label),
            "right_cond_aux_loss": F.mse_loss(gate_info["right_cond"], right_label),
        }
        aux_loss = sum(aux_loss_dict.values()) / len(aux_loss_dict)
        total_loss = flow_matching_loss + self.factorized_aux_loss_weight * aux_loss

        gates = gate_info["factorized_gates"]
        self.last_loss_dict = {
            "flow_matching_loss": float(flow_matching_loss.detach().cpu()),
            "left_flow_matching_loss": float(left_loss.detach().cpu()),
            "right_flow_matching_loss": float(right_loss.detach().cpu()),
            "factorized_aux_loss": float(aux_loss.detach().cpu()),
            **{key: float(value.detach().cpu()) for key, value in aux_loss_dict.items()},
            "factorized_w": float(gates[:, 0].mean().detach().cpu()),
            "factorized_u": float(gates[:, 1].mean().detach().cpu()),
        }
        return total_loss

    def _make_initial_latents(self, batch_size: int, generator=None):
        device = self.device
        dtype = self.dtype
        left = torch.randn(
            size=(batch_size, self.horizon, self.left_action_dim),
            dtype=dtype,
            device=device,
            generator=generator,
        )
        right = torch.randn(
            size=(batch_size, self.horizon, self.right_action_dim),
            dtype=dtype,
            device=device,
            generator=generator,
        )
        return left, right

    def _conditional_sample(
        self,
        batch_size: int,
        global_cond: torch.Tensor,
        generator=None,
        initial_left: Optional[torch.Tensor] = None,
        initial_right: Optional[torch.Tensor] = None,
        mode: str = "final",
        collect_debug: bool = True,
    ):
        if initial_left is None or initial_right is None:
            left, right = self._make_initial_latents(batch_size, generator=generator)
        else:
            left = initial_left.clone()
            right = initial_right.clone()

        num_steps = max(int(self.num_inference_steps), 1)
        dt = 1.0 / num_steps
        debug_steps = []
        for step_idx in range(num_steps):
            t = torch.full(
                (batch_size,),
                step_idx / num_steps,
                dtype=left.dtype,
                device=left.device,
            )
            left_pred, right_pred, gate_info = self._predict_factorized(left, right, t, global_cond)
            if collect_debug:
                debug_steps.append({
                    "time": float(t[0].detach().cpu()),
                    "factorized_gates": gate_info["factorized_gates"].detach().cpu(),
                    "left_marginal": gate_info["left_marginal"].detach().cpu(),
                    "right_marginal": gate_info["right_marginal"].detach().cpu(),
                    "left_cond": gate_info["left_cond"].detach().cpu(),
                    "right_cond": gate_info["right_cond"].detach().cpu(),
                    "left_pred": left_pred.detach().cpu(),
                    "right_pred": right_pred.detach().cpu(),
                })
            if mode == "marginal":
                left_pred = gate_info["left_marginal"]
                right_pred = gate_info["right_marginal"]
            elif mode == "conditional":
                left_pred = gate_info["left_cond"]
                right_pred = gate_info["right_cond"]
            elif mode != "final":
                raise ValueError(f"Unsupported factorized sample mode {mode}")
            left = left + dt * left_pred
            right = right + dt * right_pred
        if collect_debug:
            self.last_debug_info = debug_steps
        return left, right

    def _sample_debug_action_variants(self, batch_size: int, global_cond: torch.Tensor):
        initial_left, initial_right = self._make_initial_latents(batch_size)
        variants = {}
        for mode in ("marginal", "conditional", "final"):
            left, right = self._conditional_sample(
                batch_size,
                global_cond,
                initial_left=initial_left,
                initial_right=initial_right,
                mode=mode,
                collect_debug=False,
            )
            action = self.normalizer["action"].unnormalize(self._combine_full(left, right))
            variants[f"{mode}_action"] = action.detach().cpu()
            variants[f"{mode}_left_action"] = action[:, :, self.left_slice].detach().cpu()
            variants[f"{mode}_right_action"] = action[:, :, self.right_slice].detach().cpu()
        self.last_action_debug_info = variants

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        assert "past_action" not in obs_dict
        value = next(iter(obs_dict.values()))
        batch_size = value.shape[0]
        global_cond = self._encode_obs(obs_dict)

        left_sample, right_sample = self._conditional_sample(batch_size, global_cond)
        if getattr(self, "factorized_debug_actions", False):
            self._sample_debug_action_variants(batch_size, global_cond)
        normalized_full = self._combine_full(left_sample, right_sample)
        action_pred = self.normalizer["action"].unnormalize(normalized_full)

        start = self.n_obs_steps - 1
        end = start + self.n_action_steps
        result = {
            "action": action_pred[:, start:end],
            "action_pred": action_pred,
        }
        if self.last_gate_info is not None:
            result["factorized_gates"] = self.last_gate_info["factorized_gates"]
        return result
