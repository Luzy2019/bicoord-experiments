from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import reduce

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy.model.vision.multi_image_obs_encoder import MultiImageObsEncoder
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.policy.speed_modulation import SpeedModulationHead, speed_modulation_loss, warp_action_sequence


class FlowUnetImagePolicy(BaseImagePolicy):
    """Conditional flow-matching version of the original 1-UNet DP policy."""

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
        **kwargs,
    ):
        super().__init__()
        action_shape = shape_meta["action"]["shape"]
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        obs_feature_dim = obs_encoder.output_shape()[0]

        input_dim = action_dim + obs_feature_dim
        global_cond_dim = None
        if obs_as_global_cond:
            input_dim = action_dim
            global_cond_dim = obs_feature_dim * n_obs_steps

        self.model = ConditionalUnet1D(
            input_dim=input_dim,
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
            action_dim=action_dim,
            obs_dim=0 if obs_as_global_cond else obs_feature_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False,
        )
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.kwargs = kwargs

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
        self.speed_head = SpeedModulationHead(
            action_dim=action_dim,
            cond_dim=0 if global_cond_dim is None else global_cond_dim,
            hidden_dim=int(speed_modulation_hidden_dim),
            alpha_min=self.speed_modulation_min,
            alpha_max=self.speed_modulation_max,
            init_alpha=1.0,
        )
        self.last_velocity = None
        self.last_speed_alpha = None
        self.last_loss_dict = {}

    def _flow_timestep(self, t: torch.Tensor) -> torch.Tensor:
        return t * float(max(self.num_train_timesteps - 1, 1))

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def conditional_sample(self, condition_data, condition_mask, local_cond=None, global_cond=None, generator=None):
        trajectory = torch.randn(
            size=condition_data.shape,
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator,
        )
        dt = 1.0 / float(self.num_inference_steps)
        for step_idx in range(self.num_inference_steps):
            trajectory[condition_mask] = condition_data[condition_mask]
            t = torch.full(
                (trajectory.shape[0],),
                fill_value=step_idx / float(self.num_inference_steps),
                device=trajectory.device,
                dtype=trajectory.dtype,
            )
            velocity = self.model(
                trajectory,
                self._flow_timestep(t),
                local_cond=local_cond,
                global_cond=global_cond,
            )
            self.last_velocity = velocity.detach()
            trajectory = trajectory + dt * velocity
        trajectory[condition_mask] = condition_data[condition_mask]
        return trajectory

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        assert "past_action" not in obs_dict
        nobs = self.normalizer.normalize(obs_dict)
        value = next(iter(nobs.values()))
        batch_size = value.shape[0]
        horizon = self.horizon
        action_dim = self.action_dim
        n_obs_steps = self.n_obs_steps
        device = self.device
        dtype = self.dtype

        local_cond = None
        global_cond = None
        if self.obs_as_global_cond:
            this_nobs = dict_apply(nobs, lambda x: x[:, :n_obs_steps, ...].reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            global_cond = nobs_features.reshape(batch_size, -1)
            cond_data = torch.zeros(size=(batch_size, horizon, action_dim), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            this_nobs = dict_apply(nobs, lambda x: x[:, :n_obs_steps, ...].reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs).reshape(batch_size, n_obs_steps, -1)
            cond_data = torch.zeros(size=(batch_size, horizon, action_dim + self.obs_feature_dim), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:, :n_obs_steps, action_dim:] = nobs_features
            cond_mask[:, :n_obs_steps, action_dim:] = True

        nsample = self.conditional_sample(cond_data, cond_mask, local_cond=local_cond, global_cond=global_cond)
        naction_pred = nsample[..., :action_dim]
        action_pred = self.normalizer["action"].unnormalize(naction_pred)
        action_pred_raw = action_pred.detach()

        if self.speed_modulation_enabled and self.speed_modulation_learned and self.last_velocity is not None:
            zero_t = torch.zeros(batch_size, dtype=torch.long, device=device)
            alpha = self.speed_head(
                self.last_velocity[..., :action_dim],
                zero_t,
                global_cond=global_cond,
                num_train_timesteps=self.num_train_timesteps,
            )
            self.last_speed_alpha = alpha.detach()
            action_pred = warp_action_sequence(action_pred, alpha)

        start = n_obs_steps - 1
        end = start + self.n_action_steps
        result = {"action": action_pred[:, start:end], "action_pred": action_pred}
        if self.last_speed_alpha is not None:
            result["speed_alpha"] = self.last_speed_alpha
            result["action_pred_raw"] = action_pred_raw
        return result

    def compute_loss(self, batch):
        assert "valid_mask" not in batch
        nobs = self.normalizer.normalize(batch["obs"])
        nactions = self.normalizer["action"].normalize(batch["action"])
        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]

        local_cond = None
        global_cond = None
        trajectory = nactions
        cond_data = trajectory
        if self.obs_as_global_cond:
            this_nobs = dict_apply(nobs, lambda x: x[:, :self.n_obs_steps, ...].reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            global_cond = nobs_features.reshape(batch_size, -1)
            if self.speed_modulation_train_only:
                global_cond = global_cond.detach()
        else:
            this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs).reshape(batch_size, horizon, -1)
            cond_data = torch.cat([nactions, nobs_features], dim=-1)
            trajectory = cond_data.detach()

        condition_mask = self.mask_generator(trajectory.shape)
        x0 = torch.randn_like(trajectory)
        x1 = trajectory
        t = torch.rand((batch_size,), device=trajectory.device, dtype=trajectory.dtype)
        t_view = t.view(batch_size, *([1] * (trajectory.ndim - 1)))
        xt = (1.0 - t_view) * x0 + t_view * x1
        target_velocity = x1 - x0
        xt[condition_mask] = cond_data[condition_mask]

        if self.speed_modulation_train_only:
            with torch.no_grad():
                pred_velocity = self.model(xt, self._flow_timestep(t), local_cond=local_cond, global_cond=global_cond)
        else:
            pred_velocity = self.model(xt, self._flow_timestep(t), local_cond=local_cond, global_cond=global_cond)

        loss_mask = ~condition_mask
        flow_loss = F.mse_loss(pred_velocity, target_velocity, reduction="none")
        flow_loss = flow_loss * loss_mask.type(flow_loss.dtype)
        flow_loss = reduce(flow_loss, "b ... -> b (...)", "mean").mean()
        total_loss = flow_loss * 0.0 if self.speed_modulation_train_only else flow_loss
        self.last_loss_dict = {"flow_loss": float(flow_loss.detach().cpu())}

        if self.speed_modulation_enabled and self.speed_modulation_learned and self.speed_modulation_loss_weight > 0:
            speed_signal = pred_velocity.detach() if self.speed_modulation_detach_signal else pred_velocity
            alpha = self.speed_head(
                speed_signal[..., :self.action_dim],
                self._flow_timestep(t),
                global_cond=global_cond,
                num_train_timesteps=self.num_train_timesteps,
            )
            speed_loss, speed_loss_dict = speed_modulation_loss(
                alpha=alpha,
                actions=nactions,
                alpha_min=self.speed_modulation_min,
                alpha_max=self.speed_modulation_max,
                target_weight=self.speed_modulation_target_weight,
                smooth_weight=self.speed_modulation_smooth_weight,
                fast_weight=self.speed_modulation_fast_weight,
                risk_weight=self.speed_modulation_risk_weight,
            )
            total_loss = total_loss + self.speed_modulation_loss_weight * speed_loss
            self.last_loss_dict.update({
                "speed_modulation_loss": float(speed_loss.detach().cpu()),
                **{key: float(value.detach().cpu()) for key, value in speed_loss_dict.items()},
            })
        return total_loss
