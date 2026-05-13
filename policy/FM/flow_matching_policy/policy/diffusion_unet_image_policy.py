from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce

from flow_matching_policy.model.common.normalizer import LinearNormalizer
from flow_matching_policy.policy.base_image_policy import BaseImagePolicy
from flow_matching_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from flow_matching_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from flow_matching_policy.model.vision.multi_image_obs_encoder import MultiImageObsEncoder
from flow_matching_policy.common.pytorch_util import dict_apply
from flow_matching_policy.policy.speed_modulation import (
    SpeedModulationHead,
    compute_speed_alpha,
    speed_modulation_loss,
    warp_action_sequence,
)


class DiffusionUnetImagePolicy(BaseImagePolicy):

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
        speed_modulation_strength=1.0,
        speed_modulation_min=0.5,
        speed_modulation_max=2.0,
        speed_modulation_smooth=3,
        speed_modulation_learned=True,
        speed_modulation_hidden_dim=128,
        speed_modulation_loss_weight=0.01,
        speed_modulation_target_weight=1.0,
        speed_modulation_smooth_weight=0.1,
        speed_modulation_fast_weight=0.01,
        speed_modulation_risk_weight=0.1,
        speed_modulation_detach_signal=True,
        speed_modulation_train_only=False,
        # parameters passed to step
        **kwargs,
    ):
        super().__init__()

        # parse shapes
        action_shape = shape_meta["action"]["shape"]
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        # get feature dim
        obs_feature_dim = obs_encoder.output_shape()[0]

        # create flow model
        input_dim = action_dim + obs_feature_dim
        global_cond_dim = None
        if obs_as_global_cond:
            input_dim = action_dim
            global_cond_dim = obs_feature_dim * n_obs_steps

        model = ConditionalUnet1D(
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
        self.model = model
        # Kept for config/checkpoint compatibility; flow matching does not use
        # DDPM transitions for either training or sampling.
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
        self.speed_modulation_enabled = bool(speed_modulation_enabled)
        self.speed_modulation_strength = float(speed_modulation_strength)
        self.speed_modulation_min = float(speed_modulation_min)
        self.speed_modulation_max = float(speed_modulation_max)
        self.speed_modulation_smooth = int(speed_modulation_smooth)
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
        self.last_speed_alpha = None
        self.last_action_pred_raw = None
        self.last_denoise_output = None
        self.last_loss_dict = {}

        if num_inference_steps is None:
            num_inference_steps = getattr(getattr(noise_scheduler, "config", None), "num_train_timesteps", 100)
        self.num_inference_steps = num_inference_steps

    # ========= inference  ============
    def conditional_sample(
        self,
        condition_data,
        condition_mask,
        local_cond=None,
        global_cond=None,
        generator=None,
        # kept for compatibility with old scheduler kwargs
        **kwargs,
    ):
        model = self.model

        trajectory = torch.randn(
            size=condition_data.shape,
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator,
        )

        num_steps = max(int(self.num_inference_steps), 1)
        dt = 1.0 / num_steps
        batch_size = trajectory.shape[0]
        for step_idx in range(num_steps):
            t = torch.full(
                (batch_size,),
                step_idx / num_steps,
                dtype=trajectory.dtype,
                device=trajectory.device,
            )
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]

            # 2. predict vector field dx / dt
            model_output = model(trajectory, t, local_cond=local_cond, global_cond=global_cond)
            self.last_denoise_output = model_output.detach()

            # 3. Euler integrate from the noise distribution to the action distribution.
            trajectory = trajectory + dt * model_output

        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]

        return trajectory

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        assert "past_action" not in obs_dict  # not implemented yet
        # normalize input
        nobs = self.normalizer.normalize(obs_dict)
        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        Do = self.obs_feature_dim
        To = self.n_obs_steps

        # build input
        device = self.device
        dtype = self.dtype

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        if self.obs_as_global_cond:
            # condition through global feature
            this_nobs = dict_apply(nobs, lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, Do
            global_cond = nobs_features.reshape(B, -1)
            # empty data for action
            cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            # condition through impainting
            this_nobs = dict_apply(nobs, lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(B, To, -1)
            cond_data = torch.zeros(size=(B, T, Da + Do), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:, :To, Da:] = nobs_features
            cond_mask[:, :To, Da:] = True

        # run sampling
        nsample = self.conditional_sample(
            cond_data,
            cond_mask,
            local_cond=local_cond,
            global_cond=global_cond,
            **self.kwargs,
        )

        # unnormalize prediction
        naction_pred = nsample[..., :Da]
        action_pred = self.normalizer["action"].unnormalize(naction_pred)
        self.last_action_pred_raw = action_pred.detach()
        if self.speed_modulation_enabled and self.last_denoise_output is not None:
            if self.speed_modulation_learned:
                speed_alpha = self.speed_head(
                    self.last_denoise_output[..., :Da],
                    torch.ones(B, dtype=dtype, device=device),
                    global_cond=global_cond,
                    num_train_timesteps=1,
                )
            else:
                speed_alpha = compute_speed_alpha(
                    self.last_denoise_output[..., :Da],
                    strength=self.speed_modulation_strength,
                    alpha_min=self.speed_modulation_min,
                    alpha_max=self.speed_modulation_max,
                    smooth_kernel=self.speed_modulation_smooth,
                )
            self.last_speed_alpha = speed_alpha.detach()
            action_pred = warp_action_sequence(action_pred, speed_alpha)

        # get action
        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:, start:end]

        result = {"action": action, "action_pred": action_pred}
        if self.last_speed_alpha is not None:
            result["speed_alpha"] = self.last_speed_alpha
            result["action_pred_raw"] = self.last_action_pred_raw
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        # normalize input
        assert "valid_mask" not in batch
        nobs = self.normalizer.normalize(batch["obs"])
        nactions = self.normalizer["action"].normalize(batch["action"])
        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        trajectory = nactions
        cond_data = trajectory
        if self.obs_as_global_cond:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, lambda x: x[:, :self.n_obs_steps, ...].reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, Do
            global_cond = nobs_features.reshape(batch_size, -1)
            if self.speed_modulation_train_only:
                global_cond = global_cond.detach()
        else:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(batch_size, horizon, -1)
            cond_data = torch.cat([nactions, nobs_features], dim=-1)
            trajectory = cond_data.detach()

        # generate impainting mask
        condition_mask = self.mask_generator(trajectory.shape)

        # Flow matching trains a vector field from Gaussian noise x0 to data x1.
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        bsz = trajectory.shape[0]
        timesteps = torch.rand((bsz,), device=trajectory.device, dtype=trajectory.dtype)
        t_view = timesteps.view(bsz, *([1] * (trajectory.ndim - 1)))
        noisy_trajectory = (1.0 - t_view) * noise + t_view * trajectory
        target = trajectory - noise

        # compute loss mask
        loss_mask = ~condition_mask

        # apply conditioning
        noisy_trajectory[condition_mask] = cond_data[condition_mask]

        # Predict the flow velocity.
        if self.speed_modulation_train_only:
            with torch.no_grad():
                pred = self.model(noisy_trajectory, timesteps, local_cond=local_cond, global_cond=global_cond)
        else:
            pred = self.model(noisy_trajectory, timesteps, local_cond=local_cond, global_cond=global_cond)

        loss = F.mse_loss(pred, target, reduction="none")
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, "b ... -> b (...)", "mean")
        flow_matching_loss = loss.mean()
        total_loss = flow_matching_loss * 0.0 if self.speed_modulation_train_only else flow_matching_loss
        self.last_loss_dict = {"flow_matching_loss": float(flow_matching_loss.detach().cpu())}
        if self.speed_modulation_enabled and self.speed_modulation_learned and self.speed_modulation_loss_weight > 0:
            speed_signal = pred.detach() if self.speed_modulation_detach_signal else pred
            speed_alpha = self.speed_head(
                speed_signal[..., :self.action_dim],
                timesteps,
                global_cond=global_cond,
                num_train_timesteps=1,
            )
            speed_loss, speed_loss_dict = speed_modulation_loss(
                alpha=speed_alpha,
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
