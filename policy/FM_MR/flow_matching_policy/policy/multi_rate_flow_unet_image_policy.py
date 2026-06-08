import copy
from math import ceil
from typing import Dict, Optional

import dill
import torch
import torch.nn.functional as F
from einops import reduce

from flow_matching_policy.common.pytorch_util import dict_apply
from flow_matching_policy.model.common.normalizer import LinearNormalizer
from flow_matching_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from flow_matching_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from flow_matching_policy.model.vision.multi_image_obs_encoder import MultiImageObsEncoder
from flow_matching_policy.policy.base_image_policy import BaseImagePolicy
from flow_matching_policy.policy.diffusion_unet_image_policy import DiffusionUnetImagePolicy


def _slice_from_start_dim(start: int, dim: int) -> slice:
    return slice(start, start + dim)


class MultiRateFlowUnetImagePolicy(BaseImagePolicy):
    """Direct multi-rate Flow Matching policy for role-asymmetric bimanual actions."""

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
        main_arm="left",
        assist_arm="right",
        assist_stride=5,
        min_assist_horizon=4,
        assist_upsample_mode="hold",
        teacher=None,
        **kwargs,
    ):
        super().__init__()
        if not obs_as_global_cond:
            raise ValueError("MultiRateFlowUnetImagePolicy requires obs_as_global_cond=True")

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
        self.main_arm = str(main_arm)
        self.assist_arm = str(assist_arm)
        if {self.main_arm, self.assist_arm} != {"left", "right"}:
            raise ValueError("main_arm and assist_arm must be left/right in either order")

        self.horizon = int(horizon)
        self.assist_stride = max(int(assist_stride), 1)
        requested_assist_horizon = int(ceil(self.horizon / self.assist_stride))
        self.assist_horizon = min(
            self.horizon,
            max(requested_assist_horizon, int(min_assist_horizon)),
        )
        self.assist_upsample_mode = str(assist_upsample_mode)
        if self.assist_upsample_mode not in ("hold", "linear"):
            raise ValueError("assist_upsample_mode must be hold or linear")

        global_cond_dim = obs_feature_dim * n_obs_steps
        self.main_dim = self.left_action_dim if self.main_arm == "left" else self.right_action_dim
        self.assist_dim = self.right_action_dim if self.assist_arm == "right" else self.left_action_dim

        self.main_model = ConditionalUnet1D(
            input_dim=self.main_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale,
        )
        self.assist_model = ConditionalUnet1D(
            input_dim=self.assist_dim,
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
            obs_dim=0,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False,
        )
        self.normalizer = LinearNormalizer()
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.kwargs = kwargs

        teacher = teacher or {}
        self.teacher_enabled = bool(teacher.get("enabled", False))
        self.teacher_ckpt_path = teacher.get("ckpt_path", None)
        self.dense_loss_weight = float(teacher.get("dense_loss_weight", 1.0))
        self.demo_loss_weight = float(teacher.get("demo_loss_weight", 0.5))
        self.assist_smooth_weight = float(teacher.get("assist_smooth_weight", 0.01))
        self.teacher_policy_ref = []
        if self.teacher_enabled and self.teacher_ckpt_path:
            self.teacher_policy_ref.append(
                self._load_teacher(
                    shape_meta=shape_meta,
                    noise_scheduler=noise_scheduler,
                    obs_encoder=obs_encoder,
                    horizon=horizon,
                    n_action_steps=n_action_steps,
                    n_obs_steps=n_obs_steps,
                    num_inference_steps=num_inference_steps,
                    obs_as_global_cond=obs_as_global_cond,
                    diffusion_step_embed_dim=diffusion_step_embed_dim,
                    down_dims=down_dims,
                    kernel_size=kernel_size,
                    n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale,
                    kwargs=kwargs,
                )
            )

        self.last_loss_dict = {}
        self.last_action_pred_raw = None
        self.last_main_chunk = None
        self.last_assist_chunk = None

        if num_inference_steps is None:
            num_inference_steps = getattr(getattr(noise_scheduler, "config", None), "num_train_timesteps", 100)
        self.num_inference_steps = int(num_inference_steps)

    def _load_teacher(self, **ctor_kwargs):
        teacher_obs_encoder = copy.deepcopy(ctor_kwargs.pop("obs_encoder"))
        policy = DiffusionUnetImagePolicy(
            obs_encoder=teacher_obs_encoder,
            **{k: v for k, v in ctor_kwargs.items() if k != "kwargs"},
            **ctor_kwargs.get("kwargs", {}),
        )
        payload = torch.load(open(self.teacher_ckpt_path, "rb"), pickle_module=dill, map_location="cpu")
        state = payload["state_dicts"].get("ema_model", payload["state_dicts"].get("model"))
        policy.load_state_dict(state, strict=False)
        policy.eval()
        policy.requires_grad_(False)
        return policy

    @property
    def main_slice(self):
        return self.left_slice if self.main_arm == "left" else self.right_slice

    @property
    def assist_slice(self):
        return self.right_slice if self.assist_arm == "right" else self.left_slice

    def _encode_global_cond(self, obs_dict):
        nobs = self.normalizer.normalize(obs_dict)
        value = next(iter(nobs.values()))
        batch_size = value.shape[0]
        this_nobs = dict_apply(nobs, lambda x: x[:, :self.n_obs_steps, ...].reshape(-1, *x.shape[2:]))
        nobs_features = self.obs_encoder(this_nobs)
        return nobs_features.reshape(batch_size, -1)

    def _assist_indices(self, device):
        if self.assist_upsample_mode == "linear":
            return torch.linspace(0, self.horizon - 1, self.assist_horizon, device=device).round().long()
        return torch.div(
            torch.arange(self.assist_horizon, device=device) * self.horizon,
            self.assist_horizon,
            rounding_mode="floor",
        ).clamp_max(self.horizon - 1)

    def _downsample_assist(self, dense_actions):
        return dense_actions[:, self._assist_indices(dense_actions.device)]

    def _upsample_assist(self, assist_chunk):
        if self.assist_upsample_mode == "linear":
            return F.interpolate(
                assist_chunk.transpose(1, 2),
                size=self.horizon,
                mode="linear",
                align_corners=True,
            ).transpose(1, 2)
        dense_idx = torch.div(
            torch.arange(self.horizon, device=assist_chunk.device) * self.assist_horizon,
            self.horizon,
            rounding_mode="floor",
        ).clamp_max(self.assist_horizon - 1)
        return assist_chunk[:, dense_idx]

    def _unroll(self, main_chunk, assist_chunk):
        dense = torch.zeros(
            main_chunk.shape[0],
            self.horizon,
            self.action_dim,
            device=main_chunk.device,
            dtype=main_chunk.dtype,
        )
        dense[:, :, self.main_slice] = main_chunk
        dense[:, :, self.assist_slice] = self._upsample_assist(assist_chunk)
        return dense

    def _unnormalize_group(self, group_chunk, group_slice):
        full = torch.zeros(
            group_chunk.shape[0],
            group_chunk.shape[1],
            self.action_dim,
            device=group_chunk.device,
            dtype=group_chunk.dtype,
        )
        full[:, :, group_slice] = group_chunk
        return self.normalizer["action"].unnormalize(full)[:, :, group_slice]

    def conditional_sample(self, model, shape, global_cond=None, generator=None):
        trajectory = torch.randn(shape, dtype=self.dtype, device=self.device, generator=generator)
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
            trajectory = trajectory + dt * model(trajectory, t, global_cond=global_cond)
        return trajectory

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        assert "past_action" not in obs_dict
        global_cond = self._encode_global_cond(obs_dict)
        batch_size = global_cond.shape[0]
        main = self.conditional_sample(
            self.main_model,
            (batch_size, self.horizon, self.main_dim),
            global_cond=global_cond,
        )
        assist = self.conditional_sample(
            self.assist_model,
            (batch_size, self.assist_horizon, self.assist_dim),
            global_cond=global_cond,
        )
        naction_pred = self._unroll(main, assist)
        action_pred = self.normalizer["action"].unnormalize(naction_pred)
        self.last_action_pred_raw = action_pred.detach()
        self.last_main_chunk = main.detach()
        self.last_assist_chunk = assist.detach()

        start = self.n_obs_steps - 1
        end = start + self.n_action_steps
        result = {
            "action": action_pred[:, start:end],
            "action_pred": action_pred,
            "multi_rate_action_pred": action_pred,
            "main_chunk": self._unnormalize_group(main, self.main_slice),
            "assist_chunk": self._unnormalize_group(assist, self.assist_slice),
            "assist_stride": torch.tensor(self.assist_stride, device=action_pred.device),
        }
        return result

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())
        if self.teacher_policy_ref:
            self.teacher_policy_ref[0].set_normalizer(normalizer)

    def _teacher_target(self, batch):
        if not self.teacher_policy_ref:
            return None
        teacher_policy = self.teacher_policy_ref[0]
        teacher_policy.to(self.device)
        teacher_policy.eval()
        with torch.no_grad():
            teacher_out = teacher_policy.predict_action(batch["obs"])
            return self.normalizer["action"].normalize(teacher_out["action_pred"])

    def compute_loss(self, batch):
        nobs = self.normalizer.normalize(batch["obs"])
        nactions = self.normalizer["action"].normalize(batch["action"])
        batch_size = nactions.shape[0]
        this_nobs = dict_apply(nobs, lambda x: x[:, :self.n_obs_steps, ...].reshape(-1, *x.shape[2:]))
        nobs_features = self.obs_encoder(this_nobs)
        global_cond = nobs_features.reshape(batch_size, -1)

        teacher_target = self._teacher_target(batch)
        target_dense = teacher_target if teacher_target is not None else nactions
        target_main = target_dense[:, :, self.main_slice]
        target_assist = self._downsample_assist(target_dense[:, :, self.assist_slice])

        main_noise = torch.randn_like(target_main)
        assist_noise = torch.randn_like(target_assist)
        t = torch.rand((batch_size,), device=target_dense.device, dtype=target_dense.dtype)
        main_t = t.view(batch_size, 1, 1)
        assist_t = t.view(batch_size, 1, 1)
        main_noisy = (1.0 - main_t) * main_noise + main_t * target_main
        assist_noisy = (1.0 - assist_t) * assist_noise + assist_t * target_assist
        main_velocity = target_main - main_noise
        assist_velocity = target_assist - assist_noise

        main_pred = self.main_model(main_noisy, t, global_cond=global_cond)
        assist_pred = self.assist_model(assist_noisy, t, global_cond=global_cond)
        main_loss = F.mse_loss(main_pred, main_velocity)
        assist_keyframe_loss = F.mse_loss(assist_pred, assist_velocity)

        main_x1 = main_noise + main_pred
        assist_x1 = assist_noise + assist_pred
        pred_dense = self._unroll(main_x1, assist_x1)
        dense_recon_loss = F.mse_loss(pred_dense, target_dense)
        demo_recon_loss = F.mse_loss(pred_dense, nactions)
        if self.assist_horizon > 1:
            assist_smooth_loss = (assist_x1[:, 1:] - assist_x1[:, :-1]).pow(2).mean()
        else:
            assist_smooth_loss = assist_x1.sum() * 0.0

        total_loss = (
            main_loss
            + assist_keyframe_loss
            + self.dense_loss_weight * dense_recon_loss
            + self.demo_loss_weight * demo_recon_loss
            + self.assist_smooth_weight * assist_smooth_loss
        )
        self.last_loss_dict = {
            "main_loss": float(main_loss.detach().cpu()),
            "assist_keyframe_loss": float(assist_keyframe_loss.detach().cpu()),
            "dense_recon_loss": float(dense_recon_loss.detach().cpu()),
            "demo_recon_loss": float(demo_recon_loss.detach().cpu()),
            "assist_smooth_loss": float(assist_smooth_loss.detach().cpu()),
            "teacher_enabled": float(teacher_target is not None),
            "assist_horizon": float(self.assist_horizon),
        }
        return total_loss
