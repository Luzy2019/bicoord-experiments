from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from einops import reduce

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy.model.vision.multi_image_obs_encoder import MultiImageObsEncoder
from diffusion_policy.policy.base_image_policy import BaseImagePolicy


def _slice_from_start_dim(start: int, dim: int) -> slice:
    return slice(start, start + dim)


def _motion_update_target(actions: torch.Tensor, left_slice: slice, right_slice: slice, temperature: float) -> torch.Tensor:
    left = actions[:, :, left_slice]
    right = actions[:, :, right_slice]
    left_energy = torch.zeros(actions.shape[0], actions.shape[1], 1, device=actions.device)
    right_energy = torch.zeros_like(left_energy)
    left_energy[:, 1:] = (left[:, 1:] - left[:, :-1]).abs().mean(dim=-1, keepdim=True)
    right_energy[:, 1:] = (right[:, 1:] - right[:, :-1]).abs().mean(dim=-1, keepdim=True)

    energy = torch.cat([left_energy, right_energy], dim=-1)
    baseline = energy[:, 1:].mean(dim=1, keepdim=True).detach().clamp(min=1e-6)
    target = torch.sigmoid(temperature * (energy / baseline - 1.0))
    target[:, 0] = 1.0
    return target.detach()


class AsymGatingNetwork(nn.Module):
    def __init__(self, action_dim: int, cond_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(action_dim * 2 + cond_dim + 1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.expert_head = nn.Linear(hidden_dim, 4)
        self.update_net = nn.Sequential(
            nn.Linear(action_dim * 2 + cond_dim + 1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.update_head = nn.Linear(hidden_dim, 2)

        nn.init.zeros_(self.expert_head.weight)
        nn.init.constant_(self.expert_head.bias, 0.0)
        nn.init.zeros_(self.update_head.weight)
        nn.init.constant_(self.update_head.bias, 1.0)

    def forward(
        self,
        trajectory: torch.Tensor,
        timesteps: torch.Tensor,
        global_cond: Optional[torch.Tensor],
        num_train_timesteps: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=trajectory.device)
        elif timesteps.ndim == 0:
            timesteps = timesteps[None].to(trajectory.device)
        timesteps = timesteps.expand(trajectory.shape[0]).float()
        time_feat = (timesteps / max(num_train_timesteps - 1, 1)).view(-1, 1)

        diff = torch.zeros_like(trajectory)
        diff[:, 1:] = trajectory[:, 1:] - trajectory[:, :-1]
        pooled = torch.cat([trajectory.mean(dim=1), trajectory.std(dim=1, unbiased=False)], dim=-1)
        if global_cond is None:
            global_cond = torch.zeros(trajectory.shape[0], 0, device=trajectory.device, dtype=trajectory.dtype)
        hidden = self.net(torch.cat([pooled, global_cond, time_feat], dim=-1))
        expert_prob = torch.softmax(self.expert_head(hidden), dim=-1)

        step_cond = global_cond[:, None, :].expand(-1, trajectory.shape[1], -1)
        step_time = time_feat[:, None, :].expand(-1, trajectory.shape[1], -1)
        update_hidden = self.update_net(torch.cat([trajectory, diff, step_cond, step_time], dim=-1))
        update_prob = torch.sigmoid(self.update_head(update_hidden))
        first_update = torch.ones(
            update_prob.shape[0],
            1,
            update_prob.shape[2],
            device=update_prob.device,
            dtype=update_prob.dtype,
        )
        update_prob = torch.cat([first_update, update_prob[:, 1:]], dim=1)
        return expert_prob, update_prob


class AsymDiffusionUnetImagePolicy(BaseImagePolicy):
    """
    Cost-aware asymmetric bimanual diffusion policy.

    Expert order:
      0: A  - joint expert
      1: LR - left leader, right follower
      2: RL - right leader, left follower
      3: C  - independent local arms
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
        asym_hidden_dim=256,
        asym_cost_weights=(1.0, 0.5, 0.5, 0.0),
        asym_lambda_cost=1e-3,
        asym_lambda_entropy=1e-4,
        asym_lambda_update_sparse=1e-3,
        asym_lambda_update_kl=1e-2,
        asym_lambda_smooth=1e-3,
        asym_update_target_temperature=8.0,
        asym_hard_inference=True,
        asym_update_threshold=0.5,
        asym_project_action=True,
        **kwargs,
    ):
        super().__init__()
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
        self.left_slice = _slice_from_start_dim(int(left_action_start), int(left_action_dim))
        self.right_slice = _slice_from_start_dim(int(right_action_start), int(right_action_dim))

        input_dim = action_dim + obs_feature_dim
        global_cond_dim = None
        if obs_as_global_cond:
            input_dim = action_dim
            global_cond_dim = obs_feature_dim * n_obs_steps

        def build_unet():
            return ConditionalUnet1D(
                input_dim=input_dim,
                local_cond_dim=None,
                global_cond_dim=global_cond_dim,
                diffusion_step_embed_dim=diffusion_step_embed_dim,
                down_dims=down_dims,
                kernel_size=kernel_size,
                n_groups=n_groups,
                cond_predict_scale=cond_predict_scale,
            )

        self.joint_model = build_unet()
        self.lr_model = build_unet()
        self.rl_model = build_unet()
        self.left_local_model = build_unet()
        self.right_local_model = build_unet()
        self.gating = AsymGatingNetwork(
            action_dim=action_dim,
            cond_dim=global_cond_dim or 0,
            hidden_dim=asym_hidden_dim,
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

        self.asym_lambda_cost = float(asym_lambda_cost)
        self.asym_lambda_entropy = float(asym_lambda_entropy)
        self.asym_lambda_update_sparse = float(asym_lambda_update_sparse)
        self.asym_lambda_update_kl = float(asym_lambda_update_kl)
        self.asym_lambda_smooth = float(asym_lambda_smooth)
        self.asym_update_target_temperature = float(asym_update_target_temperature)
        self.asym_hard_inference = bool(asym_hard_inference)
        self.asym_update_threshold = float(asym_update_threshold)
        self.asym_project_action = bool(asym_project_action)
        self.register_buffer("asym_cost_weights", torch.tensor(asym_cost_weights, dtype=torch.float32))

        self.last_gate_info = None
        self.last_loss_dict = {}
        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

    def _left_only(self, trajectory):
        result = torch.zeros_like(trajectory)
        result[:, :, self.left_slice] = trajectory[:, :, self.left_slice]
        return result

    def _right_only(self, trajectory):
        result = torch.zeros_like(trajectory)
        result[:, :, self.right_slice] = trajectory[:, :, self.right_slice]
        return result

    def _combine_arms(self, left_pred, right_pred):
        result = torch.zeros_like(left_pred)
        result[:, :, self.left_slice] = left_pred[:, :, self.left_slice]
        result[:, :, self.right_slice] = right_pred[:, :, self.right_slice]
        return result

    def _predict_experts(self, trajectory, timesteps, local_cond=None, global_cond=None):
        left_local = self.left_local_model(
            self._left_only(trajectory),
            timesteps,
            local_cond=local_cond,
            global_cond=global_cond,
        )
        right_local = self.right_local_model(
            self._right_only(trajectory),
            timesteps,
            local_cond=local_cond,
            global_cond=global_cond,
        )
        joint = self.joint_model(trajectory, timesteps, local_cond=local_cond, global_cond=global_cond)
        lr = self._combine_arms(
            left_local,
            self.lr_model(trajectory, timesteps, local_cond=local_cond, global_cond=global_cond),
        )
        rl = self._combine_arms(
            self.rl_model(trajectory, timesteps, local_cond=local_cond, global_cond=global_cond),
            right_local,
        )
        independent = self._combine_arms(left_local, right_local)
        return torch.stack([joint, lr, rl, independent], dim=1)

    def _model_forward(self, trajectory, timesteps, local_cond=None, global_cond=None):
        expert_prob, update_prob = self.gating(
            trajectory=trajectory,
            timesteps=timesteps,
            global_cond=global_cond,
            num_train_timesteps=self.noise_scheduler.config.num_train_timesteps,
        )
        experts = self._predict_experts(trajectory, timesteps, local_cond=local_cond, global_cond=global_cond)
        pred = (expert_prob[:, :, None, None] * experts).sum(dim=1)
        self.last_gate_info = {
            "expert_prob": expert_prob,
            "update_prob": update_prob,
        }
        return pred, self.last_gate_info

    def conditional_sample(
        self,
        condition_data,
        condition_mask,
        local_cond=None,
        global_cond=None,
        generator=None,
        **kwargs,
    ):
        scheduler = self.noise_scheduler
        trajectory = torch.randn(
            size=condition_data.shape,
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator,
        )
        scheduler.set_timesteps(self.num_inference_steps)

        for t in scheduler.timesteps:
            trajectory[condition_mask] = condition_data[condition_mask]
            model_output, _ = self._model_forward(
                trajectory,
                t,
                local_cond=local_cond,
                global_cond=global_cond,
            )
            trajectory = scheduler.step(model_output, t, trajectory, generator=generator, **kwargs).prev_sample

        trajectory[condition_mask] = condition_data[condition_mask]
        return trajectory

    def _encode_obs(self, obs_dict):
        nobs = self.normalizer.normalize(obs_dict)
        value = next(iter(nobs.values()))
        batch_size = value.shape[0]
        tobs = self.n_obs_steps
        if self.obs_as_global_cond:
            this_nobs = dict_apply(nobs, lambda x: x[:, :tobs, ...].reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            return None, nobs_features.reshape(batch_size, -1)

        this_nobs = dict_apply(nobs, lambda x: x[:, :tobs, ...].reshape(-1, *x.shape[2:]))
        nobs_features = self.obs_encoder(this_nobs)
        return nobs_features.reshape(batch_size, tobs, -1), None

    @staticmethod
    def _hold_project_arm(actions, update_gate):
        held = [actions[:, 0]]
        for step in range(1, actions.shape[1]):
            gate = update_gate[:, step]
            held.append(gate * actions[:, step] + (1.0 - gate) * held[-1])
        return torch.stack(held, dim=1)

    def _project_action_frequency(self, action_pred):
        if not self.asym_project_action or self.last_gate_info is None:
            return action_pred
        update_prob = self.last_gate_info["update_prob"]
        if self.asym_hard_inference:
            update_gate = (update_prob >= self.asym_update_threshold).to(action_pred.dtype)
            update_gate[:, 0] = 1.0
        else:
            update_gate = update_prob.to(action_pred.dtype)
            update_gate[:, 0] = 1.0

        projected = action_pred.clone()
        left_gate = update_gate[:, :, 0:1]
        right_gate = update_gate[:, :, 1:2]
        projected[:, :, self.left_slice] = self._hold_project_arm(action_pred[:, :, self.left_slice], left_gate)
        projected[:, :, self.right_slice] = self._hold_project_arm(action_pred[:, :, self.right_slice], right_gate)
        return projected

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        assert "past_action" not in obs_dict
        value = next(iter(obs_dict.values()))
        batch_size = value.shape[0]
        horizon = self.horizon
        action_dim = self.action_dim
        tobs = self.n_obs_steps
        device = self.device
        dtype = self.dtype

        local_cond, global_cond = self._encode_obs(obs_dict)
        if self.obs_as_global_cond:
            cond_data = torch.zeros(size=(batch_size, horizon, action_dim), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            cond_data = torch.zeros(
                size=(batch_size, horizon, action_dim + self.obs_feature_dim),
                device=device,
                dtype=dtype,
            )
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:, :tobs, action_dim:] = local_cond
            cond_mask[:, :tobs, action_dim:] = True

        nsample = self.conditional_sample(
            cond_data,
            cond_mask,
            local_cond=None,
            global_cond=global_cond,
            **self.kwargs,
        )
        naction_pred = nsample[..., :action_dim]
        action_pred = self.normalizer["action"].unnormalize(naction_pred)
        action_pred = self._project_action_frequency(action_pred)

        start = tobs - 1
        end = start + self.n_action_steps
        action = action_pred[:, start:end]
        result = {
            "action": action,
            "action_pred": action_pred,
        }
        if self.last_gate_info is not None:
            result["expert_prob"] = self.last_gate_info["expert_prob"]
            result["update_prob"] = self.last_gate_info["update_prob"]
        return result

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def _regularization_loss(self, pred, target, loss_mask, trajectory, gate_info):
        expert_prob = gate_info["expert_prob"]
        update_prob = gate_info["update_prob"]
        losses = {}

        losses["asym_cost"] = (expert_prob * self.asym_cost_weights.to(expert_prob.device)).sum(dim=-1).mean()
        entropy = -(expert_prob.clamp(min=1e-6).log() * expert_prob).sum(dim=-1).mean()
        losses["asym_entropy"] = entropy
        losses["asym_update_sparse"] = update_prob.mean()

        update_target = _motion_update_target(
            trajectory,
            self.left_slice,
            self.right_slice,
            self.asym_update_target_temperature,
        )
        losses["asym_update_kl"] = F.binary_cross_entropy(update_prob, update_target)

        if pred.shape[1] > 1:
            valid = loss_mask[:, 1:] * loss_mask[:, :-1]
            valid_count = valid.sum().clamp(min=1.0)
            smooth = pred[:, 1:] - pred[:, :-1]
            losses["asym_smooth"] = (smooth.pow(2) * valid.type(smooth.dtype)).sum() / valid_count
        else:
            losses["asym_smooth"] = pred.sum() * 0.0

        total = (
            self.asym_lambda_cost * losses["asym_cost"]
            + self.asym_lambda_entropy * losses["asym_entropy"]
            + self.asym_lambda_update_sparse * losses["asym_update_sparse"]
            + self.asym_lambda_update_kl * losses["asym_update_kl"]
            + self.asym_lambda_smooth * losses["asym_smooth"]
        )
        return total, losses

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
        else:
            this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            nobs_features = nobs_features.reshape(batch_size, horizon, -1)
            cond_data = torch.cat([nactions, nobs_features], dim=-1)
            trajectory = cond_data.detach()

        condition_mask = self.mask_generator(trajectory.shape)
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        bsz = trajectory.shape[0]
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (bsz,),
            device=trajectory.device,
        ).long()
        noisy_trajectory = self.noise_scheduler.add_noise(trajectory, noise, timesteps)

        loss_mask = ~condition_mask
        noisy_trajectory[condition_mask] = cond_data[condition_mask]
        pred, gate_info = self._model_forward(
            noisy_trajectory,
            timesteps,
            local_cond=local_cond,
            global_cond=global_cond,
        )

        pred_type = self.noise_scheduler.config.prediction_type
        if pred_type == "epsilon":
            target = noise
        elif pred_type == "sample":
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        diffusion_loss = F.mse_loss(pred, target, reduction="none")
        diffusion_loss = diffusion_loss * loss_mask.type(diffusion_loss.dtype)
        diffusion_loss = reduce(diffusion_loss, "b ... -> b (...)", "mean").mean()
        reg_loss, reg_losses = self._regularization_loss(pred, target, loss_mask, nactions, gate_info)
        total_loss = diffusion_loss + reg_loss

        self.last_loss_dict = {
            "diffusion_loss": float(diffusion_loss.detach().cpu()),
            "asym_cost": float(reg_losses["asym_cost"].detach().cpu()),
            "asym_entropy": float(reg_losses["asym_entropy"].detach().cpu()),
            "asym_update_sparse": float(reg_losses["asym_update_sparse"].detach().cpu()),
            "asym_update_kl": float(reg_losses["asym_update_kl"].detach().cpu()),
            "asym_smooth": float(reg_losses["asym_smooth"].detach().cpu()),
        }
        return total_loss
