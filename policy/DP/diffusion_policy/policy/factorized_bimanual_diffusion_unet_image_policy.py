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
from diffusion_policy.policy.speed_modulation import (
    SpeedModulationHead,
    compute_action_risk,
    compute_branch_mse_risk,
    compute_speed_alpha,
    normalize_risk,
    speed_modulation_loss,
    warp_action_sequence,
)


def _slice_from_start_dim(start: int, dim: int) -> slice:
    return slice(start, start + dim)


class FactorizedBimanualGate(nn.Module):
    """Predicts w(t), u(t) for conditional-vs-marginal bimanual correction."""

    def __init__(self, 
        action_dim: int,
        cond_dim: int,
        hidden_dim: int = 256,
        init_bias: float = -2.0
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
                full_noisy.mean(dim=1),               # 动作在horizon维度（即时间序列）上的均值  [batch, action_dim]
                full_noisy.std(dim=1, unbiased=False) # 动作在horizon维度（即时间序列）上的标准差 [batch, action_dim]
            ],
            dim=-1,
        )
        if global_cond is None:
            global_cond = torch.zeros(
                full_noisy.shape[0], 
                0, 
                device=full_noisy.device, 
                dtype=full_noisy.dtype
            )
        # 将 pooled、global_cond 和 time_feat 拼接起来，然后通过 sigmoid 激活函数得到门控值
        return torch.sigmoid(self.net(torch.cat([pooled, global_cond, time_feat], dim=-1)))


class FactorizedBimanualDiffusionUnetImagePolicy(BaseImagePolicy):
    """
    Factorized bimanual diffusion policy.

    Original DP learns p(a_left, a_right | obs) with one joint denoiser. This
    policy uses one shared denoiser in four conditioning modes:
    - p(a_left | obs)
    - p(a_right | obs)
    - p(a_left | obs, a_right)
    - p(a_right | obs, a_left)

    The final denoising prediction uses dynamic residual gates:
    left = left_marginal + w(t) * (left_cond - left_marginal)
    right = right_marginal + u(t) * (right_cond - right_marginal)
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
        factorized_hidden_dim=256,
        factorized_gate_init_bias=-2.0,
        factorized_aux_loss_weight=0.25,
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
        speed_modulation_coupling_risk_weight=1.0,
        speed_modulation_geometry_risk_weight=0.25,
        **kwargs,
    ):
        super().__init__()
        if not obs_as_global_cond:
            raise ValueError("FactorizedBimanualDiffusionUnetImagePolicy requires obs_as_global_cond=True")

        action_shape = shape_meta["action"]["shape"]
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        obs_feature_dim = obs_encoder.output_shape()[0]

        # left_action_dim: 7
        # right_action_dim: 7
        if left_action_dim is None:
            left_action_dim = action_dim // 2
        if right_action_dim is None:
            right_action_dim = action_dim - left_action_dim

        # left_action_start: 0
        # right_action_start: 7
        if right_action_start is None:
            right_action_start = left_action_start + left_action_dim

        self.left_action_dim = int(left_action_dim)
        self.right_action_dim = int(right_action_dim)
        self.left_slice = _slice_from_start_dim(int(left_action_start), self.left_action_dim)
        self.right_slice = _slice_from_start_dim(int(right_action_start), self.right_action_dim)

        global_cond_dim = obs_feature_dim * n_obs_steps

        if self.left_action_dim != self.right_action_dim:
            raise ValueError("Shared factorized model requires equal left/right action dimensions")
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
        
        # w(t), u(t) for conditional-vs-marginal bimanual correction.
        self.factorized_gate = FactorizedBimanualGate(
            action_dim=action_dim,
            cond_dim=global_cond_dim,
            hidden_dim=factorized_hidden_dim,
            init_bias=factorized_gate_init_bias,
        )

        # 观测编码器，用于提取观测特征
        self.obs_encoder = obs_encoder
        # 添加噪声的调度器
        self.noise_scheduler = noise_scheduler
        # 输入数据归一化器
        self.normalizer = LinearNormalizer()
        # 序列长度
        self.horizon = horizon
        # 经过编码后的观测特征维度
        self.obs_feature_dim = obs_feature_dim
        # 动作维度
        self.action_dim = action_dim
        # 每个采样包含的动作步数
        self.n_action_steps = n_action_steps
        # 每个采样包含的观测步数
        self.n_obs_steps = n_obs_steps
        # 是否将观测作为全局条件
        self.obs_as_global_cond = obs_as_global_cond
        # 其他额外参数
        self.kwargs = kwargs
        # 辅助损失函数的权重
        self.factorized_aux_loss_weight = float(factorized_aux_loss_weight)
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
        # 保存最近一次门控信息，用于调试
        self.last_gate_info = None
        # 保存最近一次损失字典
        self.last_loss_dict = {}
        # 保存最近一次调试信息
        self.last_debug_info = None
        self.last_left_denoise_output = None
        self.last_right_denoise_output = None
        self.last_left_speed_alpha = None
        self.last_right_speed_alpha = None
        self.last_action_pred_raw = None

        # 如果未指定推理步数，则使用训练步数
        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        # 推理步数
        self.num_inference_steps = num_inference_steps

    def _encode_obs(self, obs_dict):
        # 首先对输入的 obs_dict（观测字典）做归一化处理
        nobs = self.normalizer.normalize(obs_dict)
        # 取归一化字典中的第一个 value，用于获取 batch 大小
        value = next(iter(nobs.values()))
        batch_size = value.shape[0]
        # 对每个观测，只截取前 n_obs_steps 步，并将其展平成 (batch_size * n_obs_steps, ...) 的格式
        this_nobs = dict_apply(nobs, lambda x: x[:, :self.n_obs_steps, ...].reshape(-1, *x.shape[2:]))
        # 输入编码器，获得观测特征 (flattened across batch)
        nobs_features = self.obs_encoder(this_nobs)
        # 把特征 reshape 返回成 (batch_size, -1)，每个 batch 对应一个 feature 向量
        return nobs_features.reshape(batch_size, -1)

    def _combine_full(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        full = torch.zeros(
            # 这里是 batch size，即每个批次中的样本数量
            left.shape[0],
            self.horizon,
            self.action_dim,
            device=left.device,
            dtype=left.dtype
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
            num_train_timesteps=self.noise_scheduler.config.num_train_timesteps,
        )

        left_context = self._context(left_noisy)
        right_context = self._context(right_noisy)
        zero_left_context = torch.zeros_like(left_context)
        zero_right_context = torch.zeros_like(right_context)

        # P(a_left | obs)
        left_marginal = self.factorized_model(
            left_noisy,
            timesteps,
            global_cond=self._branch_cond(global_cond, zero_right_context, arm_id=0.0, cond_mask=0.0),
        )
        # P(a_right | obs)
        right_marginal = self.factorized_model(
            right_noisy,
            timesteps,
            global_cond=self._branch_cond(global_cond, zero_left_context, arm_id=1.0, cond_mask=0.0),
        )

        # P(a_left | obs, a_right)
        left_cond = self.factorized_model(
            left_noisy,
            timesteps,
            global_cond=self._branch_cond(global_cond, right_context, arm_id=0.0, cond_mask=1.0),
        )
        # P(a_right | obs, a_left)
        right_cond = self.factorized_model(
            right_noisy,
            timesteps,
            global_cond=self._branch_cond(global_cond, left_context, arm_id=1.0, cond_mask=1.0),
        )

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
        if self.speed_modulation_train_only:
            global_cond = global_cond.detach()

        left_target = nactions[:, :, self.left_slice]
        right_target = nactions[:, :, self.right_slice]
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
        if self.speed_modulation_train_only:
            with torch.no_grad():
                left_pred, right_pred, gate_info = self._predict_factorized(left_noisy, right_noisy, timesteps, global_cond)
        else:
            left_pred, right_pred, gate_info = self._predict_factorized(left_noisy, right_noisy, timesteps, global_cond)

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

        aux_loss_dict = {
            "left_marginal_aux_loss": F.mse_loss(gate_info["left_marginal"], left_label),
            "right_marginal_aux_loss": F.mse_loss(gate_info["right_marginal"], right_label),
            "left_cond_aux_loss": F.mse_loss(gate_info["left_cond"], left_label),
            "right_cond_aux_loss": F.mse_loss(gate_info["right_cond"], right_label),
        }
        aux_loss = sum(aux_loss_dict.values()) / len(aux_loss_dict)
        if self.speed_modulation_train_only:
            total_loss = diffusion_loss * 0.0
        else:
            total_loss = diffusion_loss + self.factorized_aux_loss_weight * aux_loss

        speed_loss = left_target.new_zeros(())
        speed_loss_dict = {}
        if self.speed_modulation_enabled and self.speed_modulation_learned and self.speed_modulation_loss_weight > 0:
            left_speed_signal = left_pred.detach() if self.speed_modulation_detach_signal else left_pred
            right_speed_signal = right_pred.detach() if self.speed_modulation_detach_signal else right_pred
            left_alpha = self.left_speed_head(
                left_speed_signal,
                timesteps,
                global_cond=global_cond,
                num_train_timesteps=self.noise_scheduler.config.num_train_timesteps,
            )
            right_alpha = self.right_speed_head(
                right_speed_signal,
                timesteps,
                global_cond=global_cond,
                num_train_timesteps=self.noise_scheduler.config.num_train_timesteps,
            )
            left_speed_loss, left_speed_dict = speed_modulation_loss(
                alpha=left_alpha,
                actions=left_target,
                alpha_min=self.speed_modulation_min,
                alpha_max=self.speed_modulation_max,
                target_weight=self.speed_modulation_target_weight,
                smooth_weight=self.speed_modulation_smooth_weight,
                fast_weight=self.speed_modulation_fast_weight,
                risk_weight=self.speed_modulation_risk_weight,
                risk=normalize_risk(
                    self.speed_modulation_coupling_risk_weight
                    * compute_branch_mse_risk(gate_info["left_cond"], gate_info["left_marginal"])
                    + self.speed_modulation_geometry_risk_weight * compute_action_risk(left_target)
                ),
            )
            right_speed_loss, right_speed_dict = speed_modulation_loss(
                alpha=right_alpha,
                actions=right_target,
                alpha_min=self.speed_modulation_min,
                alpha_max=self.speed_modulation_max,
                target_weight=self.speed_modulation_target_weight,
                smooth_weight=self.speed_modulation_smooth_weight,
                fast_weight=self.speed_modulation_fast_weight,
                risk_weight=self.speed_modulation_risk_weight,
                risk=normalize_risk(
                    self.speed_modulation_coupling_risk_weight
                    * compute_branch_mse_risk(gate_info["right_cond"], gate_info["right_marginal"])
                    + self.speed_modulation_geometry_risk_weight * compute_action_risk(right_target)
                ),
            )
            speed_loss = 0.5 * (left_speed_loss + right_speed_loss)
            total_loss = total_loss + self.speed_modulation_loss_weight * speed_loss

            # log
            left_coupling_risk = compute_branch_mse_risk(gate_info["left_cond"], gate_info["left_marginal"])
            right_coupling_risk = compute_branch_mse_risk(gate_info["right_cond"], gate_info["right_marginal"])
            speed_loss_dict = {
                "speed_modulation_loss": speed_loss,
                "left_coupling_mse_risk": left_coupling_risk.mean(),
                "right_coupling_mse_risk": right_coupling_risk.mean(),
                **{f"left_{key}": value for key, value in left_speed_dict.items()},
                **{f"right_{key}": value for key, value in right_speed_dict.items()},
            }

        gates = gate_info["factorized_gates"]
        self.last_loss_dict = {
            "diffusion_loss": float(diffusion_loss.detach().cpu()),
            "left_diffusion_loss": float(left_loss.detach().cpu()),
            "right_diffusion_loss": float(right_loss.detach().cpu()),
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

        scheduler = self.noise_scheduler
        scheduler.set_timesteps(self.num_inference_steps)
        debug_steps = []
        for t in scheduler.timesteps:
            left_pred, right_pred, gate_info = self._predict_factorized(left, right, t, global_cond)
            if collect_debug:
                debug_steps.append({
                    "timestep": int(t.detach().cpu()) if torch.is_tensor(t) else int(t),
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
            self.last_left_denoise_output = left_pred.detach()
            self.last_right_denoise_output = right_pred.detach()
            left = scheduler.step(left_pred, t, left, generator=generator, **self.kwargs).prev_sample
            right = scheduler.step(right_pred, t, right, generator=generator, **self.kwargs).prev_sample
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
        self.last_action_pred_raw = action_pred.detach()
        if (
            self.speed_modulation_enabled
            and self.last_left_denoise_output is not None
            and self.last_right_denoise_output is not None
        ):
            if self.speed_modulation_learned:
                zero_t = torch.zeros(batch_size, dtype=torch.long, device=global_cond.device)
                left_alpha = self.left_speed_head(
                    self.last_left_denoise_output,
                    zero_t,
                    global_cond=global_cond,
                    num_train_timesteps=self.noise_scheduler.config.num_train_timesteps,
                )
                right_alpha = self.right_speed_head(
                    self.last_right_denoise_output,
                    zero_t,
                    global_cond=global_cond,
                    num_train_timesteps=self.noise_scheduler.config.num_train_timesteps,
                )
            else:
                left_alpha = compute_speed_alpha(
                    self.last_left_denoise_output,
                    strength=self.speed_modulation_strength,
                    alpha_min=self.speed_modulation_min,
                    alpha_max=self.speed_modulation_max,
                    smooth_kernel=self.speed_modulation_smooth,
                )
                right_alpha = compute_speed_alpha(
                    self.last_right_denoise_output,
                    strength=self.speed_modulation_strength,
                    alpha_min=self.speed_modulation_min,
                    alpha_max=self.speed_modulation_max,
                    smooth_kernel=self.speed_modulation_smooth,
                )
            left_action = warp_action_sequence(action_pred[:, :, self.left_slice], left_alpha)
            right_action = warp_action_sequence(action_pred[:, :, self.right_slice], right_alpha)
            action_pred = self._combine_full(left_action, right_action)
            self.last_left_speed_alpha = left_alpha.detach()
            self.last_right_speed_alpha = right_alpha.detach()

        start = self.n_obs_steps - 1
        end = start + self.n_action_steps
        result = {
            "action": action_pred[:, start:end],
            "action_pred": action_pred,
        }
        if self.last_gate_info is not None:
            result["factorized_gates"] = self.last_gate_info["factorized_gates"]
        if self.last_left_speed_alpha is not None and self.last_right_speed_alpha is not None:
            result["left_speed_alpha"] = self.last_left_speed_alpha
            result["right_speed_alpha"] = self.last_right_speed_alpha
            result["action_pred_raw"] = self.last_action_pred_raw
        return result
