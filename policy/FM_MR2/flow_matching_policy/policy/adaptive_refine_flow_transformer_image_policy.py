import copy
from typing import Dict, Optional

import torch
import torch.nn.functional as F

try:
    import dill
except ImportError:  # pragma: no cover - used on minimal smoke-test envs
    import pickle as dill

from flow_matching_policy.common.pytorch_util import dict_apply
from flow_matching_policy.model.adaptive_refine_action_expert import (
    AdaptiveRefineActionExpert,
    TemporalLossWeights,
)
from flow_matching_policy.model.common.normalizer import LinearNormalizer
from flow_matching_policy.model.vision.multi_image_obs_encoder import MultiImageObsEncoder
from flow_matching_policy.policy.base_image_policy import BaseImagePolicy
from flow_matching_policy.policy.diffusion_unet_image_policy import DiffusionUnetImagePolicy


class RefineTargetBuilder:
    """Builds per-arm round targets without requiring phase labels in v1."""

    def __init__(self, max_refine_rounds: int, temporal_alpha: float = 0.25, mode: str = "action_complexity"):
        self.max_refine_rounds = int(max_refine_rounds)
        self.temporal_alpha = float(temporal_alpha)
        self.mode = str(mode)

    def _front_weights(self, length: int, device, dtype):
        idx = torch.arange(length, device=device, dtype=dtype)
        weights = torch.exp(-self.temporal_alpha * idx)
        return weights / weights.mean().clamp_min(1.0e-6)

    def action_complexity(self, target_dense: torch.Tensor):
        if target_dense.shape[1] < 2:
            score = target_dense.pow(2).mean(dim=(1, 2))
        else:
            velocity = target_dense[:, 1:] - target_dense[:, :-1]
            v_weights = self._front_weights(velocity.shape[1], velocity.device, velocity.dtype)
            score = (velocity.pow(2).mean(dim=-1) * v_weights[None, :]).mean(dim=1)
            if velocity.shape[1] > 1:
                accel = velocity[:, 1:] - velocity[:, :-1]
                a_weights = self._front_weights(accel.shape[1], accel.device, accel.dtype)
                score = score + (accel.pow(2).mean(dim=-1) * a_weights[None, :]).mean(dim=1)
        scale = score.detach().mean().clamp_min(1.0e-6)
        refine_target = torch.sigmoid((score / scale - 1.0) * 3.0).detach()
        round_target = torch.round(refine_target * (self.max_refine_rounds - 1)).long()
        round_target = round_target.clamp(0, self.max_refine_rounds - 1)
        budget_target = torch.clamp(torch.floor(refine_target * 3.0).long(), 0, 2)
        cache_target = (1.0 - refine_target).detach()
        return refine_target, round_target, budget_target, cache_target

    def __call__(self, target_dense: torch.Tensor):
        if self.mode not in ("action_complexity", "phase_label", "hybrid"):
            raise ValueError(f"Unsupported refine target mode: {self.mode}")
        # phase_label/hybrid are reserved for future labeled datasets; current zarrs do not contain labels.
        return self.action_complexity(target_dense)


class AdaptiveRefineFlowTransformerImagePolicy(BaseImagePolicy):
    """Flow-matching policy with adaptive coarse-to-fine action refinement."""

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
        diffusion_step_embed_dim=128,
        down_dims=(256, 512, 1024),
        kernel_size=5,
        n_groups=8,
        cond_predict_scale=True,
        left_action_dim=None,
        right_action_dim=None,
        left_action_start=0,
        right_action_start=None,
        expert_horizon=None,
        coarse_plan_steps=10,
        max_refine_rounds=3,
        min_refine_rounds=1,
        temporal_weight_alpha=0.25,
        front_refine_alpha=0.25,
        transformer_hidden_dim=256,
        transformer_n_layer=6,
        transformer_n_head=8,
        transformer_ff_mult=4,
        transformer_dropout=0.0,
        num_moe_experts=4,
        enable_cache=True,
        enable_early_exit=True,
        enable_layer_skip=True,
        enable_moe=True,
        enable_pruning=True,
        enable_sparse_attention=True,
        early_exit_threshold=0.92,
        sparse_attention_band=4,
        full_compute_warmup_steps=1000,
        dynamic_round_train_prob=0.5,
        refine_target=None,
        teacher=None,
        loss=None,
        **kwargs,
    ):
        super().__init__()
        if not obs_as_global_cond:
            raise ValueError("AdaptiveRefineFlowTransformerImagePolicy requires obs_as_global_cond=True")

        action_shape = shape_meta["action"]["shape"]
        assert len(action_shape) == 1
        action_dim = int(action_shape[0])
        obs_feature_dim = int(obs_encoder.output_shape()[0])
        global_cond_dim = obs_feature_dim * int(n_obs_steps)

        self.model = AdaptiveRefineActionExpert(
            action_dim=action_dim,
            horizon=int(horizon),
            cond_dim=global_cond_dim,
            left_action_dim=left_action_dim,
            right_action_dim=right_action_dim,
            left_action_start=left_action_start,
            right_action_start=right_action_start,
            expert_horizon=expert_horizon,
            coarse_plan_steps=coarse_plan_steps,
            max_refine_rounds=max_refine_rounds,
            min_refine_rounds=min_refine_rounds,
            hidden_dim=transformer_hidden_dim,
            n_layer=transformer_n_layer,
            n_head=transformer_n_head,
            ff_mult=transformer_ff_mult,
            dropout=transformer_dropout,
            num_moe_experts=num_moe_experts,
            enable_cache=enable_cache,
            enable_early_exit=enable_early_exit,
            enable_layer_skip=enable_layer_skip,
            enable_moe=enable_moe,
            enable_pruning=enable_pruning,
            enable_sparse_attention=enable_sparse_attention,
            early_exit_threshold=early_exit_threshold,
            sparse_attention_band=sparse_attention_band,
            front_refine_alpha=front_refine_alpha,
        )
        self.obs_encoder = obs_encoder
        self.noise_scheduler = noise_scheduler
        self.normalizer = LinearNormalizer()

        self.horizon = int(horizon)
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = int(n_action_steps)
        self.n_obs_steps = int(n_obs_steps)
        self.obs_as_global_cond = bool(obs_as_global_cond)
        self.expert_horizon = int(self.model.expert_horizon)
        self.coarse_plan_steps = self.expert_horizon
        self.max_refine_rounds = int(max_refine_rounds)
        self.arm_specs = self.model.arm_specs
        self.kwargs = kwargs
        self.full_compute_warmup_steps = int(full_compute_warmup_steps)
        self.dynamic_round_train_prob = float(dynamic_round_train_prob)
        self._train_step = 0
        self.teacher_policy_kwargs = {
            "diffusion_step_embed_dim": diffusion_step_embed_dim,
            "down_dims": down_dims,
            "kernel_size": kernel_size,
            "n_groups": n_groups,
            "cond_predict_scale": cond_predict_scale,
        }

        if num_inference_steps is None:
            num_inference_steps = getattr(getattr(noise_scheduler, "config", None), "num_train_timesteps", 100)
        self.num_inference_steps = int(num_inference_steps)

        loss = loss or {}
        teacher = teacher or {}
        self.flow_or_diffusion_loss_weight = float(loss.get("flow_or_diffusion_loss_weight", 1.0))
        self.dense_loss_weight = float(loss.get("dense_loss_weight", teacher.get("dense_loss_weight", 1.0)))
        default_demo_weight = 0.5 if bool(teacher.get("enabled", False)) else 0.0
        self.demo_loss_weight = float(loss.get("demo_loss_weight", teacher.get("demo_loss_weight", default_demo_weight)))
        self.temporal_front_loss_weight = float(loss.get("temporal_front_loss_weight", 0.5))
        self.refine_improvement_weight = float(loss.get("refine_improvement_weight", 0.1))
        self.gate_budget_weight = float(loss.get("gate_budget_weight", 0.05))
        self.cache_consistency_weight = float(loss.get("cache_consistency_weight", 0.02))
        self.moe_balance_weight = float(loss.get("moe_balance_weight", 0.01))
        self.temporal_weights = TemporalLossWeights(self.horizon, temporal_weight_alpha)
        refine_target = refine_target or {}
        self.refine_target_builder = RefineTargetBuilder(
            max_refine_rounds=self.max_refine_rounds,
            temporal_alpha=float(refine_target.get("temporal_alpha", temporal_weight_alpha)),
            mode=refine_target.get("mode", "action_complexity"),
        )

        self.teacher_enabled = bool(teacher.get("enabled", False))
        self.teacher_ckpt_path = teacher.get("ckpt_path", None)
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
                )
            )

        self.last_action_pred_raw = None
        self.last_refine_aux = {}
        self.last_loss_dict = {}

    def _load_teacher(self, **ctor_kwargs):
        teacher_obs_encoder = copy.deepcopy(ctor_kwargs.pop("obs_encoder"))
        policy = DiffusionUnetImagePolicy(
            obs_encoder=teacher_obs_encoder,
            **ctor_kwargs,
            **self.teacher_policy_kwargs,
        )
        with open(self.teacher_ckpt_path, "rb") as f:
            payload = torch.load(f, pickle_module=dill, map_location="cpu")
        state = payload["state_dicts"].get("ema_model", payload["state_dicts"].get("model"))
        policy.load_state_dict(state, strict=False)
        policy.eval()
        policy.requires_grad_(False)
        return policy

    def reset(self):
        self.model.reset_cache()

    def _encode_global_cond(self, obs_dict):
        nobs = self.normalizer.normalize(obs_dict)
        value = next(iter(nobs.values()))
        batch_size = value.shape[0]
        this_nobs = dict_apply(nobs, lambda x: x[:, : self.n_obs_steps, ...].reshape(-1, *x.shape[2:]))
        nobs_features = self.obs_encoder(this_nobs)
        return nobs_features.reshape(batch_size, -1)

    def _resize_action(self, actions: torch.Tensor, horizon: int) -> torch.Tensor:
        if actions.shape[1] == horizon:
            return actions
        if horizon == 1:
            return actions.mean(dim=1, keepdim=True)
        return F.interpolate(
            actions.transpose(1, 2),
            size=horizon,
            mode="linear",
            align_corners=True,
        ).transpose(1, 2)

    def conditional_sample(self, shape, global_cond=None, generator=None):
        trajectory = torch.randn(shape, dtype=self.dtype, device=self.device, generator=generator)
        num_steps = max(int(self.num_inference_steps), 1)
        dt = 1.0 / num_steps
        batch_size = trajectory.shape[0]
        aux = {}
        gate_sample = torch.zeros_like(trajectory)
        gate_t = torch.zeros((batch_size,), dtype=trajectory.dtype, device=trajectory.device)
        gate_state = self.model.compute_gate_state(
            gate_sample,
            gate_t,
            global_cond=global_cond,
            use_cache=True,
        )
        for step_idx in range(num_steps):
            t = torch.full(
                (batch_size,),
                step_idx / num_steps,
                dtype=trajectory.dtype,
                device=trajectory.device,
            )
            model_output, aux = self.model(
                trajectory,
                t,
                global_cond=global_cond,
                use_cache=True,
                force_full_compute=False,
                gate_state=gate_state,
            )
            trajectory = trajectory + dt * model_output
        if aux:
            self.model.update_cache(plan=trajectory, hidden=None)
        self.last_refine_aux = aux
        return trajectory

    def _aux_scalar(self, aux: Dict[str, torch.Tensor], key: str, device) -> torch.Tensor:
        value = aux.get(key, None)
        if value is None:
            return torch.tensor(0.0, device=device)
        if torch.is_tensor(value):
            return value.detach().mean()
        return torch.tensor(float(value), device=device)

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        assert "past_action" not in obs_dict
        global_cond = self._encode_global_cond(obs_dict)
        batch_size = global_cond.shape[0]
        naction_pred = self.conditional_sample(
            (batch_size, self.horizon, self.action_dim),
            global_cond=global_cond,
        )
        action_pred = self.normalizer["action"].unnormalize(naction_pred)
        self.last_action_pred_raw = action_pred.detach()

        start = self.n_obs_steps - 1
        end = start + self.n_action_steps
        aux = self.last_refine_aux
        device = action_pred.device
        return {
            "action": action_pred[:, start:end],
            "action_pred": action_pred,
            "multi_rate_action_pred": action_pred,
            "adaptive_refine_action_pred": action_pred,
            "coarse_plan_steps": torch.tensor(self.coarse_plan_steps, device=device),
            "expert_horizon": self._aux_scalar(aux, "expert_horizon", device),
            "refine_rounds_used": self._aux_scalar(aux, "refine_rounds_used", device),
            "early_exit_rate": self._aux_scalar(aux, "early_exit_rate", device),
            "cache_reuse_rate": self._aux_scalar(aux, "cache_reuse_rate", device),
            "gate_reuse_rate": self._aux_scalar(aux, "gate_reuse_rate", device),
            "active_expert_tokens": self._aux_scalar(aux, "active_expert_tokens", device),
            "moe_entropy": self._aux_scalar(aux, "moe_entropy", device),
            "compute_budget_pred": self._aux_scalar(aux, "compute_budget_pred", device),
            "left_rounds_used": self._aux_scalar(aux, "left_refine_rounds_used", device),
            "right_rounds_used": self._aux_scalar(aux, "right_refine_rounds_used", device),
            "left_refine_rounds_used": self._aux_scalar(aux, "left_refine_rounds_used", device),
            "right_refine_rounds_used": self._aux_scalar(aux, "right_refine_rounds_used", device),
            "left_action_steps_used": self._aux_scalar(aux, "left_action_steps_used", device),
            "right_action_steps_used": self._aux_scalar(aux, "right_action_steps_used", device),
            "left_cache_reuse_rate": self._aux_scalar(aux, "left_cache_reuse_rate", device),
            "right_cache_reuse_rate": self._aux_scalar(aux, "right_cache_reuse_rate", device),
            "left_compute_budget_pred": self._aux_scalar(aux, "left_compute_budget_pred", device),
            "right_compute_budget_pred": self._aux_scalar(aux, "right_compute_budget_pred", device),
        }

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
            teacher_action = self.normalizer["action"].normalize(teacher_out["action_pred"])
            return self._resize_action(teacher_action, self.horizon)

    def _weights(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.temporal_weights(device=tensor.device, dtype=tensor.dtype)

    def _weighted_mse(self, pred, target, weights: Optional[torch.Tensor] = None, reduce_batch=True):
        loss = (pred - target).pow(2)
        if weights is not None:
            loss = loss * weights
        if reduce_batch:
            return loss.mean()
        return loss.mean(dim=(1, 2))

    def _front_slice(self):
        start = min(max(self.n_obs_steps - 1, 0), self.horizon - 1)
        end = min(self.horizon, start + self.n_action_steps)
        return slice(start, end)

    def _complexity_targets(self, target_dense: torch.Tensor):
        return self.refine_target_builder(target_dense)

    def _round_targets_for_arms(self, target_dense: torch.Tensor):
        round_targets = []
        for _, spec in self.arm_specs.items():
            target_arm = target_dense[:, :, spec["slice"]]
            _, round_target, _, _ = self._complexity_targets(target_arm)
            round_targets.append(round_target + 1)
        return torch.stack(round_targets, dim=1)

    def _use_dynamic_round_training(self):
        if not self.training:
            return True
        if self._train_step < self.full_compute_warmup_steps:
            return False
        if self.dynamic_round_train_prob >= 1.0:
            return True
        if self.dynamic_round_train_prob <= 0.0:
            return False
        return bool(torch.rand(()) < self.dynamic_round_train_prob)

    def _gate_budget_loss(self, aux, target_dense):
        round_ids = torch.arange(1, self.max_refine_rounds + 1, device=target_dense.device, dtype=target_dense.dtype)
        losses = []
        for arm, spec in self.arm_specs.items():
            target_arm = target_dense[:, :, spec["slice"]]
            refine_target, round_target, budget_target, cache_target = self._complexity_targets(target_arm)
            need_loss = F.binary_cross_entropy_with_logits(aux[f"{arm}_need_refine_logit"], refine_target)
            round_loss = F.cross_entropy(aux[f"{arm}_round_logits"], round_target)
            budget_loss = F.cross_entropy(aux[f"{arm}_budget_logits"], budget_target)
            cache_loss = F.binary_cross_entropy_with_logits(aux[f"{arm}_cache_logit"], cache_target)
            expected_rounds = (aux[f"{arm}_round_probs"] * round_ids[None, :]).sum(dim=-1)
            if self.max_refine_rounds > 1:
                expected_rounds = (expected_rounds - 1.0) / float(self.max_refine_rounds - 1)
            expected_loss = F.mse_loss(expected_rounds, refine_target)
            losses.append(need_loss + round_loss + budget_loss + cache_loss + expected_loss)
        return torch.stack(losses).mean()

    def _refine_improvement_loss(self, round_outputs, noise, target_dense, weights):
        if len(round_outputs) < 2:
            return target_dense.sum() * 0.0
        round_losses = [
            self._weighted_mse(noise + pred, target_dense, weights=weights, reduce_batch=False)
            for pred in round_outputs
        ]
        penalty = target_dense.sum() * 0.0
        for prev_loss, next_loss in zip(round_losses[:-1], round_losses[1:]):
            penalty = penalty + F.relu(next_loss - prev_loss).mean()
        return penalty / float(len(round_losses) - 1)

    def _cache_consistency_loss(self, aux):
        round_outputs = aux.get("round_outputs", [])
        if len(round_outputs) < 2:
            return aux["left_cache_prob"].sum() * 0.0
        losses = []
        for arm, spec in self.arm_specs.items():
            cache_prob = aux[f"{arm}_cache_prob"].view(-1, 1, 1)
            diff = round_outputs[-1][:, :, spec["slice"]] - round_outputs[0][:, :, spec["slice"]]
            losses.append((cache_prob * diff.pow(2)).mean())
        return torch.stack(losses).mean()

    def compute_loss(self, batch):
        nobs = self.normalizer.normalize(batch["obs"])
        nactions = self.normalizer["action"].normalize(batch["action"])
        batch_size = nactions.shape[0]
        this_nobs = dict_apply(nobs, lambda x: x[:, : self.n_obs_steps, ...].reshape(-1, *x.shape[2:]))
        nobs_features = self.obs_encoder(this_nobs)
        global_cond = nobs_features.reshape(batch_size, -1)

        teacher_target = self._teacher_target(batch)
        target_dense = teacher_target if teacher_target is not None else nactions
        target_dense = self._resize_action(target_dense, self.horizon)
        demo_dense = self._resize_action(nactions, self.horizon)

        noise = torch.randn_like(target_dense)
        t = torch.rand((batch_size,), device=target_dense.device, dtype=target_dense.dtype)
        t_view = t.view(batch_size, 1, 1)
        noisy = (1.0 - t_view) * noise + t_view * target_dense
        velocity = target_dense - noise

        round_targets = self._round_targets_for_arms(target_dense)
        use_dynamic_rounds = self._use_dynamic_round_training()
        gate_sample = torch.zeros_like(noisy)
        gate_t = torch.zeros((batch_size,), device=target_dense.device, dtype=target_dense.dtype)
        gate_state = self.model.compute_gate_state(
            gate_sample,
            gate_t,
            global_cond=global_cond,
            use_cache=False,
            forced_rounds=round_targets if use_dynamic_rounds else None,
        )
        pred, aux = self.model(
            noisy,
            t,
            global_cond=global_cond,
            use_cache=False,
            force_full_compute=not use_dynamic_rounds,
            gate_state=gate_state,
        )
        pred_dense = noise + pred
        weights = self._weights(target_dense)
        front = self._front_slice()

        flow_or_diffusion_loss = F.mse_loss(pred, velocity)
        dense_recon_loss = F.mse_loss(pred_dense, target_dense)
        demo_recon_loss = F.mse_loss(pred_dense, demo_dense)
        temporal_front_loss = self._weighted_mse(pred_dense, target_dense, weights=weights)
        front_action_mse = F.mse_loss(pred_dense[:, front], target_dense[:, front])
        refine_improvement_loss = self._refine_improvement_loss(
            aux.get("round_outputs", []),
            noise,
            target_dense,
            weights,
        )
        gate_budget_loss = self._gate_budget_loss(aux, target_dense)
        cache_consistency_loss = self._cache_consistency_loss(aux)
        moe_balance_loss = aux["moe_balance_loss"]

        total_loss = (
            self.flow_or_diffusion_loss_weight * flow_or_diffusion_loss
            + self.dense_loss_weight * dense_recon_loss
            + self.demo_loss_weight * demo_recon_loss
            + self.temporal_front_loss_weight * temporal_front_loss
            + self.refine_improvement_weight * refine_improvement_loss
            + self.gate_budget_weight * gate_budget_loss
            + self.cache_consistency_weight * cache_consistency_loss
            + self.moe_balance_weight * moe_balance_loss
        )
        self._train_step += 1
        self.last_loss_dict = {
            "flow_or_diffusion_loss": float(flow_or_diffusion_loss.detach().cpu()),
            "dense_recon_loss": float(dense_recon_loss.detach().cpu()),
            "demo_recon_loss": float(demo_recon_loss.detach().cpu()),
            "temporal_front_loss": float(temporal_front_loss.detach().cpu()),
            "front_action_mse": float(front_action_mse.detach().cpu()),
            "refine_improvement_loss": float(refine_improvement_loss.detach().cpu()),
            "gate_budget_loss": float(gate_budget_loss.detach().cpu()),
            "cache_consistency_loss": float(cache_consistency_loss.detach().cpu()),
            "moe_balance_loss": float(moe_balance_loss.detach().cpu()),
            "teacher_enabled": float(teacher_target is not None),
            "dynamic_round_training": float(use_dynamic_rounds),
            "expert_horizon": float(aux["expert_horizon"].detach().mean().cpu()),
            "active_expert_tokens": float(aux["active_expert_tokens"].detach().mean().cpu()),
            "gate_reuse_rate": float(aux["gate_reuse_rate"].detach().mean().cpu()),
            "refine_rounds_used": float(aux["refine_rounds_used"].detach().mean().cpu()),
            "early_exit_rate": float(aux["early_exit_rate"].detach().mean().cpu()),
            "cache_reuse_rate": float(aux["cache_reuse_rate"].detach().mean().cpu()),
            "moe_entropy": float(aux["moe_entropy"].detach().mean().cpu()),
            "compute_budget_pred": float(aux["compute_budget_pred"].detach().mean().cpu()),
            "left_rounds_used": float(aux["left_refine_rounds_used"].detach().mean().cpu()),
            "right_rounds_used": float(aux["right_refine_rounds_used"].detach().mean().cpu()),
            "left_refine_rounds_used": float(aux["left_refine_rounds_used"].detach().mean().cpu()),
            "right_refine_rounds_used": float(aux["right_refine_rounds_used"].detach().mean().cpu()),
            "left_action_steps_used": float(aux["left_action_steps_used"].detach().mean().cpu()),
            "right_action_steps_used": float(aux["right_action_steps_used"].detach().mean().cpu()),
            "left_cache_reuse_rate": float(aux["left_cache_reuse_rate"].detach().mean().cpu()),
            "right_cache_reuse_rate": float(aux["right_cache_reuse_rate"].detach().mean().cpu()),
            "left_compute_budget_pred": float(aux["left_compute_budget_pred"].detach().mean().cpu()),
            "right_compute_budget_pred": float(aux["right_compute_budget_pred"].detach().mean().cpu()),
        }
        return total_loss
