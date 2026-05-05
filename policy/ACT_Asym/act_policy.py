import torch.nn as nn
import os
import torch
import numpy as np
import pickle
from torch.nn import functional as F
import torchvision.transforms as transforms

try:
    from detr.main import (
        build_ACT_model_and_optimizer,
        build_CNNMLP_model_and_optimizer,
    )
    from asym_bimanual_adapter import (
        AsymFreqResidualAdapter,
        LearnedAsymFreqResidualAdapter,
        bernoulli_kl,
        motion_energy_update_target,
        motion_role_target,
    )
except:
    from .detr.main import (
        build_ACT_model_and_optimizer,
        build_CNNMLP_model_and_optimizer,
    )
    from .asym_bimanual_adapter import (
        AsymFreqResidualAdapter,
        LearnedAsymFreqResidualAdapter,
        bernoulli_kl,
        motion_energy_update_target,
        motion_role_target,
    )
import IPython

e = IPython.embed


class ACTPolicy(nn.Module):

    def __init__(self, args_override, RoboTwin_Config=None):
        super().__init__()
        model, optimizer = build_ACT_model_and_optimizer(args_override, RoboTwin_Config)
        self.model = model  # CVAE decoder
        self.optimizer = optimizer
        self.kl_weight = args_override["kl_weight"]
        self.use_asym_residual = bool(args_override.get("use_asym_residual", False))
        self.asym_adapter = None
        self.asym_loss_weights = {
            "res": float(args_override.get("asym_lambda_res", 1e-4)),
            "gate": float(args_override.get("asym_lambda_gate", 1e-4)),
            "smooth": float(args_override.get("asym_lambda_smooth", 1e-3)),
            "freq": float(args_override.get("asym_lambda_freq", 1e-2)),
            "update_kl": float(args_override.get("asym_lambda_update_kl", 1e-2)),
            "update_sparse": float(args_override.get("asym_lambda_update_sparse", 1e-3)),
            "role": float(args_override.get("asym_lambda_role", 1e-2)),
        }
        self.asym_use_qpos_condition = bool(args_override.get("asym_use_qpos_condition", True))
        self.asym_schedule_mode = args_override.get("asym_schedule_mode", "fixed")
        self.asym_target_mode = args_override.get("asym_target_mode", "projected")
        self.asym_stride_choices = self._parse_stride_choices(args_override.get("asym_stride_choices", ""))
        self.left_stride = int(args_override.get("left_stride", 1))
        self.right_stride = int(args_override.get("right_stride", 1))
        if self.use_asym_residual:
            action_dim = int(args_override.get("action_dim", getattr(RoboTwin_Config, "action_dim", 14)))
            left_dim = int(args_override.get("left_action_dim", action_dim // 2))
            right_dim = int(args_override.get("right_action_dim", action_dim - left_dim))
            left_start = int(args_override.get("left_action_start", 0))
            right_start = int(args_override.get("right_action_start", left_start + left_dim))
            left_slice = slice(left_start, left_start + left_dim)
            right_slice = slice(right_start, right_start + right_dim)
            cond_dim = int(args_override.get("asym_cond_dim", action_dim if self.asym_use_qpos_condition else 0))
            max_stride = max(
                [int(args_override.get("asym_max_stride", 16)), self.left_stride, self.right_stride]
                + [max(pair) for pair in self.asym_stride_choices]
            )

            if self.asym_schedule_mode == "fixed":
                self.asym_adapter = AsymFreqResidualAdapter(
                    action_dim=action_dim,
                    horizon=self.model.num_queries,
                    left_slice=left_slice,
                    right_slice=right_slice,
                    left_stride=self.left_stride,
                    right_stride=self.right_stride,
                    cond_dim=cond_dim,
                    hidden_dim=int(args_override.get("asym_hidden_dim", 256)),
                    max_stride=max_stride,
                    residual_scale=float(args_override.get("asym_residual_scale", 0.1)),
                    projector_mode=args_override.get("asym_projector_mode", "hold"),
                    mask_residual_non_update=bool(args_override.get("asym_strict_execution", True)),
                )
            elif self.asym_schedule_mode == "learned":
                self.asym_adapter = LearnedAsymFreqResidualAdapter(
                    action_dim=action_dim,
                    horizon=self.model.num_queries,
                    left_slice=left_slice,
                    right_slice=right_slice,
                    cond_dim=cond_dim,
                    hidden_dim=int(args_override.get("asym_hidden_dim", 256)),
                    residual_scale=float(args_override.get("asym_residual_scale", 0.1)),
                    hard_inference=bool(args_override.get("asym_hard_inference", True)),
                    update_threshold=float(args_override.get("asym_update_threshold", 0.5)),
                    num_roles=int(args_override.get("asym_num_roles", 4)),
                    role_emb_dim=int(args_override.get("asym_role_emb_dim", 16)),
                    use_role_condition=bool(args_override.get("asym_use_role_condition", True)),
                )
            else:
                raise ValueError(f"Unsupported asym_schedule_mode: {self.asym_schedule_mode}")
            self.optimizer.add_param_group({
                "params": self.asym_adapter.parameters(),
                "lr": args_override["lr"],
            })
        print(f"KL Weight {self.kl_weight}")

    @staticmethod
    def _parse_stride_choices(stride_choices):
        if not stride_choices:
            return []
        parsed = []
        for item in stride_choices.split(";"):
            item = item.strip()
            if not item:
                continue
            left_stride, right_stride = item.split(",")
            parsed.append((int(left_stride), int(right_stride)))
        return parsed

    def _sample_strides(self, is_training):
        if self.asym_schedule_mode != "fixed":
            return None, None
        if is_training and self.asym_stride_choices:
            choice_id = torch.randint(len(self.asym_stride_choices), (1, )).item()
            return self.asym_stride_choices[choice_id]
        return self.left_stride, self.right_stride

    def _asym_cond(self, qpos):
        if not self.asym_use_qpos_condition:
            return None
        return qpos

    def _apply_asym_adapter(self, actions, qpos, left_stride=None, right_stride=None, role_id=None):
        if not self.use_asym_residual:
            return actions, None
        if self.asym_schedule_mode == "fixed":
            return self.asym_adapter(
                actions,
                cond=self._asym_cond(qpos),
                left_stride=left_stride,
                right_stride=right_stride,
            )
        return self.asym_adapter(actions, cond=self._asym_cond(qpos), role_id=role_id)

    def _build_training_target(self, actions, left_stride=None, right_stride=None):
        if (
            not self.use_asym_residual
            or self.asym_schedule_mode != "fixed"
            or self.asym_target_mode == "original"
        ):
            return actions
        if self.asym_target_mode != "projected":
            raise ValueError(f"Unsupported asym_target_mode: {self.asym_target_mode}")
        projected_actions, _ = self.asym_adapter.projector(
            actions,
            left_stride=left_stride,
            right_stride=right_stride,
        )
        return projected_actions.detach()

    def _asym_regularization(self, a_hat, target_actions, adapter_info, is_pad):
        if adapter_info is None:
            return {}

        valid = (~is_pad).float().unsqueeze(-1)
        valid_count = valid.sum().clamp(min=1.0)
        delta = adapter_info["delta"]
        gate = adapter_info["gate"]
        final_action = a_hat

        losses = {}
        losses["asym_res"] = (delta.pow(2) * valid).sum() / (valid_count * delta.shape[-1])
        losses["asym_gate"] = (gate.abs() * valid).sum() / (valid_count * gate.shape[-1])

        if final_action.shape[1] > 1:
            smooth_valid = valid[:, 1:] * valid[:, :-1]
            smooth_count = smooth_valid.sum().clamp(min=1.0)
            smooth_diff = final_action[:, 1:] - final_action[:, :-1]
            losses["asym_smooth"] = (smooth_diff.pow(2) * smooth_valid).sum() / (
                smooth_count * final_action.shape[-1]
            )

            left_start, left_stop = adapter_info["frequency"]["left_slice"]
            right_start, right_stop = adapter_info["frequency"]["right_slice"]
            right_mask = adapter_info["frequency"].get(
                "right_update_gate",
                adapter_info["frequency"].get("right_update_mask"),
            )
            left_mask = adapter_info["frequency"].get(
                "left_update_gate",
                adapter_info["frequency"].get("left_update_mask"),
            )
            freq_valid = smooth_valid[:, :, :1]

            left_action = final_action[:, :, left_start:left_stop]
            left_diff = left_action[:, 1:] - left_action[:, :-1]
            left_non_update = 1.0 - left_mask[:, 1:]
            left_count = (left_non_update * freq_valid).sum().clamp(min=1.0)
            left_freq = (left_diff.pow(2) * left_non_update * freq_valid).sum() / (
                left_count * left_diff.shape[-1]
            )

            right_action = final_action[:, :, right_start:right_stop]
            right_diff = right_action[:, 1:] - right_action[:, :-1]
            right_non_update = 1.0 - right_mask[:, 1:]
            right_count = (right_non_update * freq_valid).sum().clamp(min=1.0)
            right_freq = (right_diff.pow(2) * right_non_update * freq_valid).sum() / (
                right_count * right_diff.shape[-1]
            )
            losses["asym_freq"] = 0.5 * (left_freq + right_freq)
        else:
            zero = final_action.sum() * 0.0
            losses["asym_smooth"] = zero
            losses["asym_freq"] = zero

        if "update_prob" in adapter_info:
            left_start, left_stop = adapter_info["frequency"]["left_slice"]
            right_start, right_stop = adapter_info["frequency"]["right_slice"]
            target_update = motion_energy_update_target(
                target_actions,
                slice(left_start, left_stop),
                slice(right_start, right_stop),
            )
            update_prob = adapter_info["update_prob"]
            update_valid = (~is_pad).float().unsqueeze(-1)
            update_count = update_valid.sum().clamp(min=1.0)
            update_kl = bernoulli_kl(target_update, update_prob)
            losses["asym_update_kl"] = (update_kl * update_valid).sum() / (update_count * update_kl.shape[-1])
            losses["asym_update_sparse"] = (update_prob[:, 1:] * update_valid[:, 1:]).sum() / (
                update_valid[:, 1:].sum().clamp(min=1.0) * update_prob.shape[-1]
            )
            role_target = motion_role_target(
                target_actions,
                slice(left_start, left_stop),
                slice(right_start, right_stop),
            )
            losses["asym_role"] = F.cross_entropy(adapter_info["role_logits"], role_target)

        return losses

    def __call__(self, qpos, image, actions=None, is_pad=None, return_info=False, role_id=None):
        env_state = None
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        image = normalize(image)
        if actions is not None:  # training time
            actions = actions[:, :self.model.num_queries]
            is_pad = is_pad[:, :self.model.num_queries]
            left_stride, right_stride = self._sample_strides(is_training=self.training)
            target_actions = self._build_training_target(actions, left_stride, right_stride)

            a_hat, is_pad_hat, (mu, logvar) = self.model(qpos, image, env_state, target_actions, is_pad)
            a_hat, adapter_info = self._apply_asym_adapter(a_hat, qpos, left_stride, right_stride, role_id=role_id)
            total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
            loss_dict = dict()
            all_l1 = F.l1_loss(target_actions, a_hat, reduction="none")
            l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
            loss_dict["l1"] = l1
            loss_dict["kl"] = total_kld[0]
            loss_dict["loss"] = loss_dict["l1"] + loss_dict["kl"] * self.kl_weight
            for name, loss in self._asym_regularization(a_hat, target_actions, adapter_info, is_pad).items():
                loss_dict[name] = loss
            if adapter_info is not None:
                for key, weight in self.asym_loss_weights.items():
                    loss_name = f"asym_{key}"
                    if loss_name in loss_dict:
                        loss_dict["loss"] = loss_dict["loss"] + weight * loss_dict[loss_name]
            return loss_dict
        else:  # inference time
            a_hat, _, (_, _) = self.model(qpos, image, env_state)  # no action, sample from prior
            left_stride, right_stride = self._sample_strides(is_training=False)
            a_hat, adapter_info = self._apply_asym_adapter(a_hat, qpos, left_stride, right_stride, role_id=role_id)
            if return_info:
                return a_hat, adapter_info
            return a_hat

    def configure_optimizers(self):
        return self.optimizer


class CNNMLPPolicy(nn.Module):

    def __init__(self, args_override):
        super().__init__()
        model, optimizer = build_CNNMLP_model_and_optimizer(args_override)
        self.model = model  # decoder
        self.optimizer = optimizer

    def __call__(self, qpos, image, actions=None, is_pad=None):
        env_state = None  # TODO
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        image = normalize(image)
        if actions is not None:  # training time
            actions = actions[:, 0]
            a_hat = self.model(qpos, image, env_state, actions)
            mse = F.mse_loss(actions, a_hat)
            loss_dict = dict()
            loss_dict["mse"] = mse
            loss_dict["loss"] = loss_dict["mse"]
            return loss_dict
        else:  # inference time
            a_hat = self.model(qpos, image, env_state)  # no action, sample from prior
            return a_hat

    def configure_optimizers(self):
        return self.optimizer


def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld


class ACT:
    ROLE_NAME_TO_ID = {
        "balanced": 0,
        "equal": 0,
        "left_primary": 1,
        "left_high": 1,
        "left_operate": 1,
        "right_primary": 2,
        "right_high": 2,
        "right_operate": 2,
        "static": 3,
        "hold": 3,
    }


    def __init__(self, args_override=None, RoboTwin_Config=None):
        if args_override is None:
            args_override = {
                "kl_weight": 0.1,  # Default value, can be overridden
                "device": "cuda:0",
            }
        self.policy = ACTPolicy(args_override, RoboTwin_Config)
        self.device = torch.device(args_override["device"])
        self.policy.to(self.device)
        self.policy.eval()

        # Temporal aggregation settings
        self.temporal_agg = args_override.get("temporal_agg", False)
        self.num_queries = args_override["chunk_size"]
        self.state_dim = RoboTwin_Config.action_dim  # Standard joint dimension for bimanual robot
        self.max_timesteps = 3000  # Large enough for deployment

        # Set query frequency based on temporal_agg - matching imitate_episodes.py logic
        self.query_frequency = self.num_queries
        if self.temporal_agg:
            self.query_frequency = 1
            # Initialize with zeros matching imitate_episodes.py format
            self.all_time_actions = torch.zeros([
                self.max_timesteps,
                self.max_timesteps + self.num_queries,
                self.state_dim,
            ]).to(self.device)
            print(f"Temporal aggregation enabled with {self.num_queries} queries")

        self.t = 0  # Current timestep
        self.last_asym_info = None
        self.last_action_packet = None

        # Load statistics for normalization
        ckpt_dir = args_override.get("ckpt_dir", "")
        if ckpt_dir:
            # Load dataset stats for normalization
            stats_path = os.path.join(ckpt_dir, "dataset_stats.pkl")
            if os.path.exists(stats_path):
                with open(stats_path, "rb") as f:
                    self.stats = pickle.load(f)
                print(f"Loaded normalization stats from {stats_path}")
            else:
                print(f"Warning: Could not find stats file at {stats_path}")
                self.stats = None

            # Load policy weights
            ckpt_path = os.path.join(ckpt_dir, "policy_best.ckpt")
            print("current pwd:", os.getcwd())
            if os.path.exists(ckpt_path):
                loading_status = self.policy.load_state_dict(
                    torch.load(ckpt_path),
                    strict=not self.policy.use_asym_residual,
                )
                print(f"Loaded policy weights from {ckpt_path}")
                print(f"Loading status: {loading_status}")
            else:
                print(f"Warning: Could not find policy checkpoint at {ckpt_path}")
        else:
            self.stats = None

    def pre_process(self, qpos):
        """Normalize input joint positions"""
        if self.stats is not None:
            return (qpos - self.stats["qpos_mean"]) / self.stats["qpos_std"]
        return qpos

    def post_process(self, action):
        """Denormalize model outputs"""
        if self.stats is not None:
            return action * self.stats["action_std"] + self.stats["action_mean"]
        return action

    def _obs_role_id(self, obs):
        role = obs.get("asym_role_id", obs.get("asym_role", None))
        if role is None and "instruction" in obs:
            instruction = str(obs["instruction"]).lower()
            if "left" in instruction and ("operate" in instruction or "move" in instruction or "primary" in instruction):
                role = "left_primary"
            elif "right" in instruction and ("operate" in instruction or "move" in instruction or "primary" in instruction):
                role = "right_primary"
            elif "hold" in instruction or "fix" in instruction or "stable" in instruction:
                if "left" in instruction and "right" not in instruction:
                    role = "right_primary"
                elif "right" in instruction and "left" not in instruction:
                    role = "left_primary"
                else:
                    role = "static"
            elif "both" in instruction or "together" in instruction or "dual" in instruction:
                role = "balanced"
        if role is None:
            return None
        if isinstance(role, str):
            role = self.ROLE_NAME_TO_ID.get(role, 0)
        return torch.tensor([int(role)], device=self.device)

    def _current_update_value(self, info, name, chunk_step):
        if info is None:
            return True
        frequency = info.get("frequency", {})
        gate = frequency.get(f"{name}_update_gate", frequency.get(f"{name}_update_mask"))
        if gate is None:
            return True
        value = gate[0, chunk_step, 0].detach().cpu().item()
        return bool(value >= 0.5)

    def _build_action_packet(self, action, chunk_step):
        action_1d = np.asarray(action).reshape(-1)
        left_slice = (0, self.state_dim // 2)
        right_slice = (self.state_dim // 2, self.state_dim)
        if self.last_asym_info is not None:
            frequency = self.last_asym_info.get("frequency", {})
            left_slice = frequency.get("left_slice", left_slice)
            right_slice = frequency.get("right_slice", right_slice)

        packet = {
            "action": action_1d,
            "left_action": action_1d[left_slice[0]:left_slice[1]],
            "right_action": action_1d[right_slice[0]:right_slice[1]],
            "left_update": self._current_update_value(self.last_asym_info, "left", chunk_step),
            "right_update": self._current_update_value(self.last_asym_info, "right", chunk_step),
            "left_slice": left_slice,
            "right_slice": right_slice,
        }
        if self.last_asym_info is not None and "role_prob" in self.last_asym_info:
            packet["role_prob"] = self.last_asym_info["role_prob"][0].detach().cpu().numpy()
        return packet

    def get_action(self, obs=None, return_packet=False):
        if obs is None:
            return None

        # Convert observations to tensors and normalize qpos - matching imitate_episodes.py
        qpos_numpy = np.array(obs["qpos"])
        qpos_normalized = self.pre_process(qpos_numpy)
        qpos = torch.from_numpy(qpos_normalized).float().to(self.device).unsqueeze(0)

        # Prepare images following imitate_episodes.py pattern
        # Stack images from all cameras
        curr_images = []
        camera_names = ["head_cam", "left_cam", "right_cam"]
        for cam_name in camera_names:
            curr_images.append(obs[cam_name])
        curr_image = np.stack(curr_images, axis=0)
        curr_image = torch.from_numpy(curr_image).float().to(self.device).unsqueeze(0)
        role_id = self._obs_role_id(obs)

        with torch.no_grad():
            # Only query the policy at specified intervals - exactly like imitate_episodes.py
            if self.t % self.query_frequency == 0:
                if self.policy.use_asym_residual:
                    self.all_actions, self.last_asym_info = self.policy(
                        qpos,
                        curr_image,
                        return_info=True,
                        role_id=role_id,
                    )
                else:
                    self.all_actions = self.policy(qpos, curr_image)
                    self.last_asym_info = None

            if self.temporal_agg:
                # Match temporal aggregation exactly from imitate_episodes.py
                self.all_time_actions[[self.t], self.t:self.t + self.num_queries] = (self.all_actions)
                actions_for_curr_step = self.all_time_actions[:, self.t]
                actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                actions_for_curr_step = actions_for_curr_step[actions_populated]

                # Use same weighting factor as in imitate_episodes.py
                k = 0.01
                exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                exp_weights = exp_weights / exp_weights.sum()
                exp_weights = (torch.from_numpy(exp_weights).to(self.device).unsqueeze(dim=1))

                raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
            else:
                # Direct action selection, same as imitate_episodes.py
                chunk_step = self.t % self.query_frequency
                raw_action = self.all_actions[:, chunk_step]

        # Denormalize action
        raw_action = raw_action.cpu().numpy()
        action = self.post_process(raw_action)
        if self.temporal_agg:
            chunk_step = min(self.t, self.num_queries - 1)
        packet = self._build_action_packet(action, chunk_step)
        self.last_action_packet = packet

        self.t += 1
        if return_packet:
            return packet
        return action