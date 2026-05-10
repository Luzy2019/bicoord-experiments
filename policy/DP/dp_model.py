import numpy as np
import torch
import hydra
import dill
import sys, os
import atexit
from pathlib import Path

current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(current_file_path)
sys.path.append(parent_dir)

from diffusion_policy.workspace.robotworkspace import RobotWorkspace
from diffusion_policy.env_runner.dp_runner import DPRunner

class DP:

    def __init__(self, ckpt_file: str):
        self.policy = self.get_policy(ckpt_file, None, "cuda:0")
        self._apply_runtime_speed_modulation()
        self.runner = DPRunner(output_dir=None)
        self.policy_call_idx = 0
        self.last_policy_debug = {}
        self.current_eval_action = None
        self.debug_dir = os.environ.get("DP_FACTOR_DEBUG_DIR")
        self.debug_records = []
        self.debug_max_calls = int(os.environ.get("DP_FACTOR_DEBUG_MAX_CALLS", "20"))
        if self.debug_dir:
            Path(self.debug_dir).mkdir(parents=True, exist_ok=True)
            atexit.register(self._flush_factorized_debug)

    def update_obs(self, observation):
        self.runner.update_obs(observation)
    
    def reset_obs(self):
        self.runner.reset_obs()
        self.policy_call_idx = 0
        self.last_policy_debug = {}
        self.current_eval_action = None

    def get_action(self, observation=None):
        if self.debug_dir:
            setattr(self.policy, "factorized_debug_actions", len(self.debug_records) < self.debug_max_calls)
        action = self.runner.get_action(self.policy, observation)
        self.last_policy_debug = self._build_policy_debug(action)
        self.policy_call_idx += 1
        self._collect_factorized_debug()
        if self.debug_dir:
            setattr(self.policy, "factorized_debug_actions", False)
        return action

    def set_current_eval_action(self, action, action_idx, chunk_len):
        action = np.asarray(action, dtype=np.float32)
        mid = action.shape[-1] // 2
        self.current_eval_action = {
            "chunk_step": int(action_idx),
            "chunk_len": int(chunk_len),
            "action": action,
            "left_action": action[:mid],
            "right_action": action[mid:],
            "left_gripper": float(action[mid - 1]) if mid > 0 else None,
            "right_gripper": float(action[-1]) if action.size > 0 else None,
        }

    def get_debug_overlay(self):
        info = dict(self.last_policy_debug)
        if self.current_eval_action is not None:
            info.update(self.current_eval_action)
        return self._json_safe(info)

    def get_last_obs(self):
        return self.runner.obs[-1]

    def _build_policy_debug(self, action):
        action_dict = getattr(self.runner, "last_action_dict", None) or {}
        info = {
            "policy_call_idx": int(self.policy_call_idx),
            "n_obs_steps": int(getattr(self.policy, "n_obs_steps", self.runner.n_obs_steps)),
            "n_action_steps": int(action.shape[0]),
            "action_dim": int(action.shape[-1]) if action.ndim > 1 else int(action.shape[0]),
            "speed_enabled": bool(getattr(self.policy, "speed_modulation_enabled", False)),
            "speed_learned": bool(getattr(self.policy, "speed_modulation_learned", False)),
            "speed_strength": float(getattr(self.policy, "speed_modulation_strength", 1.0)),
            "speed_min": float(getattr(self.policy, "speed_modulation_min", 0.0)),
            "speed_max": float(getattr(self.policy, "speed_modulation_max", 0.0)),
            "action_chunk": np.asarray(action, dtype=np.float32),
        }
        for key in ("speed_alpha", "left_speed_alpha", "right_speed_alpha"):
            if key in action_dict:
                value = np.asarray(action_dict[key]).squeeze()
                info[key] = value
                info[f"{key}_mean"] = float(np.mean(value))
                info[f"{key}_min"] = float(np.min(value))
                info[f"{key}_max"] = float(np.max(value))
        for key in ("action_pred", "action_pred_raw", "factorized_gates"):
            if key in action_dict:
                info[key] = np.asarray(action_dict[key]).squeeze()
        return info

    def _json_safe(self, value):
        if isinstance(value, dict):
            return {k: self._json_safe(v) for k, v in value.items()}
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, (list, tuple)):
            return [self._json_safe(v) for v in value]
        return value

    def get_policy(self, checkpoint, output_dir, device):
        # load checkpoint
        payload = torch.load(open(checkpoint, "rb"), pickle_module=dill)
        cfg = payload["cfg"]
        cls = hydra.utils.get_class(cfg._target_)
        workspace = cls(cfg, output_dir=output_dir)
        workspace: RobotWorkspace
        workspace.load_payload(payload, exclude_keys=("optimizer",), include_keys=None, strict=False)

        # get policy from workspace
        policy = workspace.model
        if cfg.training.use_ema:
            policy = workspace.ema_model

        device = torch.device(device)
        policy.to(device)
        policy.eval()

        return policy

    def _apply_runtime_speed_modulation(self):
        enabled = os.environ.get("DP_SPEED_MODULATION_ENABLED")
        if enabled is None:
            return
        enabled = enabled.lower() not in ("0", "false", "no", "none")
        setattr(self.policy, "speed_modulation_enabled", enabled)
        if os.environ.get("DP_SPEED_MODULATION_STRENGTH") is not None:
            setattr(self.policy, "speed_modulation_strength", float(os.environ["DP_SPEED_MODULATION_STRENGTH"]))
        if os.environ.get("DP_SPEED_MODULATION_LEARNED") is not None:
            learned = os.environ["DP_SPEED_MODULATION_LEARNED"].lower() not in ("0", "false", "no", "none")
            setattr(self.policy, "speed_modulation_learned", learned)
        if os.environ.get("DP_SPEED_MODULATION_MIN") is not None:
            setattr(self.policy, "speed_modulation_min", float(os.environ["DP_SPEED_MODULATION_MIN"]))
        if os.environ.get("DP_SPEED_MODULATION_MAX") is not None:
            setattr(self.policy, "speed_modulation_max", float(os.environ["DP_SPEED_MODULATION_MAX"]))
        if os.environ.get("DP_SPEED_MODULATION_SMOOTH") is not None:
            setattr(self.policy, "speed_modulation_smooth", int(os.environ["DP_SPEED_MODULATION_SMOOTH"]))

    def _collect_factorized_debug(self):
        if not self.debug_dir:
            return
        if len(self.debug_records) >= self.debug_max_calls:
            return
        debug_info = getattr(self.policy, "last_debug_info", None)

        record = {}
        if debug_info:
            record["timesteps"] = np.asarray([item["timestep"] for item in debug_info], dtype=np.int64)
            tensor_keys = [
                "factorized_gates",
                "left_marginal",
                "right_marginal",
                "left_cond",
                "right_cond",
                "left_pred",
                "right_pred",
            ]
            for key in tensor_keys:
                record[key] = np.stack([item[key].numpy() for item in debug_info], axis=0)

        action_debug_info = getattr(self.policy, "last_action_debug_info", None)
        if action_debug_info:
            for key, value in action_debug_info.items():
                record[key] = value.numpy()

        speed_debug_keys = [
            "last_speed_alpha",
            "last_left_speed_alpha",
            "last_right_speed_alpha",
            "last_action_pred_raw",
        ]
        for attr_name in speed_debug_keys:
            value = getattr(self.policy, attr_name, None)
            if value is not None:
                record[attr_name.replace("last_", "")] = value.detach().cpu().numpy()

        if record:
            self.debug_records.append(record)

    def _flush_factorized_debug(self):
        if not self.debug_dir or not self.debug_records:
            return

        output = {"num_policy_calls": np.asarray(len(self.debug_records), dtype=np.int64)}
        for key in self.debug_records[0].keys():
            output[key] = np.stack([record[key] for record in self.debug_records], axis=0)

        save_path = Path(self.debug_dir) / "factorized_debug.npz"
        np.savez_compressed(save_path, **output)
        print(f"[factorized debug] saved {save_path} calls={len(self.debug_records)}")