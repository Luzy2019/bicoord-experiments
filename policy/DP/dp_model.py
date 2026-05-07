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

    def get_action(self, observation=None):
        if self.debug_dir:
            setattr(self.policy, "factorized_debug_actions", len(self.debug_records) < self.debug_max_calls)
        action = self.runner.get_action(self.policy, observation)
        self._collect_factorized_debug()
        if self.debug_dir:
            setattr(self.policy, "factorized_debug_actions", False)
        return action

    def get_last_obs(self):
        return self.runner.obs[-1]

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