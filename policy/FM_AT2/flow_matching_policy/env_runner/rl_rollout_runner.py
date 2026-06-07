from collections import deque
import importlib
import os
import sys
from typing import Dict, List

import numpy as np
import torch

from flow_matching_policy.common.pytorch_util import dict_apply


def _cfg_get(cfg, key, default=None):
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _to_float(value, default=0.0):
    try:
        if isinstance(value, np.ndarray):
            return float(value.reshape(-1)[0])
        return float(value)
    except Exception:
        return float(default)


class DefaultRewardComputer:
    """Task-progress reward used by self-imitation RL and reranker updates."""

    def __init__(self, cfg=None):
        reward_cfg = _cfg_get(cfg, "reward", {})
        self.success_bonus = _cfg_get(reward_cfg, "success_bonus", 10.0)
        self.phase_delta_bonus = _cfg_get(reward_cfg, "phase_delta_bonus", 3.0)
        self.early_phase_bonus = _cfg_get(reward_cfg, "early_phase_bonus", 2.0)
        self.time_weight = _cfg_get(reward_cfg, "time_weight", 0.01)
        self.makespan_weight = _cfg_get(reward_cfg, "makespan_weight", 1.0)
        self.dag_weight = _cfg_get(reward_cfg, "dag_weight", 1.0)
        self.dynamics_weight = _cfg_get(reward_cfg, "dynamics_weight", 0.5)
        self.speed_weight = _cfg_get(reward_cfg, "speed_weight", 0.2)
        self.collision_weight = _cfg_get(reward_cfg, "collision_weight", 10.0)
        self.failure_penalty = _cfg_get(reward_cfg, "failure_penalty", 5.0)
        self.speed_limit = _cfg_get(reward_cfg, "speed_limit", 2.0)

    def __call__(
        self,
        prev_info: Dict,
        next_info: Dict,
        policy_output: Dict[str, torch.Tensor],
        final_failure: bool = False,
    ) -> float:
        prev_stage = _to_float(prev_info.get("stage_eval_score", 0.0))
        next_stage = _to_float(next_info.get("stage_eval_score", prev_stage))
        phase_delta = max(next_stage - prev_stage, 0.0)
        env_step = max(_to_float(next_info.get("env_step", next_info.get("task_time", 1.0)), 1.0), 1.0)
        success = bool(next_info.get("success", next_info.get("eval_success", False)))
        collision = bool(next_info.get("collision", False))

        reward_features = policy_output.get("reward_features", {})
        makespan = _to_float(reward_features.get("makespan", policy_output.get("makespan", 0.0)))
        dag_cost = _to_float(reward_features.get("dag_dependency_cost", policy_output.get("dag_dependency_cost", 0.0)))
        dynamics = _to_float(reward_features.get("wm_scores", policy_output.get("wm_scores", 0.0)))
        speed_max = _to_float(reward_features.get("speed_scale_max", 1.0), default=1.0)

        reward = 0.0
        reward += self.success_bonus * float(success)
        reward += self.phase_delta_bonus * phase_delta
        reward += self.early_phase_bonus * phase_delta / env_step
        reward -= self.time_weight * env_step
        reward -= self.makespan_weight * makespan
        reward -= self.dag_weight * dag_cost
        reward -= self.dynamics_weight * dynamics
        reward -= self.speed_weight * max(speed_max - self.speed_limit, 0.0)
        reward -= self.collision_weight * float(collision)
        reward -= self.failure_penalty * float(final_failure)
        return float(reward)


class RLRolloutBuffer:
    """Small in-memory rollout buffer for reward-weighted FM and reranker PPO."""

    def __init__(self, capacity: int = 2048):
        self.capacity = int(capacity)
        self.items: List[Dict] = []

    def __len__(self):
        return len(self.items)

    def add(self, item: Dict):
        self.items.append(item)
        if len(self.items) > self.capacity:
            self.items = self.items[-self.capacity:]

    @staticmethod
    def _stack_obs(items: List[Dict]) -> Dict[str, torch.Tensor]:
        keys = items[0]["obs"].keys()
        return {
            key: torch.from_numpy(np.stack([item["obs"][key] for item in items]).astype(np.float32))
            for key in keys
        }

    @staticmethod
    def _maybe_stack(items: List[Dict], key: str):
        if key not in items[0]:
            return None
        values = [item[key] for item in items if item.get(key) is not None]
        if len(values) != len(items):
            return None
        return torch.from_numpy(np.stack(values).astype(np.float32))

    def iter_batches(self, batch_size: int, device):
        if not self.items:
            return
        indices = np.random.permutation(len(self.items))
        batch_size = max(int(batch_size), 1)
        for start in range(0, len(indices), batch_size):
            batch_items = [self.items[idx] for idx in indices[start:start + batch_size]]
            batch = {
                "obs": dict_apply(self._stack_obs(batch_items), lambda x: x.to(device=device)),
                "action": torch.from_numpy(np.stack([item["action"] for item in batch_items]).astype(np.float32)).to(device),
                "reward": torch.tensor([item["reward"] for item in batch_items], dtype=torch.float32, device=device),
                "return": torch.tensor([item["return"] for item in batch_items], dtype=torch.float32, device=device),
            }
            optional_keys = (
                "candidate_actions",
                "candidate_schedules",
                "candidate_schedule_durations",
            )
            for key in optional_keys:
                stacked = self._maybe_stack(batch_items, key)
                if stacked is not None:
                    batch[key] = stacked.to(device=device)
            if "selected_candidate_idx" in batch_items[0]:
                batch["selected_candidate_idx"] = torch.tensor(
                    [item["selected_candidate_idx"] for item in batch_items],
                    dtype=torch.long,
                    device=device,
                )
            if "selected_logprob" in batch_items[0]:
                batch["selected_logprob"] = torch.tensor(
                    [item["selected_logprob"] for item in batch_items],
                    dtype=torch.float32,
                    device=device,
                )
            yield batch


class IsaacLabEnvAdapter:
    """Adapter for IsaacLab/Gymnasium environments."""

    def __init__(self, env_id: str, execute_compact_schedule="auto", env_kwargs=None):
        if env_id is None:
            raise ValueError("rl.env_id must be set for IsaacLabEnvAdapter")
        try:
            import gymnasium as gym
        except ImportError as exc:
            raise ImportError("gymnasium is required for IsaacLabEnvAdapter") from exc
        self.env = gym.make(env_id, **(env_kwargs or {}))
        self.execute_compact_schedule = execute_compact_schedule
        self.last_info = {}

    @staticmethod
    def encode_obs(obs) -> Dict[str, np.ndarray]:
        if isinstance(obs, dict) and all(key in obs for key in ("head_cam", "left_cam", "right_cam", "agent_pos")):
            return {
                "head_cam": np.asarray(obs["head_cam"], dtype=np.float32),
                "left_cam": np.asarray(obs["left_cam"], dtype=np.float32),
                "right_cam": np.asarray(obs["right_cam"], dtype=np.float32),
                "agent_pos": np.asarray(obs["agent_pos"], dtype=np.float32),
            }
        if isinstance(obs, dict) and "observation" in obs and "joint_action" in obs:
            return encode_robotwin_obs(obs)
        raise ValueError("IsaacLab obs must expose head_cam/left_cam/right_cam/agent_pos or RoboTwin-style keys")

    def reset(self, seed=None):
        out = self.env.reset(seed=seed)
        obs, info = out if isinstance(out, tuple) else (out, {})
        self.last_info = dict(info or {})
        return self.encode_obs(obs)

    def _should_execute_schedule(self):
        if self.execute_compact_schedule is True:
            return True
        if self.execute_compact_schedule == "auto":
            return bool(getattr(self.env.unwrapped, "supports_compact_schedule", False))
        return False

    def step_action_chunk(self, actions: np.ndarray, policy_output: Dict):
        obs_history = []
        infos = []
        done = False
        obs = None
        if self._should_execute_schedule():
            payload = {
                "action": actions,
                "compact_schedule": np.asarray(policy_output["compact_schedule"]).squeeze(0),
                "durations": np.asarray(policy_output["compact_schedule_durations"]).squeeze(0),
            }
            obs, reward, terminated, truncated, info = self.env.step(payload)
            done = bool(terminated or truncated)
            info = dict(info or {})
            info.setdefault("env_reward", reward)
            infos.append(info)
            obs_history.append(self.encode_obs(obs))
        else:
            for action in actions:
                obs, reward, terminated, truncated, info = self.env.step(action)
                done = bool(terminated or truncated)
                info = dict(info or {})
                info.setdefault("env_reward", reward)
                infos.append(info)
                obs_history.append(self.encode_obs(obs))
                if done:
                    break
        self.last_info = infos[-1] if infos else self.last_info
        return obs_history[-1], infos, obs_history, done

    def close(self):
        self.env.close()


def encode_robotwin_obs(observation) -> Dict[str, np.ndarray]:
    head_cam = np.moveaxis(observation["observation"]["head_camera"]["rgb"], -1, 0) / 255.0
    left_cam = np.moveaxis(observation["observation"]["left_camera"]["rgb"], -1, 0) / 255.0
    right_cam = np.moveaxis(observation["observation"]["right_camera"]["rgb"], -1, 0) / 255.0
    return {
        "head_cam": head_cam.astype(np.float32),
        "left_cam": left_cam.astype(np.float32),
        "right_cam": right_cam.astype(np.float32),
        "agent_pos": np.asarray(observation["joint_action"]["vector"], dtype=np.float32),
    }


class RoboTwinEnvAdapter:
    """Fallback adapter for the local RoboTwin/SAPIEN task envs."""

    def __init__(self, task_name: str, task_config: str, seed: int = 0):
        repo_root = os.path.abspath(os.path.join(os.getcwd(), "..", ".."))
        if repo_root not in sys.path:
            sys.path.append(repo_root)
        import yaml

        envs_module = importlib.import_module(f"envs.{task_name}")
        self.env = getattr(envs_module, task_name)()
        config_path = os.path.join(repo_root, "task_config", f"{task_config}.yml")
        with open(config_path, "r", encoding="utf-8") as f:
            self.args = yaml.safe_load(f)
        self.args["task_name"] = task_name
        self.args["eval_mode"] = True
        self.seed = int(seed)
        self.episode_idx = 0
        self.last_info = {}

    def reset(self, seed=None):
        if seed is None:
            seed = self.seed + self.episode_idx
        if self.episode_idx > 0:
            self.env.close_env(clear_cache=False)
        self.env.setup_demo(now_ep_num=self.episode_idx, seed=seed, is_test=True, **self.args)
        self.episode_idx += 1
        obs = encode_robotwin_obs(self.env.get_obs())
        self.last_info = {
            "stage_eval_score": getattr(self.env, "stage_eval_score", 0.0),
            "eval_success": getattr(self.env, "eval_success", False),
            "env_step": getattr(self.env, "take_action_cnt", 0),
        }
        return obs

    def step_action_chunk(self, actions: np.ndarray, policy_output: Dict):
        obs_history = []
        infos = []
        done = False
        for action in actions:
            self.env.take_action(action)
            obs = encode_robotwin_obs(self.env.get_obs())
            info = {
                "stage_eval_score": getattr(self.env, "stage_eval_score", 0.0),
                "success": getattr(self.env, "eval_success", False),
                "eval_success": getattr(self.env, "eval_success", False),
                "env_step": getattr(self.env, "take_action_cnt", 0),
            }
            infos.append(info)
            obs_history.append(obs)
            if info["eval_success"] or self.env.take_action_cnt >= self.env.step_lim:
                done = True
                break
        self.last_info = infos[-1] if infos else self.last_info
        return obs_history[-1], infos, obs_history, done

    def close(self):
        self.env.close_env(clear_cache=False)


class RLRolloutRunner:
    """Collect rollout chunks for self-imitation RL."""

    def __init__(self, cfg, task_name=None, task_config=None, seed=0):
        self.cfg = cfg
        self.task_name = task_name
        self.task_config = task_config
        self.seed = int(seed)
        self.n_obs_steps = int(_cfg_get(cfg, "n_obs_steps", 3))
        self.max_episode_steps = int(_cfg_get(cfg, "max_episode_steps", 300))
        self.rollout_episodes = int(_cfg_get(cfg, "rollout_episodes_per_epoch", 4))
        self.gamma = float(_cfg_get(cfg, "gamma", 0.99))
        self.buffer = RLRolloutBuffer(capacity=int(_cfg_get(cfg, "replay_capacity", 2048)))
        self.reward_computer = DefaultRewardComputer(cfg)
        backend = str(_cfg_get(cfg, "env_backend", "isaaclab")).lower()
        if backend == "robotwin":
            self.adapter = RoboTwinEnvAdapter(task_name, task_config, seed=seed)
        else:
            self.adapter = IsaacLabEnvAdapter(
                _cfg_get(cfg, "env_id", None),
                execute_compact_schedule=_cfg_get(cfg, "execute_compact_schedule", "auto"),
                env_kwargs=_cfg_get(cfg, "env_kwargs", None),
            )

    @staticmethod
    def _stack_last_n_obs(all_obs: deque, n_steps: int) -> Dict[str, np.ndarray]:
        result = {}
        for key in all_obs[0].keys():
            values = [obs[key] for obs in all_obs]
            out = np.zeros((n_steps,) + values[-1].shape, dtype=values[-1].dtype)
            start_idx = -min(n_steps, len(values))
            out[start_idx:] = np.asarray(values[start_idx:])
            if n_steps > len(values):
                out[:start_idx] = out[start_idx]
            result[key] = out.astype(np.float32)
        return result

    @staticmethod
    def _to_policy_obs(obs_chunk: Dict[str, np.ndarray], device):
        return dict_apply(obs_chunk, lambda x: torch.from_numpy(x).to(device=device).unsqueeze(0))

    @staticmethod
    def _numpy_value(value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
        if isinstance(value, dict):
            return {k: RLRolloutRunner._numpy_value(v) for k, v in value.items()}
        return value

    def collect(self, policy) -> Dict[str, float]:
        device = policy.device
        episode_returns = []
        successes = []
        for episode_idx in range(self.rollout_episodes):
            obs_queue = deque(maxlen=self.n_obs_steps + 1)
            first_obs = self.adapter.reset(seed=self.seed + episode_idx)
            obs_queue.append(first_obs)
            done = False
            episode_items = []
            total_reward = 0.0
            while (not done) and len(episode_items) < self.max_episode_steps:
                obs_chunk = self._stack_last_n_obs(obs_queue, self.n_obs_steps)
                policy_obs = self._to_policy_obs(obs_chunk, device)
                with torch.no_grad():
                    action_dict = policy.predict_action(
                        policy_obs,
                        stochastic_select=True,
                        return_candidate_batch=True,
                        rl_mode=True,
                    )
                np_action_dict = self._numpy_value(action_dict)
                actions = np_action_dict["action"].squeeze(0)
                prev_info = dict(self.adapter.last_info)
                final_obs, infos, obs_history, done = self.adapter.step_action_chunk(actions, np_action_dict)
                for obs in obs_history:
                    obs_queue.append(obs)
                next_info = dict(infos[-1] if infos else self.adapter.last_info)
                reward = self.reward_computer(prev_info, next_info, np_action_dict, final_failure=False)
                total_reward += reward
                item = {
                    "obs": obs_chunk,
                    "action": np_action_dict["action_pred"].squeeze(0).astype(np.float32),
                    "reward": reward,
                    "selected_candidate_idx": int(np.asarray(np_action_dict["selected_candidate_idx"]).reshape(-1)[0]),
                    "selected_logprob": float(np.asarray(np_action_dict["selected_logprob"]).reshape(-1)[0]),
                    "candidate_actions": np_action_dict["candidate_actions"].squeeze(0).astype(np.float32),
                    "candidate_schedules": np_action_dict["candidate_schedules"].squeeze(0).astype(np.float32),
                    "candidate_schedule_durations": np_action_dict["candidate_schedule_durations"].squeeze(0).astype(np.float32),
                }
                episode_items.append(item)
            success = bool(self.adapter.last_info.get("success", self.adapter.last_info.get("eval_success", False)))
            if episode_items and not success:
                episode_items[-1]["reward"] += self.reward_computer(
                    self.adapter.last_info,
                    self.adapter.last_info,
                    {},
                    final_failure=True,
                )
            running_return = 0.0
            for item in reversed(episode_items):
                running_return = item["reward"] + self.gamma * running_return
                item["return"] = running_return
            for item in episode_items:
                self.buffer.add(item)
            episode_returns.append(total_reward)
            successes.append(float(success))
        return {
            "rl_rollout_return": float(np.mean(episode_returns)) if episode_returns else 0.0,
            "rl_rollout_success_rate": float(np.mean(successes)) if successes else 0.0,
            "rl_rollout_buffer_size": len(self.buffer),
        }

    def iter_batches(self, batch_size: int, device):
        yield from self.buffer.iter_batches(batch_size, device)

    def close(self):
        self.adapter.close()
