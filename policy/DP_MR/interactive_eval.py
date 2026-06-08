import argparse
import base64
import importlib
import json
import os
import sys
import time
from datetime import datetime
from io import BytesIO
from pathlib import Path

import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "policy"))
sys.path.insert(0, str(REPO_ROOT / "description" / "utils"))

from envs import CONFIGS_PATH
from envs.utils.create_actor import UnStableError
from generate_episode_instructions import generate_episode_descriptions
from script.eval_policy import class_decorator, get_embodiment_config


CAMERA_NAMES = ("head_camera", "left_camera", "right_camera")


def parse_args_and_config():
    parser = argparse.ArgumentParser(
        description="Run DP eval and generate an interactive policy I/O replay."
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--overrides", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if args.overrides:
        if len(args.overrides) % 2 != 0:
            raise ValueError("--overrides must be key/value pairs")
        for idx in range(0, len(args.overrides), 2):
            key = args.overrides[idx].lstrip("--")
            value = args.overrides[idx + 1]
            try:
                value = eval(value)
            except Exception:
                pass
            config[key] = value

    return config


def eval_function_decorator(policy_name, model_name):
    policy_model = importlib.import_module(policy_name)
    return getattr(policy_model, model_name)


def get_embodiment_file(embodiment_type):
    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")
    with open(embodiment_config_path, "r", encoding="utf-8") as f:
        embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    robot_file = embodiment_types[embodiment_type]["file_path"]
    if robot_file is None:
        raise RuntimeError(f"No embodiment file for {embodiment_type}")
    return robot_file


def load_task_args(usr_args):
    task_config = usr_args["task_config"]
    with open(REPO_ROOT / "task_config" / f"{task_config}.yml", "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    args["task_name"] = usr_args["task_name"]
    args["task_config"] = task_config
    args["ckpt_setting"] = usr_args["ckpt_setting"]
    args["policy_name"] = usr_args["policy_name"]
    args["eval_mode"] = True

    # This script owns frame capture. Leaving eval_video_path enabled would make
    # BaseTask.take_action write to an ffmpeg pipe that we did not create.
    args["eval_video_log"] = False
    args["eval_video_save_dir"] = None

    embodiment_type = args.get("embodiment")
    with open(CONFIGS_PATH + "_camera_config.yml", "r", encoding="utf-8") as f:
        camera_config = yaml.load(f.read(), Loader=yaml.FullLoader)

    head_camera_type = args["camera"]["head_camera_type"]
    args["head_camera_h"] = camera_config[head_camera_type]["h"]
    args["head_camera_w"] = camera_config[head_camera_type]["w"]

    if len(embodiment_type) == 1:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[1])
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False
    else:
        raise ValueError("embodiment items should be 1 or 3")

    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])

    usr_args["left_arm_dim"] = len(args["left_embodiment_config"]["arm_joints_name"][0])
    usr_args["right_arm_dim"] = len(args["right_embodiment_config"]["arm_joints_name"][1])
    return args


def to_uint8_rgb(rgb):
    image = np.asarray(rgb)
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 1 if image.max() <= 1.0 else 255)
        if image.max() <= 1.0:
            image = image * 255
        image = image.astype(np.uint8)
    return image


def rgb_data_url(rgb, quality=82):
    image = to_uint8_rgb(rgb)
    try:
        import cv2

        ok, encoded = cv2.imencode(
            ".jpg",
            cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
            [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)],
        )
        if ok:
            payload = base64.b64encode(encoded.tobytes()).decode("ascii")
            return f"data:image/jpeg;base64,{payload}"
    except Exception:
        pass

    try:
        from PIL import Image

        buffer = BytesIO()
        Image.fromarray(image).save(buffer, format="JPEG", quality=quality)
        payload = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{payload}"
    except Exception as exc:
        raise RuntimeError("Failed to encode replay image as JPEG data URL") from exc



def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def array_summary(value):
    arr = np.asarray(value)
    if arr.size == 0:
        return {"shape": list(arr.shape), "min": None, "max": None, "mean": None}
    return {
        "shape": list(arr.shape),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
    }


def capture_cameras(observation):
    camera_urls = {}
    camera_stats = {}
    camera_rgb = {}
    for camera_name in CAMERA_NAMES:
        rgb = to_uint8_rgb(observation["observation"][camera_name]["rgb"])
        camera_rgb[camera_name] = rgb
        camera_urls[camera_name] = rgb_data_url(rgb)
        camera_stats[camera_name] = array_summary(rgb)
    return camera_urls, camera_stats, camera_rgb


def split_action(action, usr_args):
    action = np.asarray(action)
    left_width = int(usr_args["left_arm_dim"]) + 1
    right_width = int(usr_args["right_arm_dim"]) + 1
    return {
        "left": json_safe(action[:left_width]),
        "right": json_safe(action[left_width:left_width + right_width]),
    }


def build_policy_call_record(call_idx, observation, encoded_obs, actions, usr_args, timing):
    urls, stats, camera_rgb = capture_cameras(observation)
    action_chunk = np.asarray(actions, dtype=np.float32)
    record = {
        "call_index": call_idx,
        "input_camera_paths": urls,
        "input_camera_stats": stats,
        "input_agent_pos": json_safe(encoded_obs["agent_pos"]),
        "input_agent_pos_summary": array_summary(encoded_obs["agent_pos"]),
        "output_action_chunk": json_safe(action_chunk),
        "output_action_summary": array_summary(action_chunk),
        "output_action_split": [split_action(action, usr_args) for action in action_chunk],
        "timing": json_safe(timing),
    }
    arrays = {
        "camera_rgb": camera_rgb,
        "encoded_obs": {
            "head_cam": np.asarray(encoded_obs["head_cam"], dtype=np.float32),
            "left_cam": np.asarray(encoded_obs["left_cam"], dtype=np.float32),
            "right_cam": np.asarray(encoded_obs["right_cam"], dtype=np.float32),
            "agent_pos": np.asarray(encoded_obs["agent_pos"], dtype=np.float32),
        },
        "actions": action_chunk,
        "timing": timing,
    }
    return record, arrays


def build_step_record(
    step_idx,
    policy_call_idx,
    chunk_idx,
    env_step_before,
    observation,
    action,
    usr_args,
):
    urls, stats, camera_rgb = capture_cameras(observation)
    action = np.asarray(action, dtype=np.float32)
    record = {
        "step_index": step_idx,
        "policy_call_index": policy_call_idx,
        "chunk_action_index": chunk_idx,
        "env_step_before": int(env_step_before),
        "current_camera_paths": urls,
        "current_camera_stats": stats,
        "action": json_safe(action),
        "action_summary": array_summary(action),
        "action_split": split_action(action, usr_args),
    }
    arrays = {
        "camera_rgb": camera_rgb,
        "action": action,
        "policy_call_index": int(policy_call_idx),
        "chunk_action_index": int(chunk_idx),
        "env_step_before": int(env_step_before),
    }
    return record, arrays


def find_valid_seed(TASK_ENV, args, start_seed, now_id, max_trials=1000):
    render_freq = args["render_freq"]
    args["render_freq"] = 0
    now_seed = start_seed
    trials = 0
    try:
        while trials < max_trials:
            try:
                TASK_ENV.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)
                episode_info = TASK_ENV.play_once()
                TASK_ENV.close_env()
            except UnStableError:
                TASK_ENV.close_env()
                now_seed += 1
                trials += 1
                continue
            except Exception as exc:
                TASK_ENV.close_env()
                print(f"Seed {now_seed} failed during expert check: {exc}")
                now_seed += 1
                trials += 1
                continue

            if TASK_ENV.plan_success and TASK_ENV.check_success():
                return now_seed, episode_info

            now_seed += 1
            trials += 1
    finally:
        args["render_freq"] = render_freq

    raise RuntimeError(f"No valid expert seed found after {max_trials} trials")


def run_interactive_episode(
    TASK_ENV,
    args,
    usr_args,
    model,
    seed,
    now_id,
    episode_info,
    output_dir,
):
    encode_obs = eval_function_decorator(args["policy_name"], "encode_obs")
    reset_model = eval_function_decorator(args["policy_name"], "reset_model")

    episode_dir = output_dir / f"episode{now_id:03d}"
    episode_dir.mkdir(parents=True, exist_ok=True)

    TASK_ENV.setup_demo(now_ep_num=now_id, seed=seed, is_test=True, **args)
    results = generate_episode_descriptions(args["task_name"], [episode_info["info"]], 1)
    instruction_type = usr_args.get("instruction_type", "unseen")
    instruction = np.random.choice(results[0][instruction_type])
    TASK_ENV.set_instruction(instruction=instruction)

    policy_calls = []
    policy_call_arrays = []
    steps = []
    step_arrays = []
    succ = False
    episode_wall_start = datetime.now().isoformat(timespec="milliseconds")
    episode_t0 = time.perf_counter()
    previous_inference_start_s = None
    previous_inference_end_s = None
    previous_action_exec_start_s = None
    reset_model(model)

    while TASK_ENV.take_action_cnt < TASK_ENV.step_lim:
        observation = TASK_ENV.get_obs()
        encoded_obs = encode_obs(observation)
        inference_start_abs = time.perf_counter()
        inference_start_s = inference_start_abs - episode_t0
        actions = model.get_action(encoded_obs)
        inference_end_abs = time.perf_counter()
        inference_end_s = inference_end_abs - episode_t0
        call_idx = len(policy_calls)
        call_timing = {
            "inference_start_s": inference_start_s,
            "inference_end_s": inference_end_s,
            "inference_duration_s": inference_end_s - inference_start_s,
            "since_prev_inference_start_s": (
                None
                if previous_inference_start_s is None
                else inference_start_s - previous_inference_start_s
            ),
            "since_prev_inference_end_s": (
                None
                if previous_inference_end_s is None
                else inference_end_s - previous_inference_end_s
            ),
        }
        call_record, call_arrays = build_policy_call_record(
            call_idx,
            observation,
            encoded_obs,
            actions,
            usr_args,
            call_timing,
        )
        policy_calls.append(call_record)
        policy_call_arrays.append(call_arrays)
        previous_inference_start_s = inference_start_s
        previous_inference_end_s = inference_end_s

        for chunk_idx, action in enumerate(actions):
            if TASK_ENV.take_action_cnt >= TASK_ENV.step_lim:
                break

            step_idx = len(steps)
            step_record, step_array = build_step_record(
                step_idx,
                call_idx,
                chunk_idx,
                TASK_ENV.take_action_cnt,
                observation,
                action,
                usr_args,
            )

            action_exec_start_s = time.perf_counter() - episode_t0
            TASK_ENV.take_action(action)
            action_exec_end_s = time.perf_counter() - episode_t0
            step_timing = {
                "action_output_s": inference_end_s,
                "action_exec_start_s": action_exec_start_s,
                "action_exec_end_s": action_exec_end_s,
                "action_exec_duration_s": action_exec_end_s - action_exec_start_s,
                "delay_from_chunk_output_s": action_exec_start_s - inference_end_s,
                "since_prev_action_exec_start_s": (
                    None
                    if previous_action_exec_start_s is None
                    else action_exec_start_s - previous_action_exec_start_s
                ),
            }
            step_record["timing"] = json_safe(step_timing)
            step_array["timing"] = step_timing
            steps.append(step_record)
            step_arrays.append(step_array)
            previous_action_exec_start_s = action_exec_start_s

            observation = TASK_ENV.get_obs()
            encoded_obs = encode_obs(observation)
            model.update_obs(encoded_obs)

        if TASK_ENV.eval_success:
            succ = True
            break

    trace = {
        "episode_index": now_id,
        "seed": int(seed),
        "instruction": str(instruction),
        "success": bool(succ),
        "eval_success": bool(getattr(TASK_ENV, "eval_success", False)),
        "stage_eval_score": json_safe(getattr(TASK_ENV, "stage_eval_score", None)),
        "episode_wall_start": episode_wall_start,
        "timing_unit": "seconds_from_episode_start",
        "step_limit": int(TASK_ENV.step_lim),
        "num_steps": len(steps),
        "num_policy_calls": len(policy_calls),
        "camera_names": list(CAMERA_NAMES),
        "left_arm_dim": int(usr_args["left_arm_dim"]),
        "right_arm_dim": int(usr_args["right_arm_dim"]),
    }

    with open(episode_dir / "trace.json", "w", encoding="utf-8") as f:
        json.dump(trace, f, ensure_ascii=False, indent=2)

    if step_arrays:
        def timing_values(items, key):
            return np.asarray(
                [
                    np.nan if item["timing"][key] is None else item["timing"][key]
                    for item in items
                ],
                dtype=np.float64,
            )

        npz_payload = {
            "step_actions": np.stack([item["action"] for item in step_arrays]),
            "step_policy_call_indices": np.asarray(
                [item["policy_call_index"] for item in step_arrays],
                dtype=np.int64,
            ),
            "step_chunk_action_indices": np.asarray(
                [item["chunk_action_index"] for item in step_arrays],
                dtype=np.int64,
            ),
            "step_env_step_before": np.asarray(
                [item["env_step_before"] for item in step_arrays],
                dtype=np.int64,
            ),
            "policy_input_agent_pos": np.stack(
                [item["encoded_obs"]["agent_pos"] for item in policy_call_arrays]
            ),
            "policy_input_head_cam": np.stack(
                [item["encoded_obs"]["head_cam"] for item in policy_call_arrays]
            ),
            "policy_input_left_cam": np.stack(
                [item["encoded_obs"]["left_cam"] for item in policy_call_arrays]
            ),
            "policy_input_right_cam": np.stack(
                [item["encoded_obs"]["right_cam"] for item in policy_call_arrays]
            ),
            "policy_output_action_chunks": np.stack(
                [item["actions"] for item in policy_call_arrays]
            ),
            "policy_inference_start_s": timing_values(
                policy_call_arrays,
                "inference_start_s",
            ),
            "policy_inference_end_s": timing_values(
                policy_call_arrays,
                "inference_end_s",
            ),
            "policy_inference_duration_s": timing_values(
                policy_call_arrays,
                "inference_duration_s",
            ),
            "policy_since_prev_inference_start_s": timing_values(
                policy_call_arrays,
                "since_prev_inference_start_s",
            ),
            "policy_since_prev_inference_end_s": timing_values(
                policy_call_arrays,
                "since_prev_inference_end_s",
            ),
            "step_action_output_s": timing_values(
                step_arrays,
                "action_output_s",
            ),
            "step_action_exec_start_s": timing_values(
                step_arrays,
                "action_exec_start_s",
            ),
            "step_action_exec_end_s": timing_values(
                step_arrays,
                "action_exec_end_s",
            ),
            "step_action_exec_duration_s": timing_values(
                step_arrays,
                "action_exec_duration_s",
            ),
            "step_delay_from_chunk_output_s": timing_values(
                step_arrays,
                "delay_from_chunk_output_s",
            ),
            "step_since_prev_action_exec_start_s": timing_values(
                step_arrays,
                "since_prev_action_exec_start_s",
            ),
            "seeds": np.asarray([seed], dtype=np.int64),
        }
        for camera_name in CAMERA_NAMES:
            npz_payload[f"step_view_{camera_name}_rgb"] = np.stack(
                [item["camera_rgb"][camera_name] for item in step_arrays]
            )
            npz_payload[f"policy_input_{camera_name}_rgb"] = np.stack(
                [item["camera_rgb"][camera_name] for item in policy_call_arrays]
            )

        np.savez_compressed(
            episode_dir / "policy_io_arrays.npz",
            **npz_payload,
        )

    TASK_ENV.close_env(clear_cache=False)
    replay_data = dict(trace)
    replay_data["policy_calls"] = policy_calls
    replay_data["steps"] = steps
    return trace, replay_data


def render_html(trace, episode_dir):
    data = json.dumps(trace, ensure_ascii=False)
    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>DP Interactive Eval Replay</title>
<style>
:root {{
  color-scheme: dark;
  --bg: #0b1020;
  --panel: rgba(17, 24, 39, 0.82);
  --panel-strong: rgba(31, 41, 55, 0.92);
  --text: #e5e7eb;
  --muted: #9ca3af;
  --accent: #60a5fa;
  --accent-2: #a78bfa;
  --ok: #34d399;
  --warn: #fbbf24;
  --line: rgba(148, 163, 184, 0.22);
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0;
  min-height: 100vh;
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  background:
    radial-gradient(circle at top left, rgba(96, 165, 250, 0.24), transparent 34rem),
    radial-gradient(circle at top right, rgba(167, 139, 250, 0.2), transparent 30rem),
    linear-gradient(135deg, #070b16 0%, var(--bg) 48%, #111827 100%);
  color: var(--text);
}}
.shell {{ width: min(1480px, calc(100vw - 40px)); margin: 0 auto; padding: 26px 0 34px; }}
.hero {{ display: flex; justify-content: space-between; gap: 20px; align-items: flex-start; margin-bottom: 18px; }}
.title h1 {{ margin: 0 0 8px; font-size: 30px; letter-spacing: -0.04em; }}
.title p {{ margin: 0; color: var(--muted); max-width: 920px; line-height: 1.5; }}
.pills {{ display: flex; gap: 10px; flex-wrap: wrap; justify-content: flex-end; }}
.pill {{ padding: 8px 12px; border: 1px solid var(--line); border-radius: 999px; background: rgba(15, 23, 42, 0.72); color: var(--muted); }}
.pill strong {{ color: var(--text); }}
.dashboard {{ display: grid; grid-template-columns: minmax(0, 1.18fr) minmax(420px, 0.82fr); gap: 16px; align-items: start; }}
.left-col, .right-col {{ display: grid; gap: 14px; align-items: start; }}
.card.tight {{ align-self: start; }}
.card {{
  border: 1px solid var(--line);
  border-radius: 24px;
  background: var(--panel);
  box-shadow: 0 24px 80px rgba(0, 0, 0, 0.32);
  backdrop-filter: blur(18px);
  overflow: hidden;
}}
.card-h {{ padding: 16px 18px; display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid var(--line); }}
.card-h h2 {{ margin: 0; font-size: 16px; letter-spacing: -0.02em; }}
.card-body {{ padding: 16px; }}
.replay-body {{ display: grid; gap: 12px; }}
.main-img {{ width: 100%; max-height: 470px; aspect-ratio: 16 / 9; object-fit: contain; background: #020617; border-radius: 18px; border: 1px solid var(--line); }}
.thumbs {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin-top: 10px; }}
.thumb {{ border: 1px solid var(--line); border-radius: 16px; overflow: hidden; background: #020617; }}
.thumb img {{ display: block; width: 100%; aspect-ratio: 16 / 10; object-fit: cover; }}
.thumb span {{ display: block; padding: 7px 9px; color: var(--muted); font-size: 12px; }}
.controls {{ display: grid; grid-template-columns: auto 1fr auto; gap: 12px; align-items: center; margin-top: 2px; }}
button {{
  border: 0;
  color: white;
  border-radius: 14px;
  padding: 11px 16px;
  font-weight: 700;
  background: linear-gradient(135deg, var(--accent), var(--accent-2));
  cursor: pointer;
  box-shadow: 0 10px 24px rgba(96, 165, 250, 0.22);
}}
button:disabled {{ opacity: 0.45; cursor: not-allowed; }}
input[type=range] {{ width: 100%; accent-color: var(--accent); }}
.kv {{ display: grid; grid-template-columns: 130px 1fr; gap: 8px 12px; font-size: 13px; }}
.kv div:nth-child(odd) {{ color: var(--muted); }}
.meta-groups {{ display: grid; gap: 10px; }}
.meta-note {{
  padding: 10px 12px;
  border: 1px solid rgba(96, 165, 250, 0.24);
  border-radius: 14px;
  background: rgba(96, 165, 250, 0.08);
  color: #c7d2fe;
  font-size: 12px;
  line-height: 1.55;
}}
.meta-section {{
  border: 1px solid var(--line);
  border-radius: 16px;
  padding: 10px 12px;
  background: rgba(2, 6, 23, 0.36);
}}
.meta-section h3 {{
  margin: 0 0 8px;
  color: #dbeafe;
  font-size: 13px;
  letter-spacing: -0.01em;
}}
.meta-row {{
  display: grid;
  grid-template-columns: minmax(136px, 0.95fr) minmax(90px, 1fr);
  gap: 8px;
  align-items: baseline;
  padding: 4px 0;
  font-size: 12px;
  border-top: 1px solid rgba(148, 163, 184, 0.08);
}}
.meta-row:first-of-type {{ border-top: 0; }}
.meta-row .label {{ color: var(--muted); }}
.meta-row .value {{ color: var(--text); word-break: break-word; }}
.mono {{ font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; }}
.stack {{ display: grid; gap: 14px; }}
.split {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }}
.vector {{
  border: 1px solid var(--line);
  border-radius: 16px;
  padding: 12px;
  background: rgba(2, 6, 23, 0.42);
}}
.vector h3 {{ margin: 0 0 8px; font-size: 13px; color: var(--muted); font-weight: 600; }}
.nums {{ display: flex; flex-wrap: wrap; gap: 6px; }}
.num {{ padding: 5px 7px; border-radius: 9px; background: rgba(15, 23, 42, 0.9); border: 1px solid var(--line); font-size: 12px; }}
.chunk {{ max-height: 260px; overflow: auto; border: 1px solid var(--line); border-radius: 16px; }}
table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
td, th {{ padding: 8px 9px; border-bottom: 1px solid var(--line); text-align: left; }}
tr.active {{ background: rgba(96, 165, 250, 0.18); color: white; }}
.bar-wrap {{ display: grid; gap: 6px; margin-top: 10px; }}
.bar-row {{ display: grid; grid-template-columns: 42px 1fr 64px; gap: 8px; align-items: center; font-size: 12px; color: var(--muted); }}
.bar {{ height: 8px; border-radius: 999px; background: rgba(148, 163, 184, 0.18); overflow: hidden; }}
.bar i {{ display: block; height: 100%; width: 0%; background: linear-gradient(90deg, var(--accent), var(--ok)); }}
@media (max-width: 1050px) {{ .dashboard, .split {{ grid-template-columns: 1fr; }} .hero {{ flex-direction: column; }} }}
</style>
</head>
<body>
<main class="shell">
  <section class="hero">
    <div class="title">
      <h1>DP Interactive Eval Replay</h1>
      <p id="subtitle"></p>
    </div>
    <div class="pills">
      <div class="pill">Seed <strong id="seed"></strong></div>
      <div class="pill">Steps <strong id="steps"></strong></div>
      <div class="pill">Policy calls <strong id="calls"></strong></div>
      <div class="pill">Success <strong id="success"></strong></div>
    </div>
  </section>
  <section class="dashboard">
    <div class="left-col">
      <div class="card tight">
        <div class="card-h">
          <h2>Current Replay View</h2>
          <span class="mono" id="stepLabel"></span>
        </div>
        <div class="card-body replay-body">
          <img class="main-img" id="mainView" src="" alt="current head camera">
          <div class="controls">
            <button id="prevBtn">Prev</button>
            <input id="slider" type="range" min="0" value="0">
            <button id="nextBtn">Next</button>
          </div>
        </div>
      </div>
      <div class="card">
        <div class="card-h">
          <h2>Policy Input For This Action Chunk</h2>
          <span class="mono" id="callLabel"></span>
        </div>
        <div class="card-body">
          <div class="thumbs" id="inputThumbs"></div>
          <div class="vector" style="margin-top:12px;"><h3>agent_pos</h3><div class="nums" id="agentPos"></div></div>
        </div>
      </div>
      <div class="card">
        <div class="card-h"><h2>Policy Output Action Chunk</h2></div>
        <div class="card-body">
          <div class="chunk"><table id="chunkTable"></table></div>
        </div>
      </div>
    </div>
    <div class="right-col">
      <div class="card">
        <div class="card-h">
          <h2>Step Metadata</h2>
          <span class="mono">字段说明</span>
        </div>
        <div class="card-body"><div class="meta-groups" id="meta"></div></div>
      </div>
      <div class="card">
        <div class="card-h"><h2>Current Action Output</h2></div>
        <div class="card-body">
          <div class="split">
            <div class="vector"><h3>Left arm + gripper</h3><div class="nums" id="leftAction"></div></div>
            <div class="vector"><h3>Right arm + gripper</h3><div class="nums" id="rightAction"></div></div>
          </div>
          <div class="bar-wrap" id="actionBars"></div>
        </div>
      </div>
    </div>
  </section>
</main>
<script>
const DATA = {data};
let index = 0;
const fmt = (x) => String(x);
const fmtSec = (x) => x === null || x === undefined || Number.isNaN(Number(x)) ? 'n/a' : `${{Number(x).toFixed(4)}} s`;
const camTitle = (name) => name.replace('_camera', '').replace(/^./, c => c.toUpperCase());
function imgThumbs(container, paths) {{
  container.innerHTML = DATA.camera_names.map(name => `
    <div class="thumb"><img src="${{paths[name]}}" alt="${{name}}"><span>${{camTitle(name)}} camera</span></div>
  `).join('');
}}
function nums(container, values, limit=64) {{
  const arr = (values || []).slice(0, limit);
  container.innerHTML = arr.map(v => `<span class="num mono">${{fmt(v)}}</span>`).join('');
}}
function bars(container, values) {{
  const maxAbs = Math.max(1e-6, ...values.map(v => Math.abs(v)));
  container.innerHTML = values.map((v, i) => `
    <div class="bar-row"><span>a${{i}}</span><div class="bar"><i style="width:${{Math.abs(v) / maxAbs * 100}}%"></i></div><span class="mono">${{fmt(v)}}</span></div>
  `).join('');
}}
function chunkTable(container, chunk, activeIdx) {{
  const rows = chunk.map((row, r) => `<tr class="${{r === activeIdx ? 'active' : ''}}">
    <th class="mono">#${{r}}</th><td class="mono">${{row.map(fmt).join(' ')}}</td>
  </tr>`).join('');
  container.innerHTML = `<tbody>${{rows}}</tbody>`;
}}
function kv(container, pairs) {{
  container.innerHTML = pairs.map(([k, v]) => `<div>${{k}}</div><div class="mono">${{v}}</div>`).join('');
}}
function metaGroups(container, groups) {{
  const note = `<div class="meta-note">时间均为相对当前 episode 开始的秒数；推理时间描述 policy 生成 action chunk，执行时间描述单个 action 被送入环境并完成。</div>`;
  const body = groups.map(group => `
    <section class="meta-section">
      <h3>${{group.title}}</h3>
      ${{group.rows.map(([label, value]) => `
        <div class="meta-row">
          <span class="label">${{label}}</span>
          <span class="value mono">${{value}}</span>
        </div>
      `).join('')}}
    </section>
  `).join('');
  container.innerHTML = note + body;
}}
function render() {{
  const step = DATA.steps[index];
  const call = DATA.policy_calls[step.policy_call_index];
  document.getElementById('mainView').src = step.current_camera_paths.head_camera;
  document.getElementById('stepLabel').textContent = `step ${{index + 1}} / ${{DATA.steps.length}}`;
  document.getElementById('callLabel').textContent = `policy call #${{call.call_index}}, action #${{step.chunk_action_index}}, infer @ ${{fmtSec(call.timing.inference_end_s)}}`;
  imgThumbs(document.getElementById('inputThumbs'), call.input_camera_paths);
  nums(document.getElementById('leftAction'), step.action_split.left);
  nums(document.getElementById('rightAction'), step.action_split.right);
  nums(document.getElementById('agentPos'), call.input_agent_pos);
  bars(document.getElementById('actionBars'), step.action);
  chunkTable(document.getElementById('chunkTable'), call.output_action_chunk, step.chunk_action_index);
  metaGroups(document.getElementById('meta'), [
    {{
      title: '步骤定位',
      rows: [
        ['环境步数 env_step_before', step.env_step_before],
        ['Policy 调用序号', step.policy_call_index],
        ['Chunk 内 action 序号', step.chunk_action_index],
        ['Action 维度', step.action.length],
      ],
    }},
    {{
      title: 'Policy 推理时间',
      rows: [
        ['Chunk 推理开始', fmtSec(call.timing.inference_start_s)],
        ['Chunk 推理结束', fmtSec(call.timing.inference_end_s)],
        ['本次推理耗时', fmtSec(call.timing.inference_duration_s)],
        ['距上次推理开始', fmtSec(call.timing.since_prev_inference_start_s)],
      ],
    }},
    {{
      title: 'Action 执行时间',
      rows: [
        ['Action 输出时间', fmtSec(step.timing.action_output_s)],
        ['开始执行', fmtSec(step.timing.action_exec_start_s)],
        ['执行完成', fmtSec(step.timing.action_exec_end_s)],
        ['执行耗时', fmtSec(step.timing.action_exec_duration_s)],
        ['Chunk 输出到执行', fmtSec(step.timing.delay_from_chunk_output_s)],
        ['距上次 action 开始', fmtSec(step.timing.since_prev_action_exec_start_s)],
      ],
    }},
    {{
      title: '任务评估',
      rows: [
        ['阶段得分 stage_eval_score', JSON.stringify(DATA.stage_eval_score)],
        ['语言指令 instruction', DATA.instruction],
      ],
    }},
  ]);
  document.getElementById('slider').value = index;
  document.getElementById('prevBtn').disabled = index <= 0;
  document.getElementById('nextBtn').disabled = index >= DATA.steps.length - 1;
}}
document.getElementById('subtitle').textContent = DATA.instruction;
document.getElementById('seed').textContent = DATA.seed;
document.getElementById('steps').textContent = DATA.num_steps;
document.getElementById('calls').textContent = DATA.num_policy_calls;
document.getElementById('success').textContent = DATA.success ? 'yes' : 'no';
document.getElementById('slider').max = Math.max(0, DATA.steps.length - 1);
document.getElementById('prevBtn').onclick = () => {{ index = Math.max(0, index - 1); render(); }};
document.getElementById('nextBtn').onclick = () => {{ index = Math.min(DATA.steps.length - 1, index + 1); render(); }};
document.getElementById('slider').oninput = (e) => {{ index = Number(e.target.value); render(); }};
document.addEventListener('keydown', (e) => {{
  if (e.key === 'ArrowRight' || e.key === ' ') {{ index = Math.min(DATA.steps.length - 1, index + 1); render(); }}
  if (e.key === 'ArrowLeft') {{ index = Math.max(0, index - 1); render(); }}
}});
render();
</script>
</body>
</html>
"""
    with open(episode_dir / "index.html", "w", encoding="utf-8") as f:
        f.write(html)


def main(usr_args):
    os.chdir(REPO_ROOT)
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    args = load_task_args(usr_args)

    policy_name = usr_args["policy_name"]
    get_model = eval_function_decorator(policy_name, "get_model")
    TASK_ENV = class_decorator(args["task_name"])

    output_root = usr_args.get("interactive_output_dir")
    if output_root:
        output_dir = Path(output_root)
    else:
        output_dir = Path(
            f"eval_result/{args['task_name']}/{policy_name}/{args['task_config']}/"
            f"{args['ckpt_setting']}/{current_time}_interactive"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    episodes = int(usr_args.get("interactive_episodes", 1))
    max_seed_trials = int(usr_args.get("interactive_max_seed_trials", 1000))
    seed = int(usr_args["seed"])
    now_seed = 100000 * (1 + seed)
    model = get_model(usr_args)
    traces = []

    print(f"Interactive eval output: {output_dir}")
    print(f"Task={args['task_name']} Policy={policy_name} Episodes={episodes}")

    for episode_idx in range(episodes):
        valid_seed, episode_info = find_valid_seed(
            TASK_ENV,
            args,
            now_seed,
            episode_idx,
            max_trials=max_seed_trials,
        )
        print(f"\nEpisode {episode_idx}: using seed {valid_seed}")
        trace, replay_data = run_interactive_episode(
            TASK_ENV,
            args,
            usr_args,
            model,
            valid_seed,
            episode_idx,
            episode_info,
            output_dir,
        )
        traces.append(trace)
        render_html(replay_data, output_dir / f"episode{episode_idx:03d}")
        now_seed = valid_seed + 1
        print(
            f"Episode {episode_idx} saved: "
            f"{output_dir / f'episode{episode_idx:03d}' / 'index.html'}"
        )

    summary = {
        "created_at": current_time,
        "task_name": args["task_name"],
        "task_config": args["task_config"],
        "ckpt_setting": args["ckpt_setting"],
        "policy_name": policy_name,
        "episodes": [
            {
                "episode_index": trace["episode_index"],
                "seed": trace["seed"],
                "success": trace["success"],
                "num_steps": trace["num_steps"],
                "num_policy_calls": trace["num_policy_calls"],
                "html": f"episode{trace['episode_index']:03d}/index.html",
            }
            for trace in traces
        ],
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\nSummary saved: {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main(parse_args_and_config())
