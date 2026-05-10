#!/bin/bash
set -e

# Stage 2: initialize from a factorized base checkpoint, freeze base policy,
# and train only the SpeedHead modules.
# Edit diffusion_policy/config/robot_dp_*.yaml before running.
#
# Usage:
#   bash train_factorized_speed_stage2.sh [config_name]
#
# Example:
#   bash train_factorized_speed_stage2.sh robot_dp_14

stage_name=speed_stage2
config_name=${1:-robot_dp_14}
config_name=${config_name%.yaml}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${script_dir}"

config_path="diffusion_policy/config/${config_name}.yaml"
if [ ! -f "${config_path}" ]; then
    echo "Config not found: ${config_path}"
    exit 1
fi

eval "$(python - "${config_path}" "${stage_name}" <<'PY'
import shlex
import sys
from omegaconf import OmegaConf

cfg = OmegaConf.load(sys.argv[1])
stage_name = sys.argv[2]

def get(path, default=None):
    return OmegaConf.select(cfg, path, default=default)

values = {
    "TASK_NAME": get("task.name"),
    "TASK_CONFIG": get("setting"),
    "EXPERT_DATA_NUM": get("expert_data_num"),
    "SEED": get("training.seed"),
    "GPU_ID": get("runtime.gpu_id", 0),
    "BATCH_SIZE": get("dataloader.batch_size"),
    "NUM_EPOCHS": get(f"factorized_stages.{stage_name}.training.num_epochs"),
    "CHECKPOINT_EVERY": get(f"factorized_stages.{stage_name}.training.checkpoint_every"),
    "ZARR_PATH": get("task.dataset.zarr_path"),
    "BASE_CKPT_PATH": get(f"factorized_stages.{stage_name}.training.init_ckpt_path"),
    "SPEED_LOSS_WEIGHT": get(f"factorized_stages.{stage_name}.policy.speed_modulation_loss_weight"),
    "SPEED_CKPT_DIR": get(f"factorized_stages.{stage_name}.training.checkpoint_dir_name"),
    "WANDB_MODE": get("logging.mode"),
    "STAGE_EXP_NAME": get(f"factorized_stages.{stage_name}.exp_name"),
}

missing = [key for key, value in values.items() if value in (None, "")]
if missing:
    raise SystemExit(f"Missing required config values: {', '.join(missing)}")

for key, value in values.items():
    print(f"{key}={shlex.quote(str(value))}")
PY
)"

mapfile -t train_overrides < <(python - "${config_path}" "${stage_name}" <<'PY'
import sys
from omegaconf import OmegaConf

cfg = OmegaConf.load(sys.argv[1])
stage_name = sys.argv[2]

def get(path, default=None):
    return OmegaConf.select(cfg, path, default=default)

def fmt(value):
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "True" if value else "False"
    return str(value)

stage = f"factorized_stages.{stage_name}"
overrides = {
    "exp_name": get(f"{stage}.exp_name"),
    "training.resume": get(f"{stage}.training.resume"),
    "training.init_ckpt_path": get(f"{stage}.training.init_ckpt_path"),
    "training.num_epochs": get(f"{stage}.training.num_epochs"),
    "training.checkpoint_every": get(f"{stage}.training.checkpoint_every"),
    "training.checkpoint_dir_name": get(f"{stage}.training.checkpoint_dir_name"),
    "policy.speed_modulation_enabled": get(f"{stage}.policy.speed_modulation_enabled"),
    "policy.speed_modulation_learned": get(f"{stage}.policy.speed_modulation_learned"),
    "policy.speed_modulation_train_only": get(f"{stage}.policy.speed_modulation_train_only"),
    "policy.speed_modulation_loss_weight": get(f"{stage}.policy.speed_modulation_loss_weight"),
}

missing = [key for key, value in overrides.items() if value == ""]
if missing:
    raise SystemExit(f"Missing stage config values: {', '.join(missing)}")

for key, value in overrides.items():
    print(f"{key}={fmt(value)}")
PY
)

echo -e "\033[33mStage 2: freeze factorized base, train SpeedHead only\033[0m"
echo -e "\033[33mconfig: ${config_name}\033[0m"
echo -e "\033[33mexp name: ${STAGE_EXP_NAME}\033[0m"
echo -e "\033[33mtask: ${TASK_NAME} (${TASK_CONFIG}), expert data: ${EXPERT_DATA_NUM}, seed: ${SEED}\033[0m"
echo -e "\033[33mbase checkpoint: ${BASE_CKPT_PATH}\033[0m"
echo -e "\033[33mgpu id (to use): ${GPU_ID}\033[0m"
echo -e "\033[33mbatch size: ${BATCH_SIZE}\033[0m"
echo -e "\033[33mspeed modulation loss weight: ${SPEED_LOSS_WEIGHT}\033[0m"
echo -e "\033[33mnum epochs: ${NUM_EPOCHS}, checkpoint every: ${CHECKPOINT_EVERY}\033[0m"
echo -e "\033[33mspeed checkpoint dir: checkpoints/${SPEED_CKPT_DIR}\033[0m"
echo -e "\033[33mwandb mode: ${WANDB_MODE}\033[0m"

if [ ! -f "${BASE_CKPT_PATH}" ]; then
    echo "Base checkpoint not found: ${BASE_CKPT_PATH}"
    exit 1
fi

export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=${GPU_ID}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if [ ! -d "${ZARR_PATH}" ]; then
    bash process_data.sh "${TASK_NAME}" "${TASK_CONFIG}" "${EXPERT_DATA_NUM}"
elif ! python - "${ZARR_PATH}" <<'PY'
import sys
import zarr

root = zarr.open(sys.argv[1], mode="r")
data = root["data"]
required_keys = {"head_camera", "left_camera", "right_camera", "state", "action"}
missing_keys = sorted(required_keys - set(data.keys()))
if missing_keys:
    print("Missing zarr keys:", ", ".join(missing_keys))
    raise SystemExit(1)
PY
then
    echo "Existing zarr is missing wrist camera data; rebuilding ${ZARR_PATH}"
    rm -rf "${ZARR_PATH}"
    bash process_data.sh "${TASK_NAME}" "${TASK_CONFIG}" "${EXPERT_DATA_NUM}"
fi

echo "Verifying base checkpoint is factorized..."
python - "${BASE_CKPT_PATH}" <<'PY'
import sys
import dill
import torch

path = sys.argv[1]
payload = torch.load(open(path, "rb"), pickle_module=dill, map_location="cpu")
target = payload["cfg"].policy._target_
print("checkpoint target:", target)
if "factorized_bimanual_diffusion_unet_image_policy" not in target:
    raise SystemExit(f"Expected factorized checkpoint, got {target}")
PY

python train.py --config-name="${config_name}.yaml" "${train_overrides[@]}"
