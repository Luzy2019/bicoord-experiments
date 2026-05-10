#!/bin/bash
set -e

# Train factorized DP first, then train frozen factorized DP + learned speed heads.
#
# Usage:
#   bash train_factorized_speed_staged.sh <task_name> <task_config> <expert_data_num> <seed> <action_dim> <gpu_id> [batch_size] [aux_weight] [gate_bias] [speed_loss_weight] [base_ckpt_num|auto] [stage] [every_save_epoch] [total_train_epoch]
#
# Example:
#   # Train base only, save every 5 epochs, total 100 epochs. 只训练base，每5个epoch保存一次，总共训练100个epoch
#   bash train_factorized_speed_staged.sh stack_bowls demo_clean 50 100 14 0 96 0.25 -2.0 1e-2 auto base 50 600
#
#   # Train speed only from an existing factorized base checkpoint. 从已有的factorized base checkpoint开始训练speed
#   bash train_factorized_speed_staged.sh stack_bowls demo_clean 50 100 14 0 96 0.25 -2.0 1e-2 400 speed 5 100

task_name=${1}
task_config=${2}
expert_data_num=${3}
seed=${4}
action_dim=${5}
gpu_id=${6}

batch_size=${7:-24}
factorized_aux_loss_weight=${8:-0.25}
factorized_gate_init_bias=${9:--2.0}
speed_modulation_loss_weight=${10:-1e-2}
base_ckpt_num=${11:-auto}
stage=${12:-all}
every_save_epoch=${13:-}
total_train_epoch=${14:-}

if [ -z "${gpu_id}" ]; then
    echo "Usage: bash train_factorized_speed_staged.sh <task_name> <task_config> <expert_data_num> <seed> <action_dim> <gpu_id> [batch_size] [aux_weight] [gate_bias] [speed_loss_weight] [base_ckpt_num|auto] [stage: base|speed|all] [every_save_epoch] [total_train_epoch]"
    exit 1
fi

if [ "${stage}" = "true" ] || [ "${stage}" = "false" ]; then
    echo "Boolean stage flags are no longer supported. Use stage=base, speed, or all."
    exit 1
fi

if [ "${stage}" != "base" ] && [ "${stage}" != "speed" ] && [ "${stage}" != "all" ]; then
    echo "Unknown stage=${stage}; expected base, speed, or all"
    exit 1
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${script_dir}"

ckpt_dir="checkpoints/${task_name}-${task_config}-${expert_data_num}-${seed}"
ckpt_name="${task_name}-${task_config}-${expert_data_num}-${seed}"
speed_ckpt_dir="checkpoints/${ckpt_name}-speed"

echo "============================================================"
echo "Stage 1: factorized base"
echo "============================================================"
if [ "${stage}" = "base" ] || [ "${stage}" = "all" ]; then
    bash train_factorized.sh \
        "${task_name}" \
        "${task_config}" \
        "${expert_data_num}" \
        "${seed}" \
        "${action_dim}" \
        "${gpu_id}" \
        "${factorized_aux_loss_weight}" \
        "${factorized_gate_init_bias}" \
        "${batch_size}" \
        base \
        "${speed_modulation_loss_weight}" \
        "" \
        "${every_save_epoch}" \
        "${total_train_epoch}"
else
    echo "Skip base training; will reuse existing checkpoint from ${ckpt_dir}"
fi

if [ "${stage}" = "base" ]; then
    echo "Done."
    echo "Base checkpoints are saved under: ${script_dir}/${ckpt_dir}"
    exit 0
fi

if [ ! -d "${ckpt_dir}" ]; then
    echo "Base checkpoint directory not found: ${ckpt_dir}"
    exit 1
fi

if [ "${base_ckpt_num}" = "auto" ]; then
    base_ckpt_path="$(ls "${ckpt_dir}"/*.ckpt | sort -V | tail -n 1)"
else
    base_ckpt_path="${ckpt_dir}/${base_ckpt_num}.ckpt"
fi

if [ ! -f "${base_ckpt_path}" ]; then
    echo "Base checkpoint not found: ${base_ckpt_path}"
    exit 1
fi

echo "Base checkpoint: ${base_ckpt_path}"

echo "Verifying base checkpoint is factorized..."
/home/lzy/anaconda3/envs/RoboTwin/bin/python - <<PY
import dill
import torch
path = "${base_ckpt_path}"
payload = torch.load(open(path, "rb"), pickle_module=dill, map_location="cpu")
target = payload["cfg"].policy._target_
print("checkpoint target:", target)
if "factorized_bimanual_diffusion_unet_image_policy" not in target:
    raise SystemExit(f"Expected factorized checkpoint, got {target}")
PY

echo "============================================================"
echo "Stage 2: frozen factorized + speed"
echo "============================================================"
bash train_factorized.sh \
    "${task_name}" \
    "${task_config}" \
    "${expert_data_num}" \
    "${seed}" \
    "${action_dim}" \
    "${gpu_id}" \
    "${factorized_aux_loss_weight}" \
    "${factorized_gate_init_bias}" \
    "${batch_size}" \
    speed \
    "${speed_modulation_loss_weight}" \
    "${script_dir}/${base_ckpt_path}" \
    "${every_save_epoch}" \
    "${total_train_epoch}" \
    "${ckpt_name}-speed"

echo "Done."
echo "Base checkpoints are saved under: ${script_dir}/${ckpt_dir}"
echo "Speed checkpoints are saved under: ${script_dir}/${speed_ckpt_dir}"
