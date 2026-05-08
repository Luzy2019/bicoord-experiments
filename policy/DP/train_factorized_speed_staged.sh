#!/bin/bash
set -e

# Train factorized DP first, then train frozen factorized DP + learned speed heads.
#
# Usage:
#   bash train_factorized_speed_staged.sh <task_name> <task_config> <expert_data_num> <seed> <action_dim> <gpu_id> [batch_size] [aux_weight] [gate_bias] [speed_loss_weight] [base_ckpt_num|auto] [run_base]
#
# Example:
#   bash train_factorized_speed_staged.sh stack_bowls demo_clean 50 100 14 0 96 0.25 -2.0 1e-2 auto true
#
# If you already have a factorized base checkpoint:
#   bash train_factorized_speed_staged.sh stack_bowls demo_clean 50 100 14 0 96 0.25 -2.0 1e-2 400 false

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
run_base=${12:-true}

if [ -z "${gpu_id}" ]; then
    echo "Usage: bash train_factorized_speed_staged.sh <task_name> <task_config> <expert_data_num> <seed> <action_dim> <gpu_id> [batch_size] [aux_weight] [gate_bias] [speed_loss_weight] [base_ckpt_num|auto] [run_base]"
    exit 1
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${script_dir}"

ckpt_dir="checkpoints/${task_name}-${task_config}-${expert_data_num}-${seed}"
base_backup_dir="checkpoints/${task_name}-${task_config}-${expert_data_num}-${seed}-factorized_base"

echo "============================================================"
echo "Stage 1: factorized base"
echo "============================================================"
if [ "${run_base}" = "true" ]; then
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
        base
else
    echo "Skip base training; will reuse existing checkpoint from ${ckpt_dir}"
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

mkdir -p "${base_backup_dir}"
base_ckpt_name="$(basename "${base_ckpt_path}")"
base_backup_path="${base_backup_dir}/${base_ckpt_name}"
cp "${base_ckpt_path}" "${base_backup_path}"

echo "Base checkpoint: ${base_ckpt_path}"
echo "Base checkpoint backup: ${base_backup_path}"

echo "Verifying base checkpoint is factorized..."
/home/lzy/anaconda3/envs/RoboTwin/bin/python - <<PY
import dill
import torch
path = "${base_backup_path}"
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
    "${script_dir}/${base_backup_path}"

echo "Done."
echo "Speed checkpoints are saved under: ${script_dir}/${ckpt_dir}"
echo "Base backup is preserved under: ${script_dir}/${base_backup_dir}"
