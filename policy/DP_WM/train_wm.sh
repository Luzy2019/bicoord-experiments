#!/bin/bash
set -e

# Usage:
#   bash train_wm.sh <task> <task_config> <expert_data_num> <seed> <gpu_id> [base|speed|joint] [init_ckpt_path]
#
# Example:
#   bash train_wm.sh stack_bowls demo_clean 50 100 0 base
#   bash train_wm.sh stack_bowls demo_clean 50 100 0 speed checkpoints/stack_bowls-demo_clean-50-100/600.ckpt

task_name=${1}
task_config=${2}
expert_data_num=${3}
seed=${4}
gpu_id=${5}
train_stage=${6:-base}
init_ckpt_path=${7:-}

config_name=${DP_WM_CONFIG_NAME:-robot_dp_wm_16}
head_camera_type=${DP_HEAD_CAMERA_TYPE:-D435}
wandb_mode=${WANDB_MODE:-online}
debug=${DP_DEBUG:-False}
python_bin=${PYTHON_BIN:-python3}

if [ -z "${gpu_id}" ]; then
    echo "Usage: bash train_wm.sh <task> <task_config> <expert_data_num> <seed> <gpu_id> [base|speed|joint] [init_ckpt_path]"
    exit 1
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${script_dir}"

zarr_path="data/${task_name}-${task_config}-${expert_data_num}.zarr"
exp_name="${task_name}-asym_wm_flow_${train_stage}"

export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=${gpu_id}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if [ ! -d "${zarr_path}" ]; then
    bash process_data.sh "${task_name}" "${task_config}" "${expert_data_num}"
fi

stage_overrides=()
if [ "${train_stage}" = "base" ]; then
    stage_overrides+=("policy.speed_modulation_enabled=False")
    stage_overrides+=("policy.speed_modulation_train_only=False")
    stage_overrides+=("policy.speed_modulation_loss_weight=0.0")
    stage_overrides+=("policy.energy_rerank_samples=1")
elif [ "${train_stage}" = "speed" ]; then
    if [ -z "${init_ckpt_path}" ]; then
        echo "Speed stage requires init_ckpt_path."
        exit 1
    fi
    stage_overrides+=("training.resume=False")
    stage_overrides+=("training.init_ckpt_path=${init_ckpt_path}")
    stage_overrides+=("policy.speed_modulation_enabled=True")
    stage_overrides+=("policy.speed_modulation_learned=True")
    stage_overrides+=("policy.speed_modulation_train_only=True")
    stage_overrides+=("policy.speed_modulation_loss_weight=1.0e-2")
    stage_overrides+=("policy.world_model_loss_weight=0.0")
    stage_overrides+=("policy.energy_loss_weight=0.0")
    stage_overrides+=("policy.orthogonal_loss_weight=0.0")
    stage_overrides+=("policy.gate_sparsity_loss_weight=0.0")
    stage_overrides+=("policy.residual_balance_loss_weight=0.0")
elif [ "${train_stage}" = "joint" ]; then
    if [ -n "${init_ckpt_path}" ]; then
        stage_overrides+=("training.resume=False")
        stage_overrides+=("training.init_ckpt_path=${init_ckpt_path}")
    fi
    stage_overrides+=("policy.speed_modulation_enabled=True")
    stage_overrides+=("policy.speed_modulation_learned=True")
    stage_overrides+=("policy.speed_modulation_train_only=False")
    stage_overrides+=("policy.speed_modulation_loss_weight=1.0e-2")
else
    echo "Unknown train_stage=${train_stage}; expected base, speed, or joint"
    exit 1
fi

echo -e "\033[33mDP_WM config: ${config_name}, stage: ${train_stage}\033[0m"
echo -e "\033[33mtask: ${task_name} (${task_config}), expert data: ${expert_data_num}, seed: ${seed}\033[0m"
echo -e "\033[33mgpu id: ${gpu_id}, wandb mode: ${wandb_mode}\033[0m"

"${python_bin}" train.py --config-name="${config_name}.yaml" \
    task.name="${task_name}" \
    task.dataset.zarr_path="${zarr_path}" \
    training.debug="${debug}" \
    training.seed="${seed}" \
    training.device="cuda:0" \
    exp_name="${exp_name}" \
    logging.mode="${wandb_mode}" \
    setting="${task_config}" \
    expert_data_num="${expert_data_num}" \
    head_camera_type="${head_camera_type}" \
    "${stage_overrides[@]}"
