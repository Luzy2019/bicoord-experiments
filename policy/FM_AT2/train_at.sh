#!/bin/bash
set -e

# Train the action-only coupling-scheduled Flow Matching policy with KL coupling.
#
# Usage:
#   bash train_at.sh <task> <task_config> <expert_data_num> <seed> <gpu_id> [rerank_samples] [stage] [init_ckpt_path]
#
# Example:
#   bash train_at.sh stack_bowls demo_clean 50 100 0 1

task_name=${1}
task_config=${2}
expert_data_num=${3}
seed=${4}
gpu_id=${5}
rerank_samples=${6:-1}
train_stage=${7:-bc}
init_ckpt_path=${8:-}

config_name=${FM_AT_CONFIG_NAME:-robot_fm_at2_16}
head_camera_type=${DP_HEAD_CAMERA_TYPE:-D435}
wandb_mode=${WANDB_MODE:-online}
debug=${DP_DEBUG:-False}
python_bin=${PYTHON_BIN:-python3}

if [ -z "${gpu_id}" ]; then
    echo "Usage: bash train_at.sh <task> <task_config> <expert_data_num> <seed> <gpu_id> [rerank_samples]"
    exit 1
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${script_dir}"

zarr_path="data/${task_name}-${task_config}-${expert_data_num}.zarr"
exp_name="${task_name}-coupling_scheduled_fm_at2_kl"

export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=${gpu_id}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if [ ! -d "${zarr_path}" ]; then
    bash process_data.sh "${task_name}" "${task_config}" "${expert_data_num}"
fi

echo -e "\033[33mFM_AT2 config: ${config_name}\033[0m"
echo -e "\033[33mtask: ${task_name} (${task_config}), expert data: ${expert_data_num}, seed: ${seed}\033[0m"
echo -e "\033[33mgpu id: ${gpu_id}, rerank samples: ${rerank_samples}, stage: ${train_stage}\033[0m"

stage_overrides=(
    "training.stage=${train_stage}"
)
if [ "${train_stage}" = "rl" ] || [ "${train_stage}" = "joint" ]; then
    stage_overrides+=("rl.enabled=True")
    if [ -n "${init_ckpt_path}" ]; then
        stage_overrides+=("training.resume=False")
        stage_overrides+=("training.init_ckpt_path=${init_ckpt_path}")
    fi
else
    stage_overrides+=("rl.enabled=False")
fi

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
    policy.action_rerank_samples="${rerank_samples}" \
    "${stage_overrides[@]}"
