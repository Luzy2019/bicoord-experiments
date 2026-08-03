#!/bin/bash
set -euo pipefail

# Usage:
#   bash train_scan_object.sh [all|teacher|student] [gpu_id] [seed]
#
# Defaults match policy/DP/train_script.md:
#   task=scan_object, setting=aloha-agilex_clean_50, expert_data_num=50,
#   seed=0, action_dim=14, gpu_id=0.
#
# Environment overrides:
#   TASK_CONFIG=aloha-agilex_clean_50 EXPERT_DATA_NUM=50 ACTION_DIM=14
#   TEACHER_EPOCH=600 TEACHER_CKPT_PATH=checkpoints/.../600.ckpt
#   FM_MR_BATCH_SIZE=8 FM_MR_GRAD_ACCUM=2 FM_MR_SAMPLE_EVERY=0 WANDB_MODE=offline
#   FM_MR_NUM_EPOCHS=100 FM_MR_VAL_EVERY=5 FM_MR_CHECKPOINT_EVERY=10
#   PYTHON_BIN=/path/to/python

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${script_dir}"

stage=${1:-all}
gpu_id=${2:-${GPU_ID:-0}}
seed=${3:-${SEED:-0}}

task_name=scan_object
task_config=${TASK_CONFIG:-aloha-agilex_clean_50}
expert_data_num=${EXPERT_DATA_NUM:-50}
action_dim=${ACTION_DIM:-14}
teacher_epoch=${TEACHER_EPOCH:-600}
teacher_ckpt_path=${TEACHER_CKPT_PATH:-checkpoints/${task_name}-${task_config}-${expert_data_num}-${seed}-teacher/${teacher_epoch}.ckpt}

print_config() {
    echo -e "\033[33mtask: ${task_name}\033[0m"
    echo -e "\033[33msetting: ${task_config}, expert_data_num: ${expert_data_num}\033[0m"
    echo -e "\033[33mseed: ${seed}, action_dim: ${action_dim}, gpu: ${gpu_id}\033[0m"
    echo -e "\033[33mteacher checkpoint: ${teacher_ckpt_path}\033[0m"
}

run_teacher() {
    bash train_mr.sh "${task_name}" "${task_config}" "${expert_data_num}" "${seed}" "${action_dim}" "${gpu_id}" teacher
}

run_student() {
    if [ ! -f "${teacher_ckpt_path}" ]; then
        echo "Teacher checkpoint not found: ${teacher_ckpt_path}"
        echo "Run teacher first, or set TEACHER_CKPT_PATH to an existing checkpoint."
        exit 1
    fi
    bash train_mr.sh "${task_name}" "${task_config}" "${expert_data_num}" "${seed}" "${action_dim}" "${gpu_id}" student "${teacher_ckpt_path}"
}

print_config

case "${stage}" in
    all)
        run_teacher
        run_student
        ;;
    teacher)
        run_teacher
        ;;
    student)
        run_student
        ;;
    *)
        echo "Usage: bash train_scan_object.sh [all|teacher|student] [gpu_id] [seed]"
        exit 1
        ;;
esac
