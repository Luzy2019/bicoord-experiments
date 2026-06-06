#!/bin/bash
set -e

# Usage:
#   bash eval_interactive.sh <task_name> <task_config> <ckpt_setting> <expert_data_num> <seed> <gpu_id> [checkpoint_num] [episodes] [output_dir]
#
# Example:
#   bash policy/DP/eval_interactive.sh clean_table demo_clean demo_clean 50 0 0 600 1

policy_name=DP_AT
task_name=${1}
task_config=${2}
ckpt_setting=${3}
expert_data_num=${4}
seed=${5}
gpu_id=${6}
checkpoint_num=${7:-600}
interactive_episodes=${8:-1}
interactive_output_dir=${9:-}

if [ -z "${gpu_id}" ]; then
    echo "Usage: bash eval_interactive.sh <task_name> <task_config> <ckpt_setting> <expert_data_num> <seed> <gpu_id> [checkpoint_num] [episodes] [output_dir]"
    exit 1
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
cd "${repo_root}"

export CUDA_VISIBLE_DEVICES=${gpu_id}
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"
echo -e "\033[33minteractive episodes: ${interactive_episodes}, mode: full episode\033[0m"

if [ -n "${PYTHON}" ]; then
    python_cmd=${PYTHON}
elif command -v python >/dev/null 2>&1; then
    python_cmd=python
else
    python_cmd=python3
fi

PYTHONWARNINGS=ignore::UserWarning \
${python_cmd} policy/${policy_name}/interactive_eval.py --config policy/${policy_name}/deploy_policy.yml \
    --overrides \
    --task_name ${task_name} \
    --task_config ${task_config} \
    --ckpt_setting ${ckpt_setting} \
    --expert_data_num ${expert_data_num} \
    --seed ${seed} \
    --checkpoint_num ${checkpoint_num} \
    --interactive_episodes ${interactive_episodes} \
    ${interactive_output_dir:+--interactive_output_dir ${interactive_output_dir}}
