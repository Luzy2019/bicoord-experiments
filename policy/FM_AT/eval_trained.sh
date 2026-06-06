#!/bin/bash
set -e

# Usage:
#   bash eval_trained.sh <task_name> <task_config> <ckpt_setting> <expert_data_num> <seed> <gpu_id> <checkpoint_num> [speed_enabled] [factor_debug_dir] [factor_debug_max_calls] [ckpt_dir_name]
#
# Examples:
#   bash eval_trained.sh stack_bowls demo_clean demo_clean 50 0 0 400 true none
#   bash eval_trained.sh stack_bowls demo_clean demo_clean 50 0 0 400 true debug_factorized 60
#   bash eval_trained.sh stack_bowls demo_clean demo_clean 50 100 0 100 true none 60 stack_bowls-demo_clean-50-100-speed

policy_name=FM_AT
task_name=${1}
task_config=${2}
ckpt_setting=${3}
expert_data_num=${4}                # 50
seed=${5}                           # 100
gpu_id=${6}                         # 0
checkpoint_num=${7}                 # 100
speed_modulation_enabled=${8:-auto} # false / true
factor_debug_dir=${9:-none}         # none
factor_debug_max_calls=${10:-60}    # 60
ckpt_dir_name=${11:-}               # stack_bowls-demo_clean-50-100-speed

speed_modulation_strength=${DP_SPEED_MODULATION_STRENGTH:-1.0}
speed_modulation_min=${DP_SPEED_MODULATION_MIN:-0.5}
speed_modulation_max=${DP_SPEED_MODULATION_MAX:-2.0}
speed_modulation_smooth=${DP_SPEED_MODULATION_SMOOTH:-3}
speed_modulation_learned=${DP_SPEED_MODULATION_LEARNED:-true}

eval_video_overlay=${DP_EVAL_VIDEO_OVERLAY:-false}
eval_debug_json=${DP_EVAL_DEBUG_JSON:-false}
eval_video_fps=${DP_EVAL_VIDEO_FPS:-10}

if [ -z "${checkpoint_num}" ]; then
    echo "Usage: bash eval_trained.sh <task_name> <task_config> <ckpt_setting> <expert_data_num> <seed> <gpu_id> <checkpoint_num> [speed_enabled] [factor_debug_dir] [factor_debug_max_calls] [ckpt_dir_name]"
    exit 1
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
cd "${repo_root}"

if [ "${factor_debug_dir}" != "none" ] && [[ "${factor_debug_dir}" != /* ]]; then
    factor_debug_dir="${repo_root}/${factor_debug_dir}"
fi

export CUDA_VISIBLE_DEVICES=${gpu_id}
export DP_SPEED_MODULATION_STRENGTH="${speed_modulation_strength}"
export DP_SPEED_MODULATION_MIN="${speed_modulation_min}"
export DP_SPEED_MODULATION_MAX="${speed_modulation_max}"
export DP_SPEED_MODULATION_SMOOTH="${speed_modulation_smooth}"
export DP_SPEED_MODULATION_LEARNED="${speed_modulation_learned}"

if [ "${speed_modulation_enabled}" != "auto" ]; then
    export DP_SPEED_MODULATION_ENABLED="${speed_modulation_enabled}"
else
    unset DP_SPEED_MODULATION_ENABLED
fi

if [ "${factor_debug_dir}" != "none" ]; then
    export DP_FACTOR_DEBUG_DIR="${factor_debug_dir}"
    export DP_FACTOR_DEBUG_MAX_CALLS="${factor_debug_max_calls}"
else
    unset DP_FACTOR_DEBUG_DIR
    unset DP_FACTOR_DEBUG_MAX_CALLS
fi

echo -e "\033[33mgpu id: ${gpu_id}\033[0m"
if [ -n "${ckpt_dir_name}" ]; then
    echo -e "\033[33mcheckpoint: policy/${policy_name}/checkpoints/${ckpt_dir_name}/${checkpoint_num}.ckpt\033[0m"
else
    echo -e "\033[33mcheckpoint: policy/${policy_name}/checkpoints/${task_name}-${ckpt_setting}-${expert_data_num}-${seed}/${checkpoint_num}.ckpt\033[0m"
fi
echo -e "\033[33mspeed modulation: ${speed_modulation_enabled}, learned=${speed_modulation_learned}, range=[${speed_modulation_min}, ${speed_modulation_max}]\033[0m"
echo -e "\033[33mfactorized debug dir: ${factor_debug_dir}, max_calls=${factor_debug_max_calls}\033[0m"
echo -e "\033[33meval overlay: ${eval_video_overlay}, debug json: ${eval_debug_json}, video fps: ${eval_video_fps}\033[0m"

PYTHONWARNINGS=ignore::UserWarning \
python script/eval_policy.py --config policy/${policy_name}/deploy_policy.yml \
    --overrides \
    --task_name ${task_name} \
    --task_config ${task_config} \
    --ckpt_setting ${ckpt_setting} \
    --expert_data_num ${expert_data_num} \
    --seed ${seed} \
    --checkpoint_num ${checkpoint_num} \
    --eval_video_overlay ${eval_video_overlay} \
    --eval_debug_json ${eval_debug_json} \
    --eval_video_fps ${eval_video_fps} \
    ${ckpt_dir_name:+--ckpt_dir_name ${ckpt_dir_name}}
