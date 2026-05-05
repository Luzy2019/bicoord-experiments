#!/bin/bash

# bash eval.sh handover_block_with_bowls demo_clean demo_clean 50 42 0
# == keep unchanged ==
policy_name=DP
task_name=${1}
task_config=${2}
ckpt_setting=${3}
expert_data_num=${4}
seed=${5}
gpu_id=${6}
factor_debug_dir=${7:-debug_factorized}
factor_debug_max_calls=${8:-20}
DEBUG=False

export CUDA_VISIBLE_DEVICES=${gpu_id}
if [ "${factor_debug_dir}" != "none" ]; then
    export DP_FACTOR_DEBUG_DIR="${factor_debug_dir}"
    export DP_FACTOR_DEBUG_MAX_CALLS="${factor_debug_max_calls}"
    echo -e "\033[33mfactorized debug dir: ${DP_FACTOR_DEBUG_DIR}\033[0m"
    echo -e "\033[33mfactorized debug max calls: ${DP_FACTOR_DEBUG_MAX_CALLS}\033[0m"
else
    unset DP_FACTOR_DEBUG_DIR
    unset DP_FACTOR_DEBUG_MAX_CALLS
fi
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"

cd ../..

PYTHONWARNINGS=ignore::UserWarning \
python script/eval_policy.py --config policy/$policy_name/deploy_policy.yml \
    --overrides \
    --task_name ${task_name} \
    --task_config ${task_config} \
    --ckpt_setting ${ckpt_setting} \
    --expert_data_num ${expert_data_num} \
    --seed ${seed}