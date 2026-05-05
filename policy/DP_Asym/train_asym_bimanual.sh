#!/bin/bash

task_name=${1}
task_config=${2}
expert_data_num=${3}
seed=${4}
action_dim=${5}
gpu_id=${6}

# Optional role-adaptive objective hyperparameters.
left_stride=${7:-1}
right_stride=${8:-1}
lambda_cost=${9:-1e-3}
lambda_entropy=${10:-1e-4}
lambda_update_sparse=${11:-1e-3}
lambda_smooth=${12:-0.0}
lambda_role_smooth=${13:-1e-3}

head_camera_type=D435
DEBUG=False
alg_name=robot_dp_${action_dim}
config_name=${alg_name}
exp_name=${task_name}-asym_robot_dp-train

echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=${gpu_id}

if [ ! -d "./data/${task_name}-${task_config}-${expert_data_num}.zarr" ]; then
    bash process_data.sh ${task_name} ${task_config} ${expert_data_num}
fi

python train.py --config-name=${config_name}.yaml \
    task.name=${task_name} \
    task.dataset.zarr_path="data/${task_name}-${task_config}-${expert_data_num}.zarr" \
    training.debug=${DEBUG} \
    training.seed=${seed} \
    training.device="cuda:0" \
    exp_name=${exp_name} \
    logging.mode=online \
    setting=${task_config} \
    expert_data_num=${expert_data_num} \
    head_camera_type=${head_camera_type} \
    policy.left_stride=${left_stride} \
    policy.right_stride=${right_stride} \
    policy.asym_lambda_cost=${lambda_cost} \
    policy.asym_lambda_entropy=${lambda_entropy} \
    policy.asym_lambda_update_sparse=${lambda_update_sparse} \
    policy.asym_lambda_smooth=${lambda_smooth} \
    policy.asym_lambda_role_smooth=${lambda_role_smooth}
