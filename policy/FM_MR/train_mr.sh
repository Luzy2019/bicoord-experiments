#!/bin/bash

task_name=${1}
task_config=${2}
expert_data_num=${3}
seed=${4}
action_dim=${5}
gpu_id=${6}
stage=${7:-teacher}   # teacher | student
teacher_ckpt_path=${8:-}

head_camera_type=D435
DEBUG=False

alg_name=robot_mr_${action_dim}
config_name=${alg_name}

if [ "${stage}" = "teacher" ]; then
    addition_info=teacher
elif [ "${stage}" = "student" ]; then
    addition_info=student
    if [ -z "${teacher_ckpt_path}" ]; then
        echo "Usage for student stage: bash train_mr.sh <task> <config> <data_num> <seed> <action_dim> <gpu> student <teacher_ckpt_path>"
        exit 1
    fi
else
    echo "Unknown stage=${stage}; expected teacher or student"
    exit 1
fi

exp_name=${task_name}-robot_fm_mr-${addition_info}

echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"
if [ $DEBUG = True ]; then
    wandb_mode=offline
    echo -e "\033[33mDebug mode!\033[0m"
else
    wandb_mode=online
    echo -e "\033[33mTrain mode\033[0m"
fi

export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=${gpu_id}

if [ ! -d "./data/${task_name}-${task_config}-${expert_data_num}.zarr" ]; then
    bash process_data.sh ${task_name} ${task_config} ${expert_data_num}
fi

stage_overrides=()
if [ "${stage}" = "teacher" ]; then
    stage_overrides+=("policy._target_=flow_matching_policy.policy.diffusion_unet_image_policy.DiffusionUnetImagePolicy")
    stage_overrides+=("policy.teacher.enabled=False")
    stage_overrides+=("training.checkpoint_dir_name=${task_name}-${task_config}-${expert_data_num}-${seed}-teacher")
else
    stage_overrides+=("policy.teacher.enabled=True")
    stage_overrides+=("policy.teacher.ckpt_path=${teacher_ckpt_path}")
    stage_overrides+=("training.checkpoint_dir_name=${task_name}-${task_config}-${expert_data_num}-${seed}-student")
fi

python train.py --config-name=${config_name}.yaml \
    task.name=${task_name} \
    task.dataset.zarr_path="data/${task_name}-${task_config}-${expert_data_num}.zarr" \
    training.debug=$DEBUG \
    training.seed=${seed} \
    training.device="cuda:0" \
    exp_name=${exp_name} \
    logging.mode=${wandb_mode} \
    setting=${task_config} \
    expert_data_num=${expert_data_num} \
    head_camera_type=$head_camera_type \
    "${stage_overrides[@]}"
