#!/bin/bash

# Stage 2: initialize from a stable 2-UNet base checkpoint, freeze base policy,
# and train only the left/right SpeedHead modules.
#
# Usage:
#   bash train_2unet_speed_stage2.sh <task_name> <task_config> <expert_data_num> <seed> <action_dim> <gpu_id> <base_ckpt_path> [batch_size] [speed_loss_weight] [num_epochs]
#
# Example:
#   bash train_2unet_speed_stage2.sh stack_bowls demo_clean 50 0 14 0 \
#     data/outputs/2026.05.06/12.30.00_factorized_robot_stack_bowls_stack_bowls/checkpoints/latest.ckpt \
#     24 1e-2 100

set -e

task_name=${1}
task_config=${2}
expert_data_num=${3}
seed=${4}
action_dim=${5}
gpu_id=${6}
base_ckpt_path=${7}

batch_size=${8:-24}
speed_modulation_loss_weight=${9:-1e-2}
num_epochs=${10:-100}

if [ -z "${base_ckpt_path}" ]; then
    echo "Usage: bash train_2unet_speed_stage2.sh <task_name> <task_config> <expert_data_num> <seed> <action_dim> <gpu_id> <base_ckpt_path> [batch_size] [speed_loss_weight] [num_epochs]"
    exit 1
fi

head_camera_type=D435
DEBUG=False

config_name=robot_dp_factorized_${action_dim}
addition_info=2unet_speed_stage2
exp_name=${task_name}-robot_dp-${addition_info}

echo -e "\033[33mStage 2: freeze 2-UNet base, train SpeedHead only\033[0m"
echo -e "\033[33mbase checkpoint: ${base_ckpt_path}\033[0m"
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"
echo -e "\033[33mbatch size: ${batch_size}\033[0m"
echo -e "\033[33mspeed modulation loss weight: ${speed_modulation_loss_weight}\033[0m"
echo -e "\033[33mnum epochs: ${num_epochs}\033[0m"

if [ "${DEBUG}" = True ]; then
    wandb_mode=offline
else
    wandb_mode=online
fi

export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=${gpu_id}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if [ ! -d "./data/${task_name}-${task_config}-${expert_data_num}.zarr" ]; then
    bash process_data.sh ${task_name} ${task_config} ${expert_data_num}
fi

python train.py --config-name=${config_name}.yaml \
    task.name=${task_name} \
    task.dataset.zarr_path="data/${task_name}-${task_config}-${expert_data_num}.zarr" \
    training.debug=${DEBUG} \
    training.resume=False \
    training.init_ckpt_path="${base_ckpt_path}" \
    training.num_epochs=${num_epochs} \
    training.seed=${seed} \
    training.device="cuda:0" \
    exp_name=${exp_name} \
    logging.mode=${wandb_mode} \
    setting=${task_config} \
    expert_data_num=${expert_data_num} \
    head_camera_type=${head_camera_type} \
    dataloader.batch_size=${batch_size} \
    val_dataloader.batch_size=${batch_size} \
    policy.speed_modulation_enabled=True \
    policy.speed_modulation_learned=True \
    policy.speed_modulation_train_only=True \
    policy.speed_modulation_loss_weight=${speed_modulation_loss_weight}
