#!/bin/bash

# Stage 1: train a stable factorized bimanual flow-matching base policy without speed-head loss.
#
# Usage:
#   bash train_2unet_base.sh <task_name> <task_config> <expert_data_num> <seed> <action_dim> <gpu_id> [batch_size] [factorized_aux_loss_weight] [factorized_gate_init_bias]
#
# Example:
#   bash train_2unet_base.sh stack_bowls demo_clean 50 0 14 0 24 0.25 -2.0

set -e

task_name=${1}
task_config=${2}
expert_data_num=${3}
seed=${4}
action_dim=${5}
gpu_id=${6}

batch_size=${7:-24}
factorized_aux_loss_weight=${8:-0.25}
factorized_gate_init_bias=${9:--2.0}

head_camera_type=D435
DEBUG=False

config_name=robot_dp_factorized_two_unet_${action_dim}
addition_info=2unet_fm_base_train
exp_name=${task_name}-fm-${addition_info}

echo -e "\033[33mStage 1: train stable 2-UNET Flow Matching base policy\033[0m"
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"
echo -e "\033[33mconfig: ${config_name}\033[0m"
echo -e "\033[33mbatch size: ${batch_size}\033[0m"

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
    training.seed=${seed} \
    training.device="cuda:0" \
    exp_name=${exp_name} \
    logging.mode=${wandb_mode} \
    setting=${task_config} \
    expert_data_num=${expert_data_num} \
    head_camera_type=${head_camera_type} \
    dataloader.batch_size=${batch_size} \
    val_dataloader.batch_size=${batch_size} \
    policy.factorized_aux_loss_weight=${factorized_aux_loss_weight} \
    policy.factorized_gate_init_bias=${factorized_gate_init_bias} \
    policy.speed_modulation_enabled=False \
    policy.speed_modulation_loss_weight=0.0 \
    policy.speed_modulation_train_only=False
