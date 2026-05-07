#!/bin/bash

# Joint-training ablation for the factorized bimanual DP model with learned
# asymmetric speed modulation. For the recommended staged pipeline, use:
#   1) train_2unet_base.sh
#   2) train_2unet_speed_stage2.sh
#
# Usage:
#   bash train_2unet_speed.sh <task_name> <task_config> <expert_data_num> <seed> <action_dim> <gpu_id> [batch_size] [factorized_aux_loss_weight] [factorized_gate_init_bias] [speed_loss_weight]
#
# Example:
#   bash train_2unet_speed.sh stack_bowls demo_clean 50 0 14 0 24 0.25 -2.0 1e-2

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
speed_modulation_loss_weight=${10:-1e-2}

head_camera_type=D435
DEBUG=False

config_name=robot_dp_factorized_two_unet_${action_dim}
addition_info=2unet_speed_train
exp_name=${task_name}-fm-${addition_info}

echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"
echo -e "\033[33mconfig: ${config_name}\033[0m"
echo -e "\033[33mbatch size: ${batch_size}\033[0m"
echo -e "\033[33mfactorized aux loss weight: ${factorized_aux_loss_weight}\033[0m"
echo -e "\033[33mfactorized gate init bias: ${factorized_gate_init_bias}\033[0m"
echo -e "\033[33mspeed modulation loss weight: ${speed_modulation_loss_weight}\033[0m"

if [ "${DEBUG}" = True ]; then
    wandb_mode=offline
    echo -e "\033[33mDebug mode\033[0m"
else
    wandb_mode=online
    echo -e "\033[33mTrain mode\033[0m"
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
    policy.speed_modulation_enabled=True \
    policy.speed_modulation_learned=True \
    policy.speed_modulation_loss_weight=${speed_modulation_loss_weight}
