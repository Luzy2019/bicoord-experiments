#!/bin/bash

task_name=${1}
task_config=${2}
expert_data_num=${3}
seed=${4}
action_dim=${5}
gpu_id=${6}
train_stage=${7:-base}   # base | speed | joint
speed_modulation_loss_weight=${8:-1e-2}
init_ckpt_path=${9:-}

head_camera_type=D435

DEBUG=False
save_ckpt=True

alg_name=robot_dp_$action_dim
config_name=${alg_name}
addition_info=train_${train_stage}
exp_name=${task_name}-robot_dp-${addition_info}
run_dir="data/outputs/${exp_name}_seed${seed}"

echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"


if [ $DEBUG = True ]; then
    wandb_mode=offline
    # wandb_mode=online
    echo -e "\033[33mDebug mode!\033[0m"
    echo -e "\033[33mDebug mode!\033[0m"
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

speed_overrides=()
if [ "${train_stage}" = "base" ]; then
    speed_overrides+=("policy.speed_modulation_enabled=False")
    speed_overrides+=("policy.speed_modulation_train_only=False")
    speed_overrides+=("policy.speed_modulation_loss_weight=0.0")
elif [ "${train_stage}" = "speed" ]; then
    if [ -z "${init_ckpt_path}" ]; then
        echo "Usage for speed stage: bash train.sh <task> <config> <data_num> <seed> <action_dim> <gpu> speed <speed_loss_weight> <init_ckpt_path>"
        exit 1
    fi
    speed_overrides+=("training.resume=False")
    speed_overrides+=("training.init_ckpt_path=${init_ckpt_path}")
    speed_overrides+=("policy.speed_modulation_enabled=True")
    speed_overrides+=("policy.speed_modulation_learned=True")
    speed_overrides+=("policy.speed_modulation_train_only=True")
    speed_overrides+=("policy.speed_modulation_loss_weight=${speed_modulation_loss_weight}")
elif [ "${train_stage}" = "joint" ]; then
    if [ -n "${init_ckpt_path}" ]; then
        speed_overrides+=("training.resume=False")
        speed_overrides+=("training.init_ckpt_path=${init_ckpt_path}")
    fi
    speed_overrides+=("policy.speed_modulation_enabled=True")
    speed_overrides+=("policy.speed_modulation_learned=True")
    speed_overrides+=("policy.speed_modulation_train_only=False")
    speed_overrides+=("policy.speed_modulation_loss_weight=${speed_modulation_loss_weight}")
else
    echo "Unknown train_stage=${train_stage}; expected base, speed, or joint"
    exit 1
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
                            "${speed_overrides[@]}"
                            # checkpoint.save_ckpt=${save_ckpt}
                            # hydra.run.dir=${run_dir} \