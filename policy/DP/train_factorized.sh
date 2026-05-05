#!/bin/bash

task_name=${1}
task_config=${2}
expert_data_num=${3}
seed=${4}
action_dim=${5}
gpu_id=${6}

factorized_aux_loss_weight=${7:-0.25}
factorized_gate_init_bias=${8:--2.0}
batch_size=${9:-}

head_camera_type=D435

DEBUG=False
save_ckpt=True

config_name=robot_dp_factorized_${action_dim}
addition_info=factorized_train
exp_name=${task_name}-robot_dp-${addition_info}
run_dir="data/outputs/${exp_name}_seed${seed}"

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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if [ ! -d "./data/${task_name}-${task_config}-${expert_data_num}.zarr" ]; then
    bash process_data.sh ${task_name} ${task_config} ${expert_data_num}
fi

batch_size_overrides=()
if [ -n "${batch_size}" ]; then
    batch_size_overrides+=("dataloader.batch_size=${batch_size}")
    batch_size_overrides+=("val_dataloader.batch_size=${batch_size}")
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
                            policy.factorized_aux_loss_weight=${factorized_aux_loss_weight} \
                            policy.factorized_gate_init_bias=${factorized_gate_init_bias} \
                            "${batch_size_overrides[@]}"
                            # checkpoint.save_ckpt=${save_ckpt}
                            # hydra.run.dir=${run_dir} \
