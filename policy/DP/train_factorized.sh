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
train_stage=${10:-base}   # base | speed | joint
speed_modulation_loss_weight=${11:-1e-2}
init_ckpt_path=${12:-}
every_save_epoch=${13:-}
total_train_epoch=${14:-}
checkpoint_dir_name=${15:-}

head_camera_type=D435

DEBUG=False
save_ckpt=True

config_name=robot_dp_factorized_${action_dim}
addition_info=factorized_${train_stage}
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

zarr_path="./data/${task_name}-${task_config}-${expert_data_num}.zarr"
if [ ! -d "${zarr_path}" ]; then
    bash process_data.sh ${task_name} ${task_config} ${expert_data_num}
elif ! python - "${zarr_path}" <<'PY'
import sys
import zarr

root = zarr.open(sys.argv[1], mode="r")
data = root["data"]
required_keys = {"head_camera", "left_camera", "right_camera", "state", "action"}
missing_keys = sorted(required_keys - set(data.keys()))
if missing_keys:
    print("Missing zarr keys:", ", ".join(missing_keys))
    raise SystemExit(1)
PY
then
    echo "Existing zarr is missing wrist camera data; rebuilding ${zarr_path}"
    rm -rf "${zarr_path}"
    bash process_data.sh ${task_name} ${task_config} ${expert_data_num}
fi

batch_size_overrides=()
if [ -n "${batch_size}" ]; then
    batch_size_overrides+=("dataloader.batch_size=${batch_size}")
    batch_size_overrides+=("val_dataloader.batch_size=${batch_size}")
fi

training_overrides=()
if [ -n "${every_save_epoch}" ]; then
    training_overrides+=("training.checkpoint_every=${every_save_epoch}")
fi
if [ -n "${total_train_epoch}" ]; then
    training_overrides+=("training.num_epochs=${total_train_epoch}")
fi
if [ -n "${checkpoint_dir_name}" ]; then
    training_overrides+=("+training.checkpoint_dir_name=${checkpoint_dir_name}")
fi

speed_overrides=()
if [ "${train_stage}" = "base" ]; then
    speed_overrides+=("policy.speed_modulation_enabled=False")
    speed_overrides+=("policy.speed_modulation_train_only=False")
    speed_overrides+=("policy.speed_modulation_loss_weight=0.0")
elif [ "${train_stage}" = "speed" ]; then
    if [ -z "${init_ckpt_path}" ]; then
        echo "Usage for speed stage: bash train_factorized.sh <task> <config> <data_num> <seed> <action_dim> <gpu> <aux_weight> <gate_bias> <batch_size> speed <speed_loss_weight> <init_ckpt_path>"
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
                            policy.factorized_aux_loss_weight=${factorized_aux_loss_weight} \
                            policy.factorized_gate_init_bias=${factorized_gate_init_bias} \
                            "${batch_size_overrides[@]}" \
                            "${training_overrides[@]}" \
                            "${speed_overrides[@]}"
                            # checkpoint.save_ckpt=${save_ckpt}
                            # hydra.run.dir=${run_dir} \
