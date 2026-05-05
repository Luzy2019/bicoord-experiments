#!/bin/bash

task_name=${1}
task_config=${2}
expert_data_num=${3}
seed=${4}
gpu_id=${5}

# Optional asymmetric execution settings.
# left_stride/right_stride are not fixed to "left high, right low":
#   1 5 means left updates faster than right
#   5 1 means right updates faster than left
#   1 1 means equal-frequency ACT with residual enabled
left_stride=${6:-1}
right_stride=${7:-5}
num_epochs=${8:-6000}
batch_size=${9:-8}
stride_choices=${10:-"1,1;1,5;5,1"}
target_mode=${11:-original}
schedule_mode=${12:-learned}

# schedule_mode:
#   fixed   - use left_stride/right_stride at inference; optionally sample
#             stride_choices during training, e.g. "1,1;1,5;5,1".
#   learned - learn left/right update gates from the original synchronous data.
#             At inference it can use a supplied role/instruction, or infer a
#             coarse role from the predicted action dynamics.
#
# target_mode:
#   projected - train against a hold-projected version of the synchronous
#               demonstrations. This directly reuses existing same-frequency data.
#   original  - train against the original synchronous demonstrations.

export CUDA_VISIBLE_DEVICES=${gpu_id}

extra_args=()
if [ -n "${stride_choices}" ]; then
    extra_args+=(--asym_stride_choices "${stride_choices}")
fi

python3 imitate_episodes.py \
    --task_name sim-${task_name}-${task_config}-${expert_data_num} \
    --ckpt_dir ./act_ckpt/asym-act-${task_name}/${task_config}-${expert_data_num}-L${left_stride}-R${right_stride} \
    --policy_class ACT \
    --kl_weight 10 \
    --chunk_size 50 \
    --hidden_dim 512 \
    --batch_size ${batch_size} \
    --dim_feedforward 3200 \
    --num_epochs ${num_epochs} \
    --lr 1e-5 \
    --seed ${seed} \
    --action_dim 14 \
    --use_asym_residual \
    --left_action_dim 7 \
    --right_action_dim 7 \
    --left_stride ${left_stride} \
    --right_stride ${right_stride} \
    --asym_hidden_dim 256 \
    --asym_residual_scale 0.1 \
    --asym_projector_mode hold \
    --asym_schedule_mode ${schedule_mode} \
    --asym_target_mode ${target_mode} \
    --asym_lambda_res 1e-4 \
    --asym_lambda_gate 1e-4 \
    --asym_lambda_smooth 1e-3 \
    --asym_lambda_freq 1e-2 \
    --asym_lambda_update_kl 1e-2 \
    --asym_lambda_update_sparse 1e-3 \
    --asym_lambda_role 1e-2 \
    "${extra_args[@]}"
