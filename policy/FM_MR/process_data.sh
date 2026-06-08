#!/bin/bash
# 用法: bash process_data.sh <task_name> <task_config> <expert_data_num>
# 例如: bash process_data.sh beat_block_hammer default 50

task_name=${1}
task_config=${2}
expert_data_num=${3}

python process_data.py $task_name $task_config $expert_data_num