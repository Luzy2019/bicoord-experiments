#!/usr/bin/env bash
# Usage:
#   bash visualize_data.sh                          # 可视化数据集 (source + target)
#   bash visualize_data.sh outputs/xxx.npz outdir   # 可视化采样结果
set -euo pipefail

INPUT="${1:-outputs/mypolicy_samples_stage1.npz}"
OUTDIR="${2:-outputs/data_vis}"

python3 visualize_data.py \
    --input "$INPUT" \
    --output-dir "$OUTDIR" \
    --num-samples 8
