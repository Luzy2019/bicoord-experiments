#!/usr/bin/env bash
# Usage:
#   bash train.sh 1   # Stage 1: train base flow (source-only rectified flow)
#   bash train.sh 2   # Stage 2: train asymmetric per-arm warp head (frozen base flow)
set -euo pipefail

STAGE="${1:-1}"

if [[ "$STAGE" == "1" ]]; then
    python3 train.py \
        --stage 1 \
        --data data/synthetic_bimanual_fm.npz \
        --output checkpoints/mypolicy_base.pt \
        --epochs 2000 \
        --batch-size 256
elif [[ "$STAGE" == "2" ]]; then
    python3 train.py \
        --stage 2 \
        --data data/synthetic_bimanual_fm.npz \
        --base-ckpt checkpoints/mypolicy_base.pt \
        --output checkpoints/mypolicy_speed.pt \
        --epochs 800 \
        --batch-size 256 \
        --lr 1e-4
else
    echo "Unknown stage: $STAGE (use 1 or 2)" >&2
    exit 1
fi
