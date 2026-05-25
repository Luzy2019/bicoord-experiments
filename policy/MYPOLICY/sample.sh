#!/usr/bin/env bash
# Usage:
#   bash sample.sh stage1                   # sample from base flow only (no warp)
#   bash sample.sh stage2                   # sample then per-arm warp -> parallel
#   bash sample.sh stage2 warp              # warp real source samples through learned head
set -euo pipefail

WHICH="${1:-stage2}"
MODE="${2:-generate}"

if [[ "$WHICH" == "stage1" ]]; then
    python3 sample.py \
        --checkpoint checkpoints/mypolicy_base.pt \
        --data data/synthetic_bimanual_fm.npz \
        --output outputs/mypolicy_samples_stage1.npz \
        --num-samples 8 \
        --mode "$MODE"
elif [[ "$WHICH" == "stage2" ]]; then
    python3 sample.py \
        --checkpoint checkpoints/mypolicy_speed.pt \
        --data data/synthetic_bimanual_fm.npz \
        --output outputs/mypolicy_samples_stage2.npz \
        --num-samples 8 \
        --mode "$MODE"
else
    echo "Unknown checkpoint selector: $WHICH (use stage1 or stage2)" >&2
    exit 1
fi
