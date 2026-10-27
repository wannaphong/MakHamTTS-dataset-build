#!/bin/bash
# ==============================================================================
# FastThaiG2P Two-Voice Training & Packaging Runner
# Voice 1: newvoice_female (Speaker 0)
# Voice 2: newvoice_male   (Speaker 1)
# ==============================================================================
# Usage:
#   GPU=0 bash train_two_voices.sh [prepare|stage1|stage2|package|test|smoke|all]
#
# Environment variable overrides:
#   GPU=0                  (CUDA GPU index)
#   BATCH_SIZE=8           (Stage 1 batch size)
#   STAGE2_BATCH_SIZE=4    (Stage 2 batch size; prevents OOM during adversarial phase)
#   EPOCHS_1ST=20          (Stage 1 epochs)
#   EPOCHS_2ND=10          (Stage 2 epochs)
#   MAX_SAMPLES=           (Limit samples for testing, e.g. 500)
# ==============================================================================

set -euo pipefail
cd "$(dirname "$0")"

GPU="${GPU:-0}"
STAGE="${1:-all}"
BATCH_SIZE="${BATCH_SIZE:-4}"
STAGE2_BATCH_SIZE="${STAGE2_BATCH_SIZE:-4}"
EPOCHS_1ST="${EPOCHS_1ST:-20}"
EPOCHS_2ND="${EPOCHS_2ND:-10}"
MAX_SAMPLES="${MAX_SAMPLES:-}"

EXTRA_ARGS=()
if [ -n "$MAX_SAMPLES" ]; then
    EXTRA_ARGS+=(--max-samples "$MAX_SAMPLES")
fi

if [ "$STAGE" == "smoke" ]; then
    echo "=== RUNNING SMOKE TEST (100 samples per voice, 1 epoch per stage) ==="
    python3 train_two_voices.py \
        --stage prepare \
        --max-samples 100

    python3 train_two_voices.py \
        --stage stage1 \
        --gpu "$GPU" \
        --batch-size 4 \
        --epochs-1st 1

    python3 train_two_voices.py \
        --stage stage2 \
        --gpu "$GPU" \
        --stage2-batch-size 2 \
        --epochs-2nd 1

    python3 train_two_voices.py \
        --stage package

    python3 train_two_voices.py \
        --stage test
    exit 0
fi

echo "=== FastThaiG2P Multi-Voice Trainer ($STAGE) on GPU $GPU ==="
python3 train_two_voices.py \
    --stage "$STAGE" \
    --gpu "$GPU" \
    --batch-size "$BATCH_SIZE" \
    --stage2-batch-size "$STAGE2_BATCH_SIZE" \
    --epochs-1st "$EPOCHS_1ST" \
    --epochs-2nd "$EPOCHS_2ND" \
    "${EXTRA_ARGS[@]}"
