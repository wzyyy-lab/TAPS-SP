#!/bin/bash
set -euo pipefail

TARGET_MODEL=${TARGET_MODEL:?Set TARGET_MODEL}

python scripts/train_scorer.py \
  --traces outputs/traces \
  --output outputs/scorer \
  --target-model "$TARGET_MODEL" \
  --epochs 30 \
  --lr 3e-3 \
  --weight-decay 1e-4 \
  --hidden-dim 64 \
  --batch-records 64 \
  --depth-embed-dim 8 \
  --token-proj-dim 32 \
  --hidden-proj-dim 32 \
  --scalar-dim 7 \
  --draft-hidden-dim 2560 \
  --lambda-reach 0.5 \
  --val-fraction 0.1 \
  --seed 2026

echo "Done. Best checkpoint: outputs/scorer/best.pt"
