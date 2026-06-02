#!/bin/bash
set -euo pipefail

TARGET_MODEL=${TARGET_MODEL:?Set TARGET_MODEL}
DRAFT_MODEL=${DRAFT_MODEL:?Set DRAFT_MODEL}

for DATASET in alpaca sharegpt codealpaca math; do
  echo "=== Collecting traces: $DATASET ==="
  python scripts/collect_trace.py \
    --target-model "$TARGET_MODEL" \
    --draft-model "$DRAFT_MODEL" \
    --datasets "$DATASET" \
    --tree-budget-baseline 512 \
    --topk-collect 512 \
    --candidate-pool-nodes 512 \
    --candidate-pool-sequences 512 \
    --max-samples 200 \
    --shuffle-seed 2026 \
    --max-new-tokens 512 \
    --output outputs/traces/$DATASET
done

echo "Done. Traces saved to outputs/traces/"
