#!/bin/bash
set -euo pipefail

TARGET_MODEL=${TARGET_MODEL:?Set TARGET_MODEL}
DRAFT_MODEL=${DRAFT_MODEL:?Set DRAFT_MODEL}
SCORER_CKPT=${SCORER_CKPT:-outputs/scorer/best.pt}

for DATASET in aime25 gsm8k math500 humaneval livecodebench mbpp mt-bench; do
  echo "=== DDTree baseline: $DATASET ==="
  python benchmark.py \
    --model-name-or-path "$TARGET_MODEL" \
    --draft-name-or-path "$DRAFT_MODEL" \
    --dataset $DATASET \
    --shuffle-seed 2026 \
    --max-new-tokens 2048 \
    --save-path outputs/ddtree_${DATASET}.pt \
    --proposal-mode ddtree \
    --tree-budget 512

  echo "=== TAPS hybrid: $DATASET ==="
  python benchmark.py \
    --model-name-or-path "$TARGET_MODEL" \
    --draft-name-or-path "$DRAFT_MODEL" \
    --dataset $DATASET \
    --shuffle-seed 2026 \
    --max-new-tokens 2048 \
    --save-path outputs/taps_hybrid_${DATASET}.pt \
    --proposal-mode joint \
    --tree-budget 64 \
    --tiny-scorer-checkpoint "$SCORER_CKPT" \
    --joint-topk 64 \
    --candidate-pool-nodes 768 \
    --candidate-pool-sequences 48 \
    --candidate-pool-source taps_lite \
    --min-verify-nodes 4 \
    --max-verify-nodes 192 \
    --min-verify-sequences 4 \
    --max-verify-sequences 64 \
    --no-fallback-to-ddtree \
    --fallback-backend none \
    --hybrid
done

echo "Done. Results saved to outputs/"
