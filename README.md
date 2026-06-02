# TAPS: Target-Aware Prefix Tree Selection for Speculative Decoding

TAPS is a learned proposal selector for DDTree-style speculative decoding. Given a DFlash block-parallel draft model, TAPS builds a large candidate pool from draft logits, scores each candidate node with a lightweight scorer, and selects a compact, high-quality verification tree for the target model. The result is higher acceptance length with minimal throughput overhead.

## Supported Models

| Target model | Draft model |
| --- | --- |
| `Qwen/Qwen3-4B` | `Huang2020/Qwen3-4B-DFlash-b16` |

## Method

Each speculative decoding round proceeds as:

1. **Draft**: The DFlash draft model produces block-parallel draft logits in one forward pass.
2. **Lattice extraction**: Extract top-K token candidates at each draft position.
3. **Candidate pool construction**: CPU beam search builds a 768-node candidate trie from draft cumulative log probabilities (~0.3 ms).
4. **Scorer**: A 177K-parameter `TAPSLiteScorer` scores every candidate edge using target-model token embeddings, draft hidden states, and positional features (~1.2 ms on GPU).
5. **Selection**: Reach-based utility ranking + prefix-closed closure selects the best 64 nodes for verification.
6. **Verification**: The target model verifies the selected tree in a single forward pass.

The scorer uses a hybrid pipeline: CPU numpy beam search for fast pool construction, GPU MLP for accurate scoring with correct parent context.

## Installation

Tested with Python 3.11, PyTorch 2.x with CUDA, and an NVIDIA A800/A100 GPU.

```bash
git clone https://github.com/wzyyy-lab/TAPS-EMNLP2026.git
cd TAPS-EMNLP2026
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

The `flash-attn` package requires a CUDA build toolchain. If installation fails, install it separately following [flash-attention instructions](https://github.com/Dao-AILab/flash-attention).

## Quick Start

```bash
export TARGET_MODEL=/path/to/Qwen3-4B
export DRAFT_MODEL=/path/to/Qwen3-4B-DFlash-b16
export SCORER_CKPT=/path/to/taps_scorer/best.pt
```

### Reproduce the best configuration (hybrid, pool=768/48, verify=64)

```bash
python benchmark.py \
  --model-name-or-path "$TARGET_MODEL" \
  --draft-name-or-path "$DRAFT_MODEL" \
  --dataset gsm8k \
  --max-samples 64 \
  --shuffle-seed 2026 \
  --max-new-tokens 512 \
  --save-path outputs/taps_hybrid_gsm8k.pt \
  --proposal-mode joint \
  --tree-budget 64 \
  --tiny-scorer-checkpoint "$SCORER_CKPT" \
  --joint-topk 64 \
  --candidate-pool-nodes 768 \
  --candidate-pool-sequences 48 \
  --candidate-pool-source taps_lite \
  --min-verify-nodes 4 \
  --max-verify-nodes 64 \
  --min-verify-sequences 4 \
  --max-verify-sequences 64 \
  --no-fallback-to-ddtree \
  --fallback-backend none \
  --hybrid
```

The `--hybrid` flag enables the hybrid selection pipeline (CPU beam search + GPU scoring), which is the recommended configuration. To use the hybrid path programmatically:

```python
from joint.taps_lite_scorer import load_taps_lite_scorer

scorer, _ = load_taps_lite_scorer("path/to/best.pt", device=device)
scorer.set_vocab_embeds(target_model.model.embed_tokens.weight)
scorer._use_hybrid = True
```

### Run DDTree-64 baseline

```bash
python benchmark.py \
  --model-name-or-path "$TARGET_MODEL" \
  --draft-name-or-path "$DRAFT_MODEL" \
  --dataset gsm8k \
  --max-samples 64 \
  --shuffle-seed 2026 \
  --max-new-tokens 512 \
  --save-path outputs/ddtree64_gsm8k.pt \
  --proposal-mode ddtree \
  --tree-budget 64
```

## Datasets

The benchmark loader supports the following datasets:

| Dataset | Source | Domain |
| --- | --- | --- |
| `gsm8k` | [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k) | Math reasoning |
| `humaneval` | [openai/openai_humaneval](https://huggingface.co/datasets/openai/openai_humaneval) | Code generation |
| `mbpp` | [google-research-datasets/mbpp](https://huggingface.co/datasets/google-research-datasets/mbpp) | Code generation |
| `mt-bench` | [lmsys/mt_bench_human_judgments](https://huggingface.co/datasets/lmsys/mt_bench_human_judgments) | Multi-turn chat |
| `math500` | [HuggingFaceH4/MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500) | Math reasoning |

Datasets are automatically downloaded from Hugging Face on first use. Alternatively, place local copies under a directory and set `TAPS_HF_ASSETS` to point to it.

## Reproduce from Scratch

The full pipeline has three stages: trace collection, scorer training, and benchmarking.

### Step 1: Collect Traces

Traces record draft logits, candidate tries, and target-model acceptance labels for training the scorer.

```bash
for DATASET in alpaca sharegpt codealpaca math; do
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
```

This collects 200 prompts per dataset (800 total) from four diverse domains. Each prompt produces multiple decoding rounds, yielding thousands of training records. On a single A800 GPU, expect ~2 hours per dataset.

Training datasets for traces:

| Dataset | Prompts | Domain |
| --- | --- | --- |
| `alpaca` | 200 | Instruction following |
| `sharegpt` | 200 | Multi-turn chat |
| `codealpaca` | 200 | Code |
| `math` | 200 | Math reasoning |

### Step 2: Train Scorer

The `TAPSLiteScorer` (177K parameters) is trained with two losses: KL divergence on per-parent conditional distributions + BCE on reach propagation.

```bash
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
```

Training takes ~10 minutes on a single GPU. The best checkpoint is saved to `outputs/scorer/best.pt`.

### Step 3: Benchmark

Run the full benchmark comparing DDTree-64 baseline with TAPS (hybrid selection):

```bash
export SCORER_CKPT=outputs/scorer/best.pt

for DATASET in gsm8k humaneval mbpp mt-bench; do
  # DDTree-64 baseline
  python benchmark.py \
    --model-name-or-path "$TARGET_MODEL" \
    --draft-name-or-path "$DRAFT_MODEL" \
    --dataset $DATASET \
    --max-samples 64 \
    --shuffle-seed 2026 \
    --max-new-tokens 512 \
    --save-path outputs/ddtree64_${DATASET}.pt \
    --proposal-mode ddtree \
    --tree-budget 64

  # TAPS hybrid
  python benchmark.py \
    --model-name-or-path "$TARGET_MODEL" \
    --draft-name-or-path "$DRAFT_MODEL" \
    --dataset $DATASET \
    --max-samples 64 \
    --shuffle-seed 2026 \
    --max-new-tokens 512 \
    --save-path outputs/taps_hybrid_${DATASET}.pt \
    --proposal-mode joint \
    --tree-budget 64 \
    --tiny-scorer-checkpoint "$SCORER_CKPT" \
    --joint-topk 64 \
    --candidate-pool-nodes 768 \
    --candidate-pool-sequences 48 \
    --candidate-pool-source taps_lite \
    --min-verify-nodes 4 \
    --max-verify-nodes 64 \
    --min-verify-sequences 4 \
    --max-verify-sequences 64 \
    --no-fallback-to-ddtree \
    --fallback-backend none \
    --hybrid
done
```

## Scorer Architecture

The `TAPSLiteScorer` has 176,894 trainable parameters:

| Component | Shape | Parameters | Description |
| --- | --- | --- | --- |
| `token_proj` | 2560 → 32 | 81,920 | Projects target model embeddings to 32-dim |
| `hidden_proj` | 2560 → 32 | 81,920 | Projects draft hidden states to 32-dim |
| `edge_mlp` | 111 → 64 → 1 | 7,233 | Scores child-parent edges |
| `other_mlp` | 79 → 64 → 1 | 5,185 | Scores "other" (uncovered) probability per parent |
| norms + depth_embed | — | 636 | LayerNorm, depth embedding |

The scorer uses frozen target-model token embeddings (via `token_proj`) and draft-model hidden states (via `hidden_proj`). Only the projection layers and MLPs are trained. The `edge_mlp` input concatenates: child embedding (32), parent embedding (32), depth embedding (8), scalar features (7), hidden projection (32) = 111 dimensions.

## Results

All results use Qwen3-4B as target, Qwen3-4B-DFlash-b16 as draft, `shuffle_seed=2026`, `max_new_tokens=512`, and greedy decoding (`temperature=0.0`). The TAPS configuration is hybrid selection with `pool=768/48`, `verify=64`, and the v5 scorer.

### TAPS (hybrid) vs DDTree-64

| Dataset | N | DDTree-64 acc | DDTree-64 tok/s | TAPS acc | TAPS tok/s | Δacc | Δtps |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| gsm8k | 64 | 8.08 | 200.4 | 8.75 | 203.1 | +8.2% | +1.3% |
| humaneval | 64 | 8.48 | 211.2 | 8.98 | 209.7 | +5.9% | −0.7% |
| mbpp | 64 | 7.88 | 195.2 | 8.49 | 196.8 | +7.7% | +0.8% |
| mt-bench | 40 | 4.78 | 119.5 | 5.18 | 121.5 | +8.4% | +1.7% |
| **Overall** | — | 7.31 | 182.1 | 7.85 | 183.4 | **+7.4%** | **+0.7%** |

TAPS improves acceptance length by +7.4% on average across all four datasets while maintaining positive throughput (+0.7%), with only 2.6 ms per-round scorer overhead.

## Project Structure

```
├── benchmark.py                 # Main benchmark entry point
├── ddtree.py                    # DDTree baseline generation
├── dflash.py                    # DFlash generation + timing utilities
├── distributed.py               # Multi-GPU distributed utilities
├── requirements.txt
├── model/
│   ├── __init__.py
│   ├── dflash.py                # DFlash draft model
│   └── utils.py                 # Dataset loading, sampling utilities
├── joint/
│   ├── __init__.py
│   ├── config.py                # JointDDTConfig dataclass
│   ├── lattice.py               # Top-K lattice extraction
│   ├── pool.py                  # Candidate trie construction
│   ├── runtime.py               # Joint generation loop
│   ├── segments.py              # Grouped softmax, KL divergence
│   ├── selector.py              # Reach propagation, prefix-closed selection
│   ├── taps_lite_scorer.py      # TAPSLiteScorer model + hybrid selection
│   ├── trace.py                 # Trace data structures for collection
│   └── tree.py                  # Verification tree compilation
└── scripts/
    ├── collect_trace.py         # Trace collection script
    └── train_scorer.py          # Scorer training script
```

## Acknowledgements

TAPS builds on the [DDTree](https://github.com/z-lab/dflash) and [DFlash](https://github.com/z-lab/dflash) speculative decoding framework. We thank the authors of [Domino](https://github.com/jianuo-huang/Domino) for their open-source implementation of fused speculative decoding kernels, which inspired the overhead optimization in this work.
