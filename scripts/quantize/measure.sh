#!/bin/bash
# Measure a checkpoint's quantization quality against bf16 with the yardstick of
# docs/quantization.md: KL over 81,920 positions, WikiText-2 perplexity, top-1
# agreement. Takes ~4 min per candidate once the reference exists.
#
#   bash scripts/quantize/measure.sh /workspace/models/Qwen3.8-27B-int4g128-gptq-mse [name]
#
# The bf16 reference (41 GB of logits) is built on first use: one pass of the
# bf16 checkpoint through our own streamed forward, ~2 min on one card, no
# two-GPU box needed. The rivals' rows need their checkpoints; see
# docs/quantization.md for the source specs bench/quality_sources.py accepts.
set -euo pipefail
cd "$(dirname "$0")/../.."

MODELS=${MODELS:-/workspace/models}
SRC=${SRC:-$MODELS/Qwen3.8-27B}
REF=${REF:-/workspace/ref/logits}
CHUNKS=${CHUNKS:-data/quality/chunks.npz}
CKPT=${1:?usage: measure.sh <packed checkpoint dir> [name]}
NAME=${2:-$(basename "$CKPT")}

if [ ! -f "$REF/math_01.pt" ]; then
    echo "== bf16 reference logits -> $REF (41 GB)"
    python bench/quality_logits.py --src "hf:$SRC" --hf "$SRC" --chunks "$CHUNKS" \
        --save-ref "$REF" --out "results/quality/bf16_ref.json"
fi

echo "== $NAME"
python bench/quality_logits.py --src "packed:$CKPT" --hf "$SRC" --chunks "$CHUNKS" \
    --ref "$REF" --out "results/quality/$NAME.json"

echo
echo "== attribution (how much of the loss is the head)"
python bench/quality_logits.py --src "packed:$CKPT" --hf "$SRC" --chunks "$CHUNKS" \
    --ref "$REF" --keep-bf16 lm_head --out "results/quality/${NAME}_bf16head.json" | tail -n 2
