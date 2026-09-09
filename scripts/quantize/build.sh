#!/bin/bash
# Build the engine's checkpoint from the public bf16 weights, on a machine that
# has nothing on it. This is the whole recipe: nothing on a vast instance
# survives a recycle, so this script is the checkpoint.
#
#   bash scripts/quantize/build.sh                      # the default: int4 g128 GPTQ + MSE
#   VARIANT=rtn bash scripts/quantize/build.sh          # the uncalibrated baseline (docs/quantization.md)
#   VARIANT=gptq bash scripts/quantize/build.sh         # GPTQ without the range search
#
# One RTX 5090 is enough (the quantizer streams one layer at a time). 20 min
# for GPTQ, 40 s for RTN, plus ~4 min to download the 55.6 GB of bf16 weights.
# Reproducibility: the calibration ids are committed (data/quality/calib.npz),
# so the codes do not depend on this machine's datasets or torch version.
set -euo pipefail
cd "$(dirname "$0")/../.."

MODELS=${MODELS:-/workspace/models}
SRC=${SRC:-$MODELS/Qwen3.8-27B}
VARIANT=${VARIANT:-gptq-mse}
DST=${DST:-$MODELS/Qwen3.8-27B-int4g128-$VARIANT}
CALIB=${CALIB:-data/quality/calib.npz}

echo "== bf16 weights -> $SRC"
if [ ! -d "$SRC" ]; then
    hf download Qwen/Qwen3.8-27B --local-dir "$SRC"
else
    echo "   already there"
fi

echo "== quantize ($VARIANT) -> $DST"
case "$VARIANT" in
    rtn)      python -m tokenrush.quantize --src "$SRC" --dst "$DST" ;;
    gptq)     python -m tokenrush.gptq --src "$SRC" --dst "$DST" --calib "$CALIB" ;;
    gptq-mse) python -m tokenrush.gptq --src "$SRC" --dst "$DST" --calib "$CALIB" --mse ;;
    *) echo "unknown VARIANT: $VARIANT (rtn | gptq | gptq-mse)"; exit 1 ;;
esac

echo "== smoke"
python -m tokenrush.run --model "$DST" --chat --no-spec --max-new 40 \
    --prompt "Explain in three sentences why the sky is blue."
echo
echo "checkpoint at $DST ($(cat "$DST/tokenrush.json" | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d["method"], "group", d["group"], "mse", d.get("mse"), "%.2f GB" % (d["packed_bytes"]/1e9))'))"
echo "measure it: bash scripts/quantize/measure.sh $DST"
