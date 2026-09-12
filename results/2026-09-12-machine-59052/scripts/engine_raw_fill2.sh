#!/bin/bash
# The Marlin-layout raw point at 96k, whose bench/decode.py died silently twice during the prefill
# (engine_raw_sweep.sh and engine_raw_fill.sh: weights loaded, graphs captured, no prefill line).
# Under strace's signal tracing, exit status recorded.
R=/workspace/token-rush/results/2026-09-12-machine-59052/sweep; cd /workspace/token-rush; source /venv/main/bin/activate
echo "===== engine raw, marlin, 96k, third attempt under strace ($(date +%H:%M)) ====="
strace -f -e trace=none -e signal=all -o $R/engine_raw_marlin_96k.strace \
  python -X faulthandler bench/decode.py --backend marlin --kv fp8 --max-len 262144 --context 96000 --steps 30 --warmup 5 2>&1 \
  | grep --line-buffered -v "^\s*$" | tee -a $R/engine_raw_marlin.log | grep "decode at\|Traceback\|Error\|Fatal"
echo "pipeline status: ${PIPESTATUS[*]} ($(date +%H:%M))"
echo "STAGE DONE ($(date +%H:%M))"
