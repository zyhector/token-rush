#!/bin/bash
# The two Marlin-layout raw points (96k, 200k) whose bench/decode.py runs died silently in
# engine_raw_sweep.sh (no line in the log; same silent death as the first engine_spec pass).
R=/workspace/token-rush/results/2026-09-12-machine-59052/sweep; cd /workspace/token-rush; source /venv/main/bin/activate
echo "===== engine raw, marlin, 96k and 200k again ($(date +%H:%M)) ====="
for c in 96000 200000; do
  python bench/decode.py --backend marlin --kv fp8 --max-len 262144 --context $c --steps 30 --warmup 5 2>&1 | grep -v "^\s*$" | tee -a $R/engine_raw_marlin.log | grep "decode at"
done
echo "STAGE DONE ($(date +%H:%M))"
