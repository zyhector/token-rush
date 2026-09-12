#!/bin/bash
# Engine raw decode vs. context, ten points, both GEMV backends (random-token prefill, fp8 KV).
R=/workspace/token-rush/results/2026-09-12-machine-59052/sweep; cd /workspace/token-rush; source /venv/main/bin/activate
for b in triton marlin; do
  echo "===== engine raw, $b ($(date +%H:%M)) ====="
  : > $R/engine_raw_$b.log
  for c in 0 8000 16000 32000 64000 96000 128000 160000 200000 240000; do
    python bench/decode.py --backend $b --kv fp8 --max-len 262144 --context $c --steps 30 --warmup 5 2>&1 | grep -v "^\s*$" | tee -a $R/engine_raw_$b.log | grep "decode at"
  done
done
echo "STAGE DONE"
