#!/bin/bash
R=/workspace/token-rush/results/2026-09-09-machine-36542/sglang; V=/workspace/venvs/sglang/bin; cd /workspace/token-rush
M=/workspace/models/Qwen3.8-27B-NVFP4
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True setsid nohup $V/python -m sglang.launch_server --model-path $M --trust-remote-code \
  --kv-cache-dtype fp8_e4m3 --mamba-ssm-dtype float32 --max-running-requests 1 --max-mamba-cache-size 6 \
  --cuda-graph-max-bs 1 --disable-prefill-cuda-graph --port 30000 --context-length 262144 \
  --chunked-prefill-size 4096 --max-prefill-tokens 4096 --mem-fraction-static 0.88 --max-total-tokens 215000 > $R/server_ctx.log 2>&1 < /dev/null &
SPID=$!
for i in $(seq 1 240); do sleep 5; curl -sf http://127.0.0.1:30000/health >/dev/null 2>&1 && { echo "server ctx up after $((i*5))s"; break; }; done
$V/python scripts/rivals/sglang_bench.py context 0 22000 90000 200000 2>&1 | tee $R/results_context.log
kill -- -$SPID 2>/dev/null; pkill -f "sglang.launch_server" 2>/dev/null
for i in $(seq 1 60); do sleep 2; [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 2000 ] && { echo "gpu freed"; break; }; done
echo "STAGE DONE"
