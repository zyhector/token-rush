#!/bin/bash
# SGLang stage, machine 59052: raw + the ten-point context sweep (4096-token chunks, a 245k
# pool), then DSpark on the three short prompts and on the PG-19 prompt files that fit its
# 30k window (contexts 0 / 8k / 16k, six positions each).
R=/workspace/token-rush/results/2026-09-12-machine-59052/sglang; V=/workspace/venvs/sglang/bin; cd /workspace/token-rush
M=/workspace/models/Qwen3.8-27B-NVFP4; PR=/workspace/data/prompts/pg19
serve() { local name=$1; shift
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True setsid nohup $V/python -m sglang.launch_server --model-path $M --trust-remote-code \
    --kv-cache-dtype fp8_e4m3 --mamba-ssm-dtype float32 --max-running-requests 1 --max-mamba-cache-size 6 \
    --cuda-graph-max-bs 1 --disable-prefill-cuda-graph --port 30000 "$@" > $R/server_$name.log 2>&1 < /dev/null &
  SPID=$!
  for i in $(seq 1 360); do sleep 5; curl -sf http://127.0.0.1:30000/health >/dev/null 2>&1 && { echo "server $name up after $((i*5))s"; return 0; }; grep -q "Traceback\|Error" $R/server_$name.log && ! pgrep -f "sglang.launch_server" >/dev/null && { echo "server $name DIED"; tail -5 $R/server_$name.log; return 1; }; done
  echo "server $name TIMEOUT"; return 1; }
stop() { kill -- -$SPID 2>/dev/null; pkill -f "sglang.launch_server" 2>/dev/null; for i in $(seq 1 60); do sleep 2; [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 2000 ] && { echo "gpu freed"; return; }; done; pkill -9 -f "sglang"; sleep 5; }
echo "===== SGLang raw + context, ten points ($(date +%H:%M)) ====="
if serve ctx --context-length 262144 --chunked-prefill-size 4096 --max-prefill-tokens 4096 --mem-fraction-static 0.88 --max-total-tokens 245000; then
  $V/python scripts/rivals/sglang_bench.py raw 2>&1 | tee $R/results_raw.log
  $V/python scripts/rivals/sglang_bench.py context 0 8000 16000 32000 64000 96000 128000 160000 200000 240000 2>&1 | tee $R/results_context.log
fi; stop
echo "===== SGLang + DSpark ($(date +%H:%M)) ====="
if SGLANG_RAGGED_VERIFY_MODE=static serve spec --context-length 32768 --chunked-prefill-size 4096 --max-prefill-tokens 4096 --mem-fraction-static 0.92 --max-total-tokens 30000 \
    --speculative-algorithm DSPARK --speculative-draft-model-path /workspace/models/Qwen3.8-27B-DSpark \
    --speculative-draft-model-quantization unquant --speculative-draft-attention-backend flashinfer \
    --speculative-dspark-block-size 7 --speculative-num-steps 1 --speculative-eagle-topk 1; then
  $V/python scripts/rivals/sglang_bench.py spec --n 1024 2>&1 | tee $R/results_spec.log
  $V/python scripts/rivals/sglang_bench.py files --n 512 $(ls $PR/N_0_p_*.txt $PR/N_8000_p_*.txt $PR/N_16000_p_*.txt | sort -t_ -k2,2n -k4,4n) 2>&1 | tee $R/results_spec_pg19.log
fi; stop
echo "STAGE DONE ($(date +%H:%M))"
