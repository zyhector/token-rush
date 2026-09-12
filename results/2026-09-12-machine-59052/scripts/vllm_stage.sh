#!/bin/bash
# vLLM stage, machine 59052: raw + the ten-point context sweep at --max-model-len 245000,
# then every speculative path it has (MTP, DSpark, DFlash2) on the three short prompts.
R=/workspace/token-rush/results/2026-09-12-machine-59052/vllm; V=/workspace/venvs/vllm/bin; cd /workspace/token-rush
M=/workspace/models/Qwen3.8-27B-NVFP4
serve() { local name=$1; shift
  setsid nohup $V/vllm serve $M --trust-remote-code --kv-cache-dtype fp8 --max-num-seqs 1 --port 8000 "$@" > $R/server_$name.log 2>&1 < /dev/null &
  SPID=$!
  for i in $(seq 1 360); do sleep 5; curl -sf http://127.0.0.1:8000/v1/models >/dev/null 2>&1 && { echo "server $name up after $((i*5))s"; return 0; }; kill -0 $SPID 2>/dev/null || { echo "server $name DIED"; tail -5 $R/server_$name.log; return 1; }; done
  echo "server $name TIMEOUT"; return 1
}
stop() { kill -- -$SPID 2>/dev/null; for i in $(seq 1 60); do sleep 2; [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 2000 ] && { echo "gpu freed"; return; }; done; pkill -9 -f "^$V/vllm"; pkill -9 -f "VLLM::EngineCore"; sleep 5; }
echo "===== vLLM raw + context, ten points (245k) ($(date +%H:%M)) ====="
if serve ctx --max-model-len 245000 --max-num-batched-tokens 8192 --gpu-memory-utilization 0.95; then
  $V/python scripts/rivals/vllm_bench.py raw 2>&1 | tee $R/results_raw.log
  $V/python scripts/rivals/vllm_bench.py context 0 8000 16000 32000 64000 96000 128000 160000 200000 240000 2>&1 | tee $R/results_context.log
fi; stop
echo "===== vLLM MTP 1 ($(date +%H:%M)) ====="
if serve mtp1 --max-model-len 32768 --gpu-memory-utilization 0.9 --speculative-config '{"method":"mtp","num_speculative_tokens":1}'; then
  $V/python scripts/rivals/vllm_bench.py raw 2>&1 | tee $R/results_mtp1.log
  grep -h "acceptance\|Accept" $R/server_mtp1.log | tail -3
fi; stop
echo "===== vLLM DSpark ($(date +%H:%M)) ====="
if serve dspark --max-model-len 8192 --gpu-memory-utilization 0.92 --speculative-config '{"method":"dspark","model":"/workspace/models/Qwen3.8-27B-DSpark-ct","num_speculative_tokens":7}'; then
  $V/python scripts/rivals/vllm_bench.py raw 2>&1 | tee $R/results_dspark.log
  grep -h "acceptance\|Accept" $R/server_dspark.log | tail -3
fi; stop
echo "===== vLLM DFlash2 ($(date +%H:%M)) ====="
if serve dflash --max-model-len 8192 --gpu-memory-utilization 0.92 --speculative-config '{"method":"dflash","model":"/workspace/models/Qwen3.8-27B-DFlash2","num_speculative_tokens":8}'; then
  $V/python scripts/rivals/vllm_bench.py raw 2>&1 | tee $R/results_dflash.log
  grep -h "acceptance\|Accept" $R/server_dflash.log | tail -3
fi; stop
echo "STAGE DONE ($(date +%H:%M))"
