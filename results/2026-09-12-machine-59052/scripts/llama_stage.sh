#!/bin/bash
# llama.cpp stage, machine 59052: the Phase 4 recipe (docs/baselines.md) plus the ten-point
# depth sweep and the built-in MTP on the PG-19 prompt files of the multi-position protocol.
R=/workspace/token-rush/results/2026-09-12-machine-59052/llama.cpp
B=/workspace/rivals/llama.cpp/build/bin
M=/workspace/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_M.gguf
DSPARK=/workspace/models/Qwen3.8-27B-DSpark-GGUF/Qwen3.8-27B-DSpark-BF16.gguf
DFLASH=/workspace/models/Qwen3.8-27B-DFlash2-GGUF/Qwen3.8-27B-DFlash2-BF16.gguf
PR=/workspace/data/prompts/pg19
declare -A P
P[essay]="Write a detailed essay on the causes and consequences of the fall of the Western Roman Empire, covering political, economic and military factors."
P[code]="Write a Python implementation of a thread-safe LRU cache with TTL expiry, including unit tests."
P[math]="Solve step by step: A train leaves city A at 60 km/h and another leaves city B, 450 km away, at 90 km/h toward it one hour later. When and where do they meet?"

echo "===== llama-bench tg128 ($(date +%H:%M)) ====="
$B/llama-bench -m $M -ngl 999 -p 512 -n 128 -r 3 2>&1 | tee $R/llama_bench_tg128.log | grep -E "^\|"

cli() {  # name, extra args...
  local name=$1; shift
  for p in essay code math; do
    for rep in 1 2 3; do
      $B/llama-cli -m $M -ngl 999 -c 4096 -n 200 --temp 0 -p "${P[$p]}" --single-turn --no-warmup "$@" \
        > $R/llama_cli_${name}_${p}_r${rep}.log 2>&1 < /dev/null
      echo "$p $name r$rep: $(grep -o '\[ Prompt:.*\]' $R/llama_cli_${name}_${p}_r${rep}.log | tail -1)"
    done
  done
}
echo "===== llama-cli raw ($(date +%H:%M)) ====="; cli raw
echo "===== llama-cli draft-mtp ====="; cli mtp --spec-type draft-mtp
echo "===== llama-cli draft-mtp n-max 4 ====="; cli mtp4 --spec-type draft-mtp --spec-draft-n-max 4
echo "===== llama-cli draft-dspark ====="; cli dspark --spec-type draft-dspark -md $DSPARK -ngld 999
echo "===== llama-cli draft-dflash ====="; cli dflash --spec-type draft-dflash -md $DFLASH -ngld 999

echo "===== llama-bench depth sweep, ten points (q8_0 KV, fa) ($(date +%H:%M)) ====="
$B/llama-bench -m $M -ngl 999 -fa 1 -ctk q8_0 -ctv q8_0 -p 0 -n 128 -d 0,8000,16000,32000,64000,96000,128000,160000,200000,240000 -r 2 2>&1 | tee $R/llama_depth.log | grep -E "^\|"

echo "===== llama-server draft-mtp on the PG-19 prompt files: raw completion, 512 new tokens, q8_0 KV, fa ($(date +%H:%M)) ====="
# llama-cli wraps prompts in the chat template (the model answers the book instead of continuing
# it), so the protocol runs through llama-server's /completion (scripts/rivals/llama_server_bench.py)
setsid nohup $B/llama-server -m $M -ngl 999 -fa 1 -ctk q8_0 -ctv q8_0 -c 245760 -np 1 --spec-type draft-mtp --port 8091 \   # 8080 is Jupyter's on this image: the first attempt bound nothing (llama_mtp_stage.sh re-ran this part)
  > $R/server_mtp_pg19.log 2>&1 < /dev/null & SPID=$!
for i in $(seq 1 120); do sleep 5; curl -sf http://127.0.0.1:8091/health >/dev/null 2>&1 && { echo "server up after $((i*5))s"; break; }; done
python3 scripts/rivals/llama_server_bench.py files $(ls $PR/N_*_p_*.txt | sort -t_ -k2,2n -k4,4n) --n 512 --url http://127.0.0.1:8091 2>&1 | tee $R/llama_mtp_pg19.log
kill -- -$SPID 2>/dev/null; sleep 3; pkill -x llama-server; sleep 3
echo "STAGE DONE ($(date +%H:%M))"
