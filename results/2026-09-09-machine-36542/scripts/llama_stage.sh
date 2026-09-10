#!/bin/bash
# Phase 4 llama.cpp stage: the recipe in docs/baselines.md, logs per run.
R=/workspace/token-rush/results/2026-09-09-machine-36542/llama.cpp
B=/workspace/rivals/llama.cpp/build/bin
M=/workspace/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_M.gguf
DSPARK=/workspace/models/Qwen3.8-27B-DSpark-GGUF/Qwen3.8-27B-DSpark-BF16.gguf
DFLASH=/workspace/models/Qwen3.8-27B-DFlash2-GGUF/Qwen3.8-27B-DFlash2-BF16.gguf
declare -A P
P[essay]="Write a detailed essay on the causes and consequences of the fall of the Western Roman Empire, covering political, economic and military factors."
P[code]="Write a Python implementation of a thread-safe LRU cache with TTL expiry, including unit tests."
P[math]="Solve step by step: A train leaves city A at 60 km/h and another leaves city B, 450 km away, at 90 km/h toward it one hour later. When and where do they meet?"

echo "===== llama-bench tg128 ====="
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
echo "===== llama-cli raw ====="; cli raw
echo "===== llama-cli draft-mtp ====="; cli mtp --spec-type draft-mtp
echo "===== llama-cli draft-mtp n-max 4 ====="; cli mtp4 --spec-type draft-mtp --spec-draft-n-max 4
echo "===== llama-cli draft-dspark ====="; cli dspark --spec-type draft-dspark -md $DSPARK -ngld 999
echo "===== llama-cli draft-dflash ====="; cli dflash --spec-type draft-dflash -md $DFLASH -ngld 999
echo "===== llama-bench depth sweep (q8_0 KV, fa) ====="
$B/llama-bench -m $M -ngl 999 -fa 1 -ctk q8_0 -ctv q8_0 -p 0 -n 128 -d 0,22000,90000,200000 -r 2 2>&1 | tee $R/llama_depth.log | grep -E "^\|"
echo "STAGE DONE"
