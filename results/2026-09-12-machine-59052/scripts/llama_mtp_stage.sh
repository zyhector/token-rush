#!/bin/bash
# The PG-19 prompt-file sweep of llama.cpp's built-in MTP, re-run on port 8091 (8080 is Jupyter's
# on this image; the first attempt in llama_stage.sh bound nothing and the bench hit the wrong
# server). Same server arguments otherwise: raw completion through /completion, 512 new tokens.
R=/workspace/token-rush/results/2026-09-12-machine-59052/llama.cpp; cd /workspace/token-rush
B=/workspace/rivals/llama.cpp/build/bin
M=/workspace/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_M.gguf
PR=/workspace/data/prompts/pg19
echo "===== llama-server draft-mtp on the PG-19 prompt files: raw completion, 512 new tokens, q8_0 KV, fa ($(date +%H:%M)) ====="
setsid nohup $B/llama-server -m $M -ngl 999 -fa 1 -ctk q8_0 -ctv q8_0 -c 245760 -np 1 --spec-type draft-mtp --port 8091 \
  > $R/server_mtp_pg19.log 2>&1 < /dev/null & SPID=$!
for i in $(seq 1 120); do sleep 5; curl -sf http://127.0.0.1:8091/health >/dev/null 2>&1 && { echo "server up after $((i*5))s"; break; }; done
python3 scripts/rivals/llama_server_bench.py files $(ls $PR/N_*_p_*.txt | sort -t_ -k2,2n -k4,4n) --n 512 --url http://127.0.0.1:8091 2>&1 | tee $R/llama_mtp_pg19.log
kill -- -$SPID 2>/dev/null; sleep 3; pkill -x llama-server; sleep 3
echo "STAGE DONE ($(date +%H:%M))"
