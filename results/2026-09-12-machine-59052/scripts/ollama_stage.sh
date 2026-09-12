#!/bin/bash
R=/workspace/token-rush/results/2026-09-12-machine-59052/ollama; cd /workspace/token-rush
setsid nohup ollama serve > $R/serve.log 2>&1 < /dev/null & OPID=$!
sleep 5; ollama --version
ollama pull qwen3.8:27b 2>&1 | tail -1
ollama show qwen3.8:27b | head -12
python3 scripts/rivals/ollama_bench.py qwen3.8:27b 2>&1 | tee $R/results.log
ollama stop qwen3.8:27b; sleep 3; kill -- -$OPID; sleep 5
nvidia-smi --query-gpu=memory.used --format=csv,noheader
echo "STAGE DONE ($(date +%H:%M))"
