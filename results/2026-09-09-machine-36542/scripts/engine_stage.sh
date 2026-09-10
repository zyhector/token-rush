#!/bin/bash
R=/workspace/token-rush/results/2026-09-09-machine-36542/engine; cd /workspace/token-rush; source /venv/main/bin/activate
M=/workspace/models/Qwen3.8-27B-int4g128-gptq-mse; D=/workspace/models/Qwen3.8-27B-DFlash2
run() { local name=$1; shift; echo "===== $name ====="; python "$@" 2>&1 | grep -v "^\s*$" | tee $R/$name.log | grep -v "Warning\|loaded\|captured"; }
run decode_marlin bench/decode.py --model $M --steps 50 --warmup 10
run decode_triton bench/decode.py --model $M --backend triton --steps 50 --warmup 10
run decode_200k_marlin bench/decode.py --model $M --kv fp8 --max-len 262144 --context 200000 --steps 50 --warmup 10
run decode_200k_triton bench/decode.py --model $M --backend triton --kv fp8 --max-len 262144 --context 200000 --steps 50 --warmup 10
run families_greedy bench/families.py --model $M --dflash-path $D
run families_sampled bench/families.py --model $M --dflash-path $D --temperature 0.7 --top-p 0.9
run spec_context_prose_mtp bench/spec_context.py --model $M --text /workspace/data/prose.txt --draft mtp
run spec_context_prose_dflash bench/spec_context.py --model $M --text /workspace/data/prose.txt --draft dflash --dflash-path $D --no-raw
run spec_context_code_mtp bench/spec_context.py --model $M --text /workspace/data/code.txt --draft mtp
run spec_context_code_dflash bench/spec_context.py --model $M --text /workspace/data/code.txt --draft dflash --dflash-path $D --no-raw
run needle bench/needle.py --model $M --text /workspace/data/prose.txt
run gsm8k bench/quality_gsm8k_engine.py --model $M --out $R/gsm8k_engine.json
echo "STAGE DONE"
