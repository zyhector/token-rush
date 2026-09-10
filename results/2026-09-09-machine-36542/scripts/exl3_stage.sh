#!/bin/bash
R=/workspace/token-rush/results/2026-09-09-machine-36542/exllamav3; P=/workspace/venvs/exl3/bin/python
M=/workspace/models/Qwen3.8-27B-exl3-4.0; cd /workspace/token-rush
echo "===== exl3 raw ====="; $P scripts/rivals/exl3_bench.py $M raw 2>&1 | tee $R/exl3_raw.log | grep -v Warning
echo "===== exl3 context ====="; $P scripts/rivals/exl3_bench.py $M context 0 22000 90000 200000 2>&1 | tee $R/exl3_context.log | grep -v Warning
echo "===== exl3 mtp 1 ====="; $P scripts/rivals/exl3_bench.py $M mtp --draft-tokens 1 2>&1 | tee $R/exl3_mtp1.log | grep -v Warning
echo "===== exl3 mtp 2 ====="; $P scripts/rivals/exl3_bench.py $M mtp --draft-tokens 2 2>&1 | tee $R/exl3_mtp2.log | grep -v Warning
echo "STAGE DONE"
