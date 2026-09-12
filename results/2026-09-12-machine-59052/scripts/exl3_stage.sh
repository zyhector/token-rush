#!/bin/bash
# ExLlamaV3 stage, machine 59052: raw, the context sweep (240k does not fit next to the
# weights), chained MTP with 1 and 2 draft tokens, and MTP x2 vs. context (random tokens,
# its harness; stops where the draft no longer fits).
R=/workspace/token-rush/results/2026-09-12-machine-59052/exllamav3; P=/workspace/venvs/exl3/bin/python
M=/workspace/models/Qwen3.8-27B-exl3-4.0; cd /workspace/token-rush
echo "===== exl3 raw ($(date +%H:%M)) ====="; $P scripts/rivals/exl3_bench.py $M raw 2>&1 | tee $R/exl3_raw.log | grep -v Warning
echo "===== exl3 context ====="; $P scripts/rivals/exl3_bench.py $M context 0 8000 16000 32000 64000 96000 128000 160000 200000 2>&1 | tee $R/exl3_context.log | grep -v Warning
echo "===== exl3 mtp 1 ====="; $P scripts/rivals/exl3_bench.py $M mtp --draft-tokens 1 2>&1 | tee $R/exl3_mtp1.log | grep -v Warning
echo "===== exl3 mtp 2 ====="; $P scripts/rivals/exl3_bench.py $M mtp --draft-tokens 2 2>&1 | tee $R/exl3_mtp2.log | grep -v Warning
echo "===== exl3 mtp 2 vs context ====="; $P scripts/rivals/exl3_bench.py $M mtp --draft-tokens 2 0 8000 16000 32000 64000 96000 128000 2>&1 | tee $R/exl3_mtp_context.log | grep -v Warning
echo "STAGE DONE ($(date +%H:%M))"
