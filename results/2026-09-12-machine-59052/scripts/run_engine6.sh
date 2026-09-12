#!/bin/bash
# Sixth queue: after run_engine5.sh, the Marlin 96k raw point under strace.
D=/workspace/token-rush/results/2026-09-12-machine-59052/scripts; R=/workspace/token-rush/results/2026-09-12-machine-59052
until grep -q "ALL DONE" $R/console_run_engine5.txt 2>/dev/null; do sleep 30; done
echo "########## engine_raw_fill2 stage ($(date '+%F %H:%M')) ##########"
bash $D/engine_raw_fill2.sh 2>&1 | tee $R/console_engine_raw_fill2.txt
echo "ALL DONE ($(date '+%F %H:%M'))"
