#!/bin/bash
# Follow-up queue: waits for run_all.sh, then re-runs the llama.cpp MTP prompt-file sweep.
D=/workspace/token-rush/results/2026-09-12-machine-59052/scripts; R=/workspace/token-rush/results/2026-09-12-machine-59052
until grep -q "ALL DONE" $R/console_run_all.txt 2>/dev/null; do sleep 30; done
echo "########## llama_mtp stage ($(date '+%F %H:%M')) ##########"
bash $D/llama_mtp_stage.sh 2>&1 | tee $R/console_llama_mtp.txt
echo "AFTER DONE ($(date '+%F %H:%M'))"
