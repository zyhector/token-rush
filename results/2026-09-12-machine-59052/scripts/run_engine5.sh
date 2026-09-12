#!/bin/bash
# Fifth queue: after run_engine4.sh, the code contexts the engine_spec passes did not reach.
D=/workspace/token-rush/results/2026-09-12-machine-59052/scripts; R=/workspace/token-rush/results/2026-09-12-machine-59052
until grep -q "ALL DONE" $R/console_run_engine4.txt 2>/dev/null; do sleep 30; done
echo "########## engine_spec_fill_code stage ($(date '+%F %H:%M')) ##########"
bash $D/engine_spec_fill_code.sh 2>&1 | tee $R/console_engine_spec_fill_code.txt
echo "ALL DONE ($(date '+%F %H:%M'))"
