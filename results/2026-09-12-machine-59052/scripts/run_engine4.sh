#!/bin/bash
# Fourth queue: after run_engine3.sh, the PG-19 contexts the second pass did not reach.
D=/workspace/token-rush/results/2026-09-12-machine-59052/scripts; R=/workspace/token-rush/results/2026-09-12-machine-59052
until grep -q "ALL DONE" $R/console_run_engine3.txt 2>/dev/null; do sleep 30; done
echo "########## engine_spec_fill stage ($(date '+%F %H:%M')) ##########"
bash $D/engine_spec_fill.sh 2>&1 | tee $R/console_engine_spec_fill.txt
echo "ALL DONE ($(date '+%F %H:%M'))"
