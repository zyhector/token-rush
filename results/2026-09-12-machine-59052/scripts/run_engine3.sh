#!/bin/bash
# Third queue: after run_engine2.sh, the two missing Marlin raw points.
D=/workspace/token-rush/results/2026-09-12-machine-59052/scripts; R=/workspace/token-rush/results/2026-09-12-machine-59052
until grep -q "ALL DONE" $R/console_run_engine2.txt 2>/dev/null; do sleep 30; done
echo "########## engine_raw_fill stage ($(date '+%F %H:%M')) ##########"
bash $D/engine_raw_fill.sh 2>&1 | tee $R/console_engine_raw_fill.txt
echo "ALL DONE ($(date '+%F %H:%M'))"
