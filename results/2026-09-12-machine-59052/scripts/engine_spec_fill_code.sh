#!/bin/bash
# The code contexts the engine_spec passes did not reach: both passes died silently during the
# 64k prefill at the first position, after the 32k summary (no signal logged by the script's own
# SIGTERM/SIGHUP/SIGINT handler in the second pass). 64k..240k at the same six positions,
# appended to the same log, under strace's signal tracing, exit status recorded.
R=/workspace/token-rush/results/2026-09-12-machine-59052; cd /workspace/token-rush; source /venv/main/bin/activate
POS=240064,472814,705565,938315,1171066,1403817
echo "===== engine spec vs context, code, both drafts, 64k..240k at the same positions ($(date +%H:%M)) ====="
echo "===== continued $(date '+%F %H:%M'): contexts 64000..240000 at positions $POS (engine_spec_fill_code.sh) =====" >> $R/sweep/engine_spec_code_both.log
strace -f -e trace=none -e signal=all -o $R/sweep/engine_spec_code_fill.strace \
  python bench/spec_context.py --text /workspace/data/code.txt --draft both --contexts 64000,96000,128000,160000,200000,240000 --positions $POS --new 512 2>&1 \
  | grep --line-buffered -v "^\s*$\|Warning" | tee -a $R/sweep/engine_spec_code_both.log | grep "summary\|positions\|Traceback\|Error\|received signal"
echo "pipeline status: ${PIPESTATUS[*]} ($(date +%H:%M))"
echo "STAGE DONE ($(date +%H:%M))"
