#!/bin/bash
# The PG-19 contexts the second engine_spec pass did not reach (its process died silently again
# at 160k, after the 128k summary): 160k / 200k / 240k at the same six positions, appended to the
# same log. Run under strace's signal tracing so that a third death names its sender, with the
# pipeline's exit status recorded (bash does not report a signal death of a non-final pipe member).
R=/workspace/token-rush/results/2026-09-12-machine-59052; cd /workspace/token-rush; source /venv/main/bin/activate
POS=240064,488493,736923,985353,1233783,1482213
echo "===== engine spec vs context, pg19, both drafts, 160k/200k/240k at the same positions ($(date +%H:%M)) ====="
echo "===== continued $(date '+%F %H:%M'): contexts 160000,200000,240000 at positions $POS (engine_spec_fill.sh) =====" >> $R/sweep/engine_spec_pg19_both.log
strace -f -e trace=none -e signal=all -o $R/sweep/engine_spec_pg19_fill.strace \
  python bench/spec_context.py --text /workspace/data/pg19.txt --draft both --contexts 160000,200000,240000 --positions $POS --new 512 2>&1 \
  | grep --line-buffered -v "^\s*$\|Warning" | tee -a $R/sweep/engine_spec_pg19_both.log | grep "summary\|positions\|Traceback\|Error\|received signal"
echo "pipeline status: ${PIPESTATUS[*]} ($(date +%H:%M))"
echo "STAGE DONE ($(date +%H:%M))"
