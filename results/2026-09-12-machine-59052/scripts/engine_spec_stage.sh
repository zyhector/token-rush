#!/bin/bash
# Engine speculative decode vs. context under the multi-position protocol (bench/spec_context.py):
# PG-19 prose and torch code, six positions, 512 greedy tokens, both drafts from one shared
# prefill, raw at every position too. Then needle retrieval at 128k and 256k.
R=/workspace/token-rush/results/2026-09-12-machine-59052; cd /workspace/token-rush; source /venv/main/bin/activate
C=0,8000,16000,32000,64000,96000,128000,160000,200000,240000
for t in pg19 code; do
  echo "===== engine spec vs context, $t, both drafts, 6 positions x 512 ($(date +%H:%M)) ====="
  python bench/spec_context.py --text /workspace/data/$t.txt --draft both --contexts $C --positions 6 --new 512 2>&1 | grep -v "^\s*$\|Warning" | tee $R/sweep/engine_spec_${t}_both.log | grep "summary\|positions\|Traceback\|Error"
done
echo "===== needle ($(date +%H:%M)) ====="
python bench/needle.py --text /workspace/data/prose.txt 2>&1 | grep -v "^\s*$\|Warning" | tee $R/engine/needle.log | grep "context\|RETRIEVED\|MISSED"
echo "STAGE DONE ($(date +%H:%M))"
