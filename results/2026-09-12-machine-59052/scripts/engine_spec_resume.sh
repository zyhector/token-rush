#!/bin/bash
# The remaining multi-position points, resumable: bench/spec_context.py --resume-log skips the
# (context, position) units already in the log and folds their stats into the summaries, so each
# host-side SIGSEGV of the prefill's dequantize path (see engine_spec_{pg19,code}_fill.strace: flaky,
# on the compat and the native libcuda alike, never under gdb) costs only the unit in flight.
# Up to 40 attempts per corpus; the cache is sized to the sweep's largest context (245760 rows, not
# 262144: 0.5 GB less at the allocator's limit) and expandable segments are on — neither changes a
# measured number (decode reads the live length; the graph bucket is the cache size either way).
R=/workspace/token-rush/results/2026-09-12-machine-59052; cd /workspace/token-rush; source /venv/main/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Attempts 3-7 on code/160k all died within three minutes of start, i.e. during the first prefill after
# graph capture, and the same unit passed under gdb: a startup race. Kernels are loaded eagerly from here on.
export CUDA_MODULE_LOADING=EAGER
run() { # corpus contexts positions
  local log=$R/sweep/engine_spec_$1_both.log
  for i in $(seq 1 40); do
    echo "===== engine spec vs context, $1, both drafts, contexts $2, attempt $i ($(date +%H:%M)) ====="
    echo "===== continued $(date '+%F %H:%M'): contexts $2 at positions $3, attempt $i, resuming from this log (engine_spec_resume.sh) =====" >> $log
    python -X faulthandler bench/spec_context.py --text /workspace/data/$1.txt --draft both --contexts $2 --positions $3 --new 512 --max-len 245760 --resume-log $log 2>&1 \
      | grep --line-buffered -v "^\s*$\|Warning" | tee -a $log | grep "summary\|positions\|resuming\|Traceback\|Error\|Fatal\|received signal"
    local st=${PIPESTATUS[0]}
    echo "attempt $i exit status $st ($(date +%H:%M))"
    [ "$st" = "0" ] && return 0
  done
  return 1
}
run pg19 160000,200000,240000 240064,488493,736923,985353,1233783,1482213
run code 64000,96000,128000,160000,200000,240000 240064,472814,705565,938315,1171066,1403817
echo "STAGE DONE ($(date +%H:%M))"
