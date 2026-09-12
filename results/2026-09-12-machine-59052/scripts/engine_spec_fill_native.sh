#!/bin/bash
# The remaining multi-position points, run against the host's own libcuda (595.84) instead of the
# CUDA 13.3 forward-compatibility shim (610.57.04) that this image puts first on the loader path.
# Every silent death of the sweep so far was a host-side SIGSEGV inside plain torch ops on the
# prefill's dequantize path (strace + faulthandler in engine_spec_{pg19,code}_fill.strace and the
# logs), flaky per input, on a driver shim NVIDIA supports only on datacenter GPUs; torch cu130
# needs no shim on a CUDA-13.2 driver. Same script, same positions, appended to the same logs.
R=/workspace/token-rush/results/2026-09-12-machine-59052; cd /workspace/token-rush; source /venv/main/bin/activate
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu
run() { # corpus contexts positions
  echo "===== engine spec vs context, $1, both drafts, contexts $2, native libcuda ($(date +%H:%M)) ====="
  echo "===== continued $(date '+%F %H:%M'): contexts $2 at positions $3, native libcuda 595.84 (engine_spec_fill_native.sh) =====" >> $R/sweep/engine_spec_$1_both.log
  python -X faulthandler bench/spec_context.py --text /workspace/data/$1.txt --draft both --contexts $2 --positions $3 --new 512 2>&1 \
    | grep --line-buffered -v "^\s*$\|Warning" | tee -a $R/sweep/engine_spec_$1_both.log | grep "summary\|positions\|Traceback\|Error\|Fatal\|received signal"
  echo "pipeline status: ${PIPESTATUS[*]} ($(date +%H:%M))"
}
run pg19 160000,200000,240000 240064,488493,736923,985353,1233783,1482213
run code 64000,96000,128000,160000,200000,240000 240064,472814,705565,938315,1171066,1403817
echo "STAGE DONE ($(date +%H:%M))"
