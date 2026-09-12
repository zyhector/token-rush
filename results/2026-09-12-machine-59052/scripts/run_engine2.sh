#!/bin/bash
# Second queue: the engine's multi-position sweep again, then the llama.cpp MTP prompt-file sweep.
# The first pass of engine_spec_stage.sh under run_all.sh died silently twice (pg19 at 16k, code at
# 64k: no traceback, no OOM event, reproduced fine in the foreground), so this queue runs each stage
# in its own session (setsid) the way the rival servers ran, with expandable segments for the allocator.
D=/workspace/token-rush/results/2026-09-12-machine-59052/scripts; R=/workspace/token-rush/results/2026-09-12-machine-59052
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for s in engine_spec llama_mtp; do
  echo "########## $s stage ($(date '+%F %H:%M')) ##########"
  bash $D/${s}_stage.sh 2>&1 | tee $R/console_$s.txt
done
echo "ALL DONE ($(date '+%F %H:%M'))"
