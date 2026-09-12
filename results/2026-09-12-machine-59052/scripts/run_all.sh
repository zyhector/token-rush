#!/bin/bash
# The whole measurement queue on machine 59052, one sitting: rivals first, then the engine's
# multi-position speculative sweep and the needle. The engine raw sweep (engine_raw_sweep.sh)
# ran before this from the same shell.
D=/workspace/token-rush/results/2026-09-12-machine-59052/scripts; R=/workspace/token-rush/results/2026-09-12-machine-59052
until grep -q "STAGE DONE" $R/sweep/console_engine_raw.txt 2>/dev/null; do sleep 20; done
for s in llama vllm sglang exl3 ollama engine_spec; do
  echo "########## $s stage ($(date '+%F %H:%M')) ##########"
  bash $D/${s}_stage.sh 2>&1 | tee $R/console_$s.txt
done
echo "ALL DONE ($(date '+%F %H:%M'))"
