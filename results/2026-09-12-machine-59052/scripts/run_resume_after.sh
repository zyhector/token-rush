#!/bin/bash
# Waits for the orphaned attempt-3 python (its driver was killed by hand) to finish, then restarts
# the resumable driver, which picks up from the log.
while kill -0 164818 2>/dev/null; do sleep 10; done
echo "########## resume driver restarted ($(date '+%F %H:%M')) ##########"
bash /workspace/token-rush/results/2026-09-12-machine-59052/scripts/engine_spec_resume.sh
