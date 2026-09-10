#!/bin/bash
# wb_sync_loop.sh <run>... — every 30 min, sync each run offline W&B dir to entity lukedhlee-marin (live runs need --no-skip-synced).
# Runs in Jupiter tmux `wb_sync`; log experiments/wb_sync.log. Session 99a1d5ec 2026-09-05.
E=/e/fscratch/reformo/lee27/experiments; WB=/e/project1/transfernetx/lee27/code/envs/snowball/bin/wandb
set -a; source /e/fscratch/reformo/lee27/keys/secrets.env; set +a
export OMP_NUM_THREADS=1
while true; do
  for r in "$@"; do
    d=$E/$r/wandb/wandb; [ -d "$d" ] || { echo "$(date +%F_%T) $r: no wandb dir" >> $E/wb_sync.log; continue; }
    for o in $(ls -d $d/offline-run-* 2>/dev/null); do
      res=$(cd $d && WANDB_MODE=online timeout 900 $WB sync --no-skip-synced --entity lukedhlee-marin --project jupiter-snowball-r2egym $(basename $o) 2>&1 | grep -o "wandb.ai/[^ ]*\|Skipped.*\|unexpected EOF\|Error.*" | head -2 | tr "\n" " ")
      echo "$(date +%F_%T) $r $(basename $o): $res" >> $E/wb_sync.log
    done
  done
  sleep 1800
done
