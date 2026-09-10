#!/bin/bash
# eval_done_reaper.sh <probe>... — cancel a probe the moment its eval pass reports 100 %.
# probe_watch.sh is supposed to do this, but it only cancels if pass8_table.py succeeds first, and on a 1,840-task
# probe that table takes longer than the trainer needs to start a SECOND eval pass into eval_step1 — which pollutes
# the trial tree and steals fleet seats from the other probes (2026-09-09).
E=${SNOWBALL_EXP:-/e/fscratch/reformo/lee27/experiments}
LOG=$E/eval_done_reaper.log
echo "$(date) reaper start: $*" >> $LOG
while true; do
  left=0
  for p in "$@"; do
    jid=$(squeue -h -u $USER -n $p -o %i | head -1); [ -z "$jid" ] && continue
    left=$((left+1))
    l=$(ls -t $E/$p/logs/${p}_*.out 2>/dev/null | head -1); [ -z "$l" ] && continue
    if sed 's/\x1b\[[0-9;]*m//g' "$l" | grep -q "Evaluation Progress: 58/58 (100%)"; then
      echo "$(date) $p: eval 100% -> scancel $jid" >> $LOG; scancel "$jid"
    fi
  done
  [ "$left" -eq 0 ] && { echo "$(date) reaper done" >> $LOG; exit 0; }
  sleep 30
done
