#!/bin/bash
E=/e/fscratch/reformo/lee27/experiments; LOG=$E/compact_all.log; echo "$(date) start" >> $LOG
running=$(squeue -h -u $USER -o %j | sort -u)
for T in $E/*/*/trace_jobs; do
  X=$(basename $(dirname $(dirname $T)))
  echo "$running" | grep -qx "$X" && { echo "skip live $X" >> $LOG; continue; }
  case $X in snowball_smoke64k*|snowball_probe_val_b60k_parity) echo "skip $X" >> $LOG; continue;; esac
  [ -f $E/$X/.compacted ] && continue
  bash /e/project1/transfernetx/lee27/code/snowball/compact_traces.sh $T >> $LOG 2>&1 && touch $E/$X/.compacted
done
echo "$(date) done" >> $LOG
