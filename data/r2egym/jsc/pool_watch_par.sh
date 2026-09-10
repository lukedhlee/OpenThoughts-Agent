#!/bin/bash
# pool_watch_par.sh <prefix> [shards=8] — parallel successor of pool_watch.sh: every 60 s, for each shard job whose log has
# the eval block, start pass8_table.py in the background (lock $d/.tabling); once a shard's pass8_summary.json exists,
# scancel its job; once the job is gone, compact the tree in the background. Idempotent; exits when all shards are tabled+compacted.
PREFIX=$1; N=${2:-8}
EXP=/e/fscratch/reformo/lee27/experiments; PT=/e/project1/transfernetx/lee27/code/snowball/pass8_table.py
LOG=$EXP/${PREFIX}_watch.log
echo "$(date) pool_watch_par start prefix=$PREFIX shards=$N" >> $LOG
while true; do
  remaining=0
  for i in $(seq 0 $((N-1))); do
    name=${PREFIX}_s$i; d=$EXP/$name; marker=$d/pass8_summary.json
    jid=$(squeue -h -u $USER -n $name -o %i | head -1)
    if [ -f "$marker" ]; then
      if [ -n "$jid" ]; then scancel $jid && echo "$(date) $name: tabled -> cancelled $jid" >> $LOG; remaining=$((remaining+1)); continue; fi
      [ -f "$d/.compacted" ] && continue
      remaining=$((remaining+1))
      if [ ! -f "$d/.compacting" ]; then
        touch "$d/.compacting"; echo "$(date) $name: compacting" >> $LOG
        ( bash /e/project1/transfernetx/lee27/code/snowball/compact_traces.sh $d/$name/trace_jobs >> $LOG 2>&1; mv "$d/.compacting" "$d/.compacted" ) &
      fi
      continue
    fi
    remaining=$((remaining+1))
    l=$(ls -t $d/logs/${name}_*.out 2>/dev/null | head -1); [ -z "$l" ] && continue
    if grep -q "WANDB_MIRROR kind=eval" "$l" && [ ! -f "$d/.tabling" ]; then
      touch "$d/.tabling"; echo "$(date) $name: eval block present (job $jid) -> tabling in background" >> $LOG
      ( python3 $PT $d/$name/trace_jobs --k 8 --out $d/pass8 >> $LOG 2>&1 && echo "$(date) $name: table written" >> $LOG; rm -f "$d/.tabling" ) &
    fi
  done
  [ $remaining -eq 0 ] && { echo "$(date) all shards tabled+compacted; exiting" >> $LOG; exit 0; }
  sleep 60
done
