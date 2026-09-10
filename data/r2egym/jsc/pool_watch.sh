#!/bin/bash
# pool_watch.sh <prefix> [shards=8] [dry]
# Every 2 min: for each shard job <prefix>_s<i>, once its log has the eval block (WANDB_MIRROR kind=eval),
# write the pass@8 table (pass8_table.py --out <exp>/pass8) and scancel the job so the automatic train
# step + post-eval do not burn fleet slots. Idempotent; exits when every shard is tabled and gone.
PREFIX=$1; N=${2:-8}; DRY=${3:-}
EXP=/e/fscratch/reformo/lee27/experiments; PT=/e/project1/transfernetx/lee27/code/snowball/pass8_table.py
LOG=$EXP/${PREFIX}_watch.log
echo "$(date) pool_watch start prefix=$PREFIX shards=$N dry=$DRY" >> $LOG
while true; do
  remaining=0
  for i in $(seq 0 $((N-1))); do
    name=${PREFIX}_s$i; d=$EXP/$name; done_marker=$d/pass8_summary.json
    jid=$(squeue -h -u $USER -n $name -o %i | head -1)
    [ -f "$done_marker" ] && [ -z "$jid" ] && [ -f "$d/.compacted" ] && continue
    remaining=$((remaining+1))
    l=$(ls -t $d/logs/${name}_*.out 2>/dev/null | head -1); [ -z "$l" ] && continue
    if grep -q "WANDB_MIRROR kind=eval" "$l" && [ ! -f "$done_marker" ]; then
      echo "$(date) $name: eval block present (job $jid) -> table + scancel" >> $LOG
      if [ -n "$DRY" ]; then echo "DRY: python3 $PT $d/$name/trace_jobs --k 8 --out $d/pass8; scancel $jid" >> $LOG; continue; fi
      python3 $PT $d/$name/trace_jobs --k 8 --out $d/pass8 >> $LOG 2>&1 && [ -n "$jid" ] && scancel $jid && echo "$(date) $name: cancelled $jid" >> $LOG
    fi
    # compaction: once tabled AND the job is gone, shrink the trace tree ~10x (dup results, indentation, giant panes) in the background
    if [ -f "$done_marker" ] && [ -z "$jid" ] && [ ! -f "$d/.compacted" ] && [ ! -f "$d/.compacting" ]; then
      touch "$d/.compacting"; echo "$(date) $name: compacting" >> $LOG
      ( bash /e/project1/transfernetx/lee27/code/snowball/compact_traces.sh $d/$name/trace_jobs >> $LOG 2>&1; mv "$d/.compacting" "$d/.compacted" ) &
    fi
  done
  [ $remaining -eq 0 ] && { echo "$(date) all shards tabled; exiting" >> $LOG; exit 0; }
  sleep 120
done
