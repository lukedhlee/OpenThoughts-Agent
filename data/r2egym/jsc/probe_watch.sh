#!/bin/bash
# probe_watch.sh <run>... — pool_watch.sh for arbitrarily named eval-only probes (no _s<i> shard pattern).
# Every 120 s: once a run's log has the eval block (WANDB_MIRROR kind=eval) and no pass8_summary.json exists yet,
# write the pass@8 table (pass8_table.py --out <exp>/<run>/pass8) and scancel the job so the automatic train step +
# post-eval do not burn fleet seats. Idempotent; exits when every run is tabled and gone. Log: <exp>/probe_watch_<first run>.log
EXP=${SNOWBALL_EXP:-/e/fscratch/reformo/lee27/experiments}; PT=/e/project1/transfernetx/lee27/code/snowball/pass8_table.py
LOG=$EXP/probe_watch_$1.log
echo "$(date) probe_watch start runs=$*" >> $LOG
while true; do
  remaining=0
  for name in "$@"; do
    d=$EXP/$name; done_marker=$d/pass8_summary.json
    jid=$(squeue -h -u $USER -n $name -o %i | head -1)
    if [ -f "$done_marker" ] && [ -z "$jid" ]; then
      # tabled and gone: archive the trace tree as ONE inode (tar, verified) and delete it — probes are not store-backed
      TJ=$d/$name/trace_jobs
      if [ -d "$TJ" ] && [ ! -f "$d/$name/trace_archive.tar" ]; then
        echo "$(date) $name: archiving $TJ ($(find $TJ -type f | wc -l) files)" >> $LOG
        if tar -cf "$d/$name/trace_archive.tar" -C "$d/$name" trace_jobs && tar -tf "$d/$name/trace_archive.tar" > /dev/null; then rm -rf "$TJ"; echo "$(date) $name: archived + deleted trace_jobs" >> $LOG; else echo "$(date) $name: tar FAILED, tree kept" >> $LOG; rm -f "$d/$name/trace_archive.tar"; fi
      fi
      continue
    fi
    remaining=$((remaining+1))
    l=$(ls -t $d/logs/${name}_*.out 2>/dev/null | head -1); [ -z "$l" ] && continue
    if grep -q "WANDB_MIRROR kind=eval" "$l" && [ ! -f "$done_marker" ]; then
      echo "$(date) $name: eval block present (job $jid) -> table + scancel" >> $LOG
      python3 $PT $d/$name/trace_jobs --k 8 --out $d/pass8 >> $LOG 2>&1 && [ -n "$jid" ] && scancel $jid && echo "$(date) $name: cancelled $jid" >> $LOG
    fi
    # crash safety: job gone without the eval block -> table whatever traces exist (the eval-only concatenate crash leaves traces intact)
    if [ -z "$jid" ] && [ ! -f "$done_marker" ] && [ -d "$d/$name/trace_jobs/eval_sessions" ]; then
      echo "$(date) $name: job gone before tabling -> tabling from traces" >> $LOG
      python3 $PT $d/$name/trace_jobs --k 8 --out $d/pass8 >> $LOG 2>&1
    fi
  done
  [ $remaining -eq 0 ] && { echo "$(date) all runs tabled; exiting" >> $LOG; exit 0; }
  sleep 120
done
