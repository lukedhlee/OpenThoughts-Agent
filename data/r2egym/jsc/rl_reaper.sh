#!/bin/bash
# rl_reaper.sh <run>... — login-node loop (every 600 s) that keeps an RL run's inode footprint flat:
#  (1) deletes finished TRAINING trial dirs (trial-level result.json older than AGE_MIN; the trainer dumps every step's
#      rollouts to exports/dumped_data/global_step_N_train_rollouts.jsonl from memory and never re-reads the trial dir);
#  (2) for each finished EVAL session (every trial dir has result.json, nothing modified for AGE_MIN) tables it with
#      pass8_table.py into <run>/<run>/eval_tables/<session>_{pass8_table.csv,summary.json,strict_mixed.txt}, tars the
#      tree into <run>/<run>/eval_archive/<session>.tar (one inode) and deletes it.
# Idempotent; never exits. Log: $E/rl_reaper.log. Do not run on a probe/pool tree (those are tabled by pool_watch_par.sh).
E=/e/fscratch/reformo/lee27/experiments; C=/e/project1/transfernetx/lee27/code/snowball; LOG=$E/rl_reaper.log; AGE_MIN=${AGE_MIN:-30}
echo "$(date '+%m-%d %H:%M') reaper start runs=$* age=${AGE_MIN}m" >> $LOG
while true; do
  for R in "$@"; do
    D=$E/$R/$R; TJ=$D/trace_jobs; [ -d "$TJ" ] || continue
    n=0
    for t in "$TJ"/*__*/; do
      [ -f "$t/result.json" ] || continue
      [ -n "$(find "$t/result.json" -mmin +$AGE_MIN 2>/dev/null)" ] || continue
      rm -rf "$t" && n=$((n+1))
    done
    [ $n -gt 0 ] && echo "$(date '+%m-%d %H:%M') $R: reaped $n train trials; remaining $(ls -d "$TJ"/*__*/ 2>/dev/null | wc -l)" >> $LOG
    for s in "$TJ"/eval_sessions/*/; do
      [ -d "$s" ] || continue; s=${s%/}; name=$(basename "$s")
      tot=$(ls -d "$s"/*__*/ 2>/dev/null | wc -l); [ "$tot" -gt 0 ] || continue
      fin=$(ls "$s"/*__*/result.json 2>/dev/null | wc -l); [ "$fin" -eq "$tot" ] || continue
      [ -z "$(find "$s" -type f -mmin -$AGE_MIN 2>/dev/null | head -1)" ] || continue
      mkdir -p "$D/eval_tables" "$D/eval_archive"; tmp=$(mktemp -d /tmp/lee27_reap_XXXXXX); mkdir -p "$tmp/eval_sessions"; ln -s "$s" "$tmp/eval_sessions/$name"
      python3 $C/pass8_table.py "$tmp" --k 8 --out "$D/eval_tables/$name" > "$D/eval_tables/$name.txt" 2>&1; rc=$?; rm -rf "$tmp"
      if [ $rc -ne 0 ] || [ ! -s "$D/eval_tables/${name}_summary.json" ]; then echo "$(date '+%m-%d %H:%M') $R: $name table FAILED rc=$rc" >> $LOG; continue; fi
      if tar -cf "$D/eval_archive/$name.tar" -C "$TJ/eval_sessions" "$name" && tar -tf "$D/eval_archive/$name.tar" > /dev/null; then
        rm -rf "$s"; echo "$(date '+%m-%d %H:%M') $R: $name tabled+archived ($tot trials; $(head -c 300 "$D/eval_tables/${name}_summary.json" | tr -d '\n '))" >> $LOG
      else echo "$(date '+%m-%d %H:%M') $R: $name tar FAILED" >> $LOG; fi
    done
  done
  sleep 600
done
