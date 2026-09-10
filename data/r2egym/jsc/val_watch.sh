#!/bin/bash
# val_watch.sh <name> <jobid> <out_suffix>  — when the eval block appears, write the pass@8 table and scancel (its train step adds nothing).
NAME=$1; J=$2; SUF=$3; E=/e/fscratch/reformo/lee27/experiments; PT=/e/project1/transfernetx/lee27/code/snowball/pass8_table.py
L=$E/$NAME/logs/${NAME}_$J.out; LOG=$E/${NAME}_watch.log; echo "$(date) val_watch start $NAME $J" >> $LOG
while true; do
  st=$(squeue -h -j $J -o %T 2>/dev/null)
  if [ -f "$L" ] && grep -q "WANDB_MIRROR kind=eval" "$L" && [ ! -f $E/$NAME/pass8_${SUF}_summary.json ]; then
    echo "$(date) eval block present -> table + scancel" >> $LOG
    python3 $PT $E/$NAME/$NAME/trace_jobs --k 8 --out $E/$NAME/pass8_$SUF >> $LOG 2>&1 && [ -n "$st" ] && scancel $J && echo "$(date) cancelled $J" >> $LOG
    exit 0
  fi
  [ -z "$st" ] && { echo "$(date) job gone before eval block; writing whatever exists" >> $LOG; python3 $PT $E/$NAME/$NAME/trace_jobs --k 8 --out $E/$NAME/pass8_${SUF}_partial >> $LOG 2>&1; exit 1; }
  sleep 120
done
