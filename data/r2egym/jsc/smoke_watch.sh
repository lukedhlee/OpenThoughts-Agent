#!/bin/bash
# smoke_watch.sh <name> <jobid>: scancel once the 32 memory snapshots for step 1 exist (or the train metrics line is logged), so no final save/eval burns time.
NAME=$1; J=$2; E=/e/fscratch/reformo/lee27/experiments; D=$E/$NAME/$NAME; LOG=$E/${NAME}_watch.log
echo "$(date) smoke_watch start $NAME $J" >> $LOG
while true; do
  st=$(squeue -h -j $J -o %T 2>/dev/null); [ -z "$st" ] && { echo "$(date) job gone" >> $LOG; exit 0; }
  n=$(ls $D/checkpoints/memory_snapshots/policy_rank_*_training_step_1_*.pickle 2>/dev/null | wc -l)
  L=$(ls -t $E/$NAME/logs/*.out 2>/dev/null | head -1)
  if [ "$n" -ge 32 ] || { [ -n "$L" ] && grep -q "WANDB_MIRROR kind=train" "$L"; }; then
    sleep 90; echo "$(date) snapshots=$(ls $D/checkpoints/memory_snapshots/ 2>/dev/null | wc -l) train-line=$(grep -c "WANDB_MIRROR kind=train" "$L") -> scancel $J" >> $LOG; scancel $J; exit 0
  fi
  sleep 60
done
