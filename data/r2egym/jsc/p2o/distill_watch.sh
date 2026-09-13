#!/bin/bash
# distill_watch.sh <arm_id> <fleet_id> <run> — every 60 s: new WANDB_MIRROR train steps become one gates line each
# (distill_gates.py); a HARD_FAIL on the step-1 gates cancels the arm; a complete checkpoint 12 cancels the arm (the
# fully-async trainer keeps generating past max_steps); arm gone -> fleet released by id, bridge 9924 stopped 10 min later;
# fleet gone while the arm runs -> arm cancelled (no sandboxes, no point burning GPUs).
E=/e/fscratch/reformo/lee27/experiments; J=$1; F=$2; DST=$3
SOCK=$(eval echo ~/.ssh/cm_juwels/bridge); LOG=$E/p2o/distill_watch.log; GATES=$E/p2o/distill_gates.log
CK=$E/$DST/$DST/checkpoints; last_step=0; fleet_gone=0
log(){ echo "$(date '+%m-%d %H:%M') $*" >> $LOG; }
log "watch arm $J fleet $F run $DST"
while true; do
  RL=$(ls -t $E/$DST/logs/*.out 2>/dev/null | head -1)
  if [ -n "$RL" ]; then
    ns=$(grep -o "WANDB_MIRROR kind=train step=[0-9]*" "$RL" | tail -1 | grep -o "[0-9]*$"); ns=${ns:-0}
    if [ "$ns" -gt "$last_step" ]; then
      python3 $E/p2o/distill_gates.py "$RL" $last_step >> $GATES 2>>$LOG
      last_step=$ns; log "step $ns: $(tail -1 $GATES)"
      if tail -1 $GATES | grep -q "HARD_FAIL"; then log "hard gate failed at step $ns -> scancel $J"; scancel $J; fi
    fi
  fi
  if [ "$(cat $CK/latest_ckpt_global_step.txt 2>/dev/null)" = "12" ] && [ -f $CK/global_step_12/trainer_state.pt ] \
     && [ -z "$(find $CK/global_step_12 -mmin -2 2>/dev/null | head -1)" ] && squeue -h -j $J -o %T 2>/dev/null | grep -q .; then
    log "checkpoint 12 complete -> scancel $J"; scancel $J
  fi
  A=$(squeue -h -j $J -o "%T %M" 2>/dev/null)
  if [ -z "$A" ]; then
    log "arm $J gone -> scancel fleet $F"; ssh -o BatchMode=yes -S $SOCK juwels "scancel $F; sleep 2; squeue -h -u \$USER -o '%i %j %T'" >> $LOG 2>&1
    log "fleet released; bridge 9924 stops in 10 min"; sleep 600
    tmux kill-session -t apptainer_bridge_9924 2>/dev/null; pkill -u $USER -f "server.py --port 9924" 2>/dev/null; log "bridge 9924 stopped"; exit 0
  fi
  if [ $fleet_gone = 0 ]; then
    FL=$(ssh -o BatchMode=yes -S $SOCK juwels "squeue -h -j $F -o '%T %M'" 2>/dev/null); rc=$?
    if [ $rc = 0 ] && [ -z "$FL" ]; then fleet_gone=1; log "fleet $F gone while arm $J runs -> scancel $J"; scancel $J; fi
  fi
  sleep 60
done
