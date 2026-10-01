#!/bin/bash
# after_arm.sh <arm run> <start tag> <start export dir> — the verdict pipeline of one RL arm, from a login-node tmux:
#   1. wait for the arm's job to leave the queue; require checkpoint <STEP> (default 30)
#   2. export_hf.sbatch of that checkpoint -> experiments/rl/exports/<arm>_step<STEP>/model (aux files from the start export)
#   3. eval_sft.sh queue (GROUPED, REPS 1 2 3: TB2.1, SWE-bench Verified random-100, TB-lite; the gist's harness and serve)
#      for the RL export, tag <arm without _>s<STEP>, in its own tmux window, logging to experiments/rl/<arm>/evals.log
#   4. held-out probes at the screens' exact layout (4 policy + 4 engines, 360 seats, K=8, draft on): the start export and
#      the RL export on the 123 clean held-out tasks, one after the other (the org's 1,200-sandbox cap), each waiting for
#      sandbox headroom; then screen_report.py of both into experiments/rl/<arm>/heldout/.
# Markers in experiments/rl/<arm>/chain/ let a re-run resume. Read-only towards everything it did not start.
set -uo pipefail
ARM=${1:?arm}; START_TAG=${2:?start tag}; START=${3:?start export}; STEP=${STEP:-30}
E=/scratch/11584/lukedhlee/experiments/rl; OTA=$HOME/snowball/ota-rl; RL=$OTA/data/r2egym/horizon/rl
PY=$HOME/snowball/envs/snowball/bin/python; P=/scratch/11584/lukedhlee/rl_pool
C=$E/$ARM/chain; mkdir -p $C; LOG=$E/$ARM/chain.log
log() { echo "[$(TZ=America/Los_Angeles date '+%F %T PT')] $*" | tee -a $LOG; }
J=$(tail -n 1 $E/$ARM/jobs.txt); TAG=${ARM//_/}s$STEP; EXP=$E/exports/${ARM}_step$STEP/model
DRAFT=$(ls -d /scratch/11584/lukedhlee/hf_hub/models--laion--snowball-64k-eagle3-draft-r2egym/snapshots/*)
log "chain start: arm $ARM job $J, start $START_TAG, tag $TAG"

# 1. arm done
while squeue -h -j "$J" -t PENDING,CONFIGURING,RUNNING -o %i 2>/dev/null | grep -q .; do sleep 120; done
ck=$(cat $E/$ARM/$ARM/checkpoints/latest_ckpt_global_step.txt 2>/dev/null)
[ "$ck" = "$STEP" ] || { log "FAILED: arm ended at checkpoint '${ck}', not $STEP"; exit 1; }
log "arm ended; checkpoint $STEP present"

# 2. export
if [ ! -f $C/exported ]; then
  for try in 1 2; do
    X=$(cd $OTA && sbatch --parsable --export=ALL,RUN=$ARM,STEP=$STEP,BASE=$START -J rl_export $RL/export_hf.sbatch | tail -n 1 | grep -oE '^[0-9]+')
    log "export job $X (try $try)"
    while squeue -h -j "$X" -o %i 2>/dev/null | grep -q .; do sleep 30; done
    grep -q "missing files: \[\]" $E/exports/logs/rl_export_$X.out && grep -q "tensor-name set equal to base: True" $E/exports/logs/rl_export_$X.out && { touch $C/exported; break; }
    log "export $X did not verify"
  done
  [ -f $C/exported ] || { log "FAILED: export"; exit 1; }
fi
log "export verified: $EXP"

# 3. evals (own tmux window of the session this runs in)
if [ ! -f $C/evals_started ]; then
  tmux new-window -d -t "${TMUX_SESSION:-rlchain}" -n "ev_$ARM" "cd $OTA/data/r2egym/horizon/sft && MODELS='$TAG=$EXP' REPS='1 2 3' GROUPED=1 bash eval_sft.sh >> $E/$ARM/evals.log 2>&1"
  touch $C/evals_started; log "eval queue started (tag $TAG): $E/$ARM/evals.log"
fi

# 4. held-out probes, one at a time
probe() {  # <name> <model>
  local n=$1 m=$2
  [ -f $C/probe_$n.done ] && return 0
  if [ ! -f $C/probe_$n.submitted ]; then
    (cd $RL && python3 make_arm.py --name $n --recipe arm --probe 8 --model $m --tree $P/heldout_tree --steps 1 --seats 360 \
       --coords 16 --policy-nodes 4 --engines 4 --draft $DRAFT --wall 04:00:00 >> $LOG 2>&1) || { log "render $n FAILED"; return 1; }
    for i in $(seq 1 72); do
      (cd $OTA && bash $RL/launch_arm.sh $E/$n >> $LOG 2>&1) && { touch $C/probe_$n.submitted; break; }
      log "probe $n not launched yet (sandbox headroom?); retry in 10 min"; sleep 600
    done
    [ -f $C/probe_$n.submitted ] || { log "FAILED: probe $n never launched"; return 1; }
  fi
  local pj; pj=$(tail -n 1 $E/$n/jobs.txt)
  while squeue -h -j "$pj" -o %i 2>/dev/null | grep -q .; do sleep 120; done
  local t; t=$(grep -m1 -o "kind=eval step=0 logged [0-9:]*" $E/$n/logs/${n}_$pj.out)
  [ -n "$t" ] || { log "probe $n ended without its eval line"; return 1; }
  touch $C/probe_$n.done; log "probe $n done ($t)"
}
probe ho_${START_TAG} $START && probe ho_$TAG $EXP || { log "FAILED: held-out probes"; exit 1; }
mkdir -p $E/$ARM/heldout
for n in ho_${START_TAG} ho_$TAG; do
  python3 $RL/screen_report.py --out $E/$ARM/heldout/$n --repo-map $P/coverage.csv $E/$n >> $LOG 2>&1
done
log "CHAIN_DONE $ARM: export $EXP, evals tag $TAG ($E/$ARM/evals.log), held-out reports in $E/$ARM/heldout/"
