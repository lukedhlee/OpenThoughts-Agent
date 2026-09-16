#!/bin/bash
# cont_s24_pipeline.sh — the step-24 continuation on the refreshed 1,247-task pool, end to end, from ONE Jupiter login-node tmux
# (2026-09-15/16; plan: ai_memory/active/snowball-r2egym/experiments/2026-09-15_s24_continuation_plan.md, verdict at step 24 only).
#   A  anchor: the step-24 export re-scored on val441 under today's recipe (cont_s24_probe.sh); 8 GPU nodes + a 16-node JUWELS fleet
#   B  arm: 24 steps restarted from that export on refresh_s24_20260915r2_train_x16 (clone_newstack_arm.sh MODEL=...); 20 GPU nodes
#      (8 policy + 12 EAGLE-3 engines, 768 seats) + a 24-node fleet; lr 5e-7 after a 3-step warmup, staleness 2; self-stops at
#      max_steps 24 and keeps raw checkpoint 24 (max_ckpts_to_keep 2 -> 18 and 24 survive)
#   C  final: export_hf.sbatch of the arm's global_step_24 (1 node, ~4 min), then the same probe on that export; 8 GPU + 16-node fleet
#   R  readout: v2val_compare.py + heldout_compare.py, anchor vs final -> $D/readout.md
# Every JUWELS fleet is submitted here and released BY JOB ID the moment its consumer leaves squeue (09-12: 432 CPU node-h idled when
# the release depended on a Mac session). Markers in $D let a re-run resume where it stopped. The only GPU job this script cancels
# is the arm once checkpoint 24 is on disk; an arm Slurm kills before it writes a log (node-boot failure, 09-16) is resubmitted at
# most twice; any other dead arm is reported (FAILED marker) and left to the operator.
# Usage (login02, inside tmux):  bash cont_s24_pipeline.sh            env: NAME ARM ANCHOR FINAL (defaults below)
set -uo pipefail
NAME=${NAME:-cont_s24_refresh}; ARM=${ARM:-snowball_ttband_ns_cont24_refresh_a}
ANCHOR=${ANCHOR:-cont_s24_anchor_val441}; FINAL=${FINAL:-cont_s24_step24_val441}
E=/e/fscratch/reformo/lee27/experiments; T=/e/fscratch/reformo/lee27/tasks; C=/e/project1/transfernetx/lee27/code/snowball
O=/e/project1/transfernetx/lee27/code/OpenThoughts-Agent; D=$E/$NAME; mkdir -p $D; LOG=$D/pipeline.log
S24=$E/exports/snowball_ttband_ns_c_lr5e7_c_step24/model; TREE=refresh_s24_20260915r2_train_x16
SOCK=$HOME/.ssh/cm_juwels/bridge; JW="ssh -o BatchMode=yes -o ConnectTimeout=10 -S $SOCK juwels"
PORT=9930; BRIDGE=http://10.128.1.2:$PORT
FLEET_DIR=/p/project1/synthlaion/lee27/fleet; WORKER=$FLEET_DIR/refresh_sparse_worker.py; FLOG=/p/scratch/synthlaion/lee27/dc_agent_eval/logs
export OMP_NUM_THREADS=1
log(){ echo "$(date '+%F %T') $*" | tee -a $LOG; }
status(){ curl -s -m 5 http://127.0.0.1:$PORT/status; }
workers_alive(){ status | grep -q '"workers_alive": true'; }
fleet_submit(){ # <label> <nodes> <wall>: sbatch on JUWELS through the login-node master; id -> $D/fleet_<label>
  local label=$1 nodes=$2 wall=$3 id
  id=$($JW "cd $FLEET_DIR && sbatch --parsable --nodes=$nodes --time=$wall --export=ALL,HARBOR_SRC=/p/project1/synthlaion/lee27/harbor/src,WORKER_SCRIPT=$WORKER,WORKERS_PER_NODE=32,STAGING_BASE=/tmp/apptainer_staging,BRIDGE_LOGIN=jwlogin03i,BRIDGE_PORT=$PORT,MAX_CHAIN=0 -J ${NAME}_${label}_fleet juwels_workers.sbatch" 2>&1 | tail -1)
  [[ "$id" =~ ^[0-9]+$ ]] || { log "FLEET $label submit FAILED: $id"; return 1; }
  echo $id > $D/fleet_$label; log "FLEET $label $id ($nodes nodes x 32 workers, wall $wall)"
}
fleet_ready(){ # <label> <nodes> [max_min]: every node logged its worker start and the bridge sees workers
  local id nodes=$2 max=${3:-40} n=0; id=$(cat $D/fleet_$1)
  for i in $(seq 1 $((max*2))); do
    n=$($JW "grep -c 'starting 32 workers' $FLOG/apptainer_workers_juwels_$id.out 2>/dev/null; true" 2>/dev/null | tail -1)
    [[ "$n" =~ ^[0-9]+$ ]] && [ "$n" -ge "$nodes" ] && workers_alive && { log "FLEET $1 ready ($n/$nodes nodes up)"; return 0; }
    sleep 30
  done
  log "FLEET $1 NOT ready after $max min ($n/$nodes nodes)"; return 1
}
fleet_release(){ local id; id=$(cat $D/fleet_$1 2>/dev/null) || return 0; [ -n "$id" ] || return 0; $JW "scancel $id" && { mv $D/fleet_$1 $D/fleet_$1.released; log "RELEASE fleet $1 $id"; } || log "RELEASE fleet $1 $id FAILED (retry next poll)"; }
job_gone(){ # true only when squeue answered without the job AND sacct shows a terminal state; a slurmctld hiccup also returns
  # empty output (09-16: one poll read a PENDING arm as gone), and a false "gone" would release the fleet and resubmit the arm
  local out rc; out=$(squeue -h -j "$1" -o %T 2>&1); rc=$?
  if [ $rc -eq 0 ]; then [ -z "$out" ] || return 1; else echo "$out" | grep -q "Invalid job id" || return 1; fi
  sacct -j "$1" -X -n -o State 2>/dev/null | grep -qE "COMPLETED|CANCELLED|FAILED|TIMEOUT|NODE_FAIL|OUT_OF_MEMORY|PREEMPTED|DEADLINE|BOOT_FAIL"; }
snapshot(){ log "queue: $(squeue -h -u $USER -o '%i:%j:%T:%M' | tr '\n' ' ') | fleets: $($JW "squeue -h -u lee27 -o '%i:%T:%M'" 2>/dev/null | tr '\n' ' ') | bridge: $(status | cut -c1-120)"; }

log "=== pipeline start NAME=$NAME ARM=$ARM ANCHOR=$ANCHOR FINAL=$FINAL"
[ -f $D/DONE ] && { log "already DONE"; exit 0; }
status > /dev/null || { log "bridge $PORT not answering on 127.0.0.1"; exit 1; }
$JW "true" || { log "JUWELS master $SOCK dead"; exit 1; }
[ -d $T/$TREE ] && [ -f $S24/model.safetensors.index.json ] || { log "tree or s24 export missing"; exit 1; }

# ---------- A + B: GPU jobs first; each fleet is submitted by the poll loop once its GPU job is on nodes (JUWELS batch starts in
# ~2 min, the GPU job needs ~15 min to load the model, so the seats are always up before the first trial) ----------
job_started(){ squeue -h -j "$1" -o %T 2>/dev/null | grep -q -E "RUNNING|CONFIGURING|COMPLETING"; }
if [ ! -f $D/A_DONE ] && [ ! -f $D/job_A ]; then
  bash $C/cont_s24_probe.sh $ANCHOR $S24 1 2>&1 | tee -a $LOG
  [ -f $E/$ANCHOR/job_id ] || { log "anchor probe not submitted"; exit 1; }
  cp $E/$ANCHOR/job_id $D/job_A; log "PROBE A $(cat $D/job_A)"
fi
if [ ! -f $D/B_DONE ] && [ ! -f $D/job_B ]; then
  if [ ! -f $E/$ARM/sbatch/${ARM}_rl.sbatch ]; then
    NODES=20 SPEC=1 TRAIN_TREE=$TREE MODEL=$S24 BRIDGE=$BRIDGE bash $C/clone_newstack_arm.sh $ARM 0 \
      "++terminal_bench_config.harbor.n_concurrent_trials=768" "trainer.max_steps=24" \
      "trainer.policy.optimizer_config.num_warmup_steps=3" 2>&1 | tee -a $LOG
    [ -f $E/$ARM/sbatch/${ARM}_rl.sbatch ] || { log "arm build failed"; exit 1; }
  fi
  J=$(cd $O && DCFT=$PWD sbatch --parsable $E/$ARM/sbatch/${ARM}_rl.sbatch) || { log "arm sbatch failed"; exit 1; }
  echo $J > $D/job_B; log "ARM B $J"
fi
# ---------- poll: bring a fleet up only once its GPU job is on nodes (09-16: the arm sat PENDING 8 h behind another user's planned
# job while its 24-node fleet idled), release each fleet the moment its GPU job leaves the queue ----------
while [ ! -f $D/A_DONE ] || [ ! -f $D/B_DONE ]; do
  if [ ! -f $D/A_DONE ] && [ ! -f $D/fleet_A ] && [ ! -f $D/fleet_A.released ] && job_started $(cat $D/job_A); then
    log "PROBE A $(cat $D/job_A) is on nodes; bringing its fleet up"; fleet_submit A 16 03:45:00 || log "fleet A submit failed, retrying next poll"
  fi
  [ ! -f $D/A_DONE ] && job_gone $(cat $D/job_A) && { fleet_release A; touch $D/A_DONE; log "A gone ($(sacct -j $(cat $D/job_A) -n -o State,Elapsed,NNodes -X | head -1))"; }
  if [ ! -f $D/B_DONE ] && [ ! -f $D/fleet_B ] && [ ! -f $D/fleet_B.released ] && job_started $(cat $D/job_B); then
    log "ARM B $(cat $D/job_B) is on nodes; bringing its fleet up"; fleet_submit B 24 05:30:00 || log "fleet B submit failed, retrying next poll"
  fi
  # training is over once checkpoint 24 is complete and quiet; cancel the job then (own job, max_steps reached) so no end-of-run
  # eval or HF export burns the fleet and the GPUs; export_hf.sbatch reads the raw checkpoint (the P²O and control arms were
  # cancelled at their checkpoints the same way)
  if [ ! -f $D/B_DONE ] && [ ! -f $D/B_TRAINED ] && [ "$(cat $E/$ARM/$ARM/checkpoints/latest_ckpt_global_step.txt 2>/dev/null)" = "24" ] \
     && [ -f $E/$ARM/$ARM/checkpoints/global_step_24/trainer_state.pt ] && [ -z "$(find $E/$ARM/$ARM/checkpoints/global_step_24 -mmin -2 2>/dev/null | head -1)" ]; then
    touch $D/B_TRAINED; log "ARM B checkpoint 24 complete and quiet; cancelling $(cat $D/job_B)"; scancel $(cat $D/job_B)
  fi
  if [ ! -f $D/B_DONE ] && job_gone $(cat $D/job_B); then
    fleet_release B
    st=$(sacct -j $(cat $D/job_B) -n -o State%30,Elapsed,NNodes -X | head -1)
    # 09-16: Slurm cancelled the arm ("CANCELLED by 0") 2 min into CONFIGURING, no log written (a node failed to boot), no requeue;
    # resubmit the same sbatch, at most twice in total, never a job that got far enough to write its log
    if [ ! -f $D/B_TRAINED ] && [ -z "$(ls $E/$ARM/logs/*_$(cat $D/job_B).out 2>/dev/null)" ] && [ $(ls $D/job_B.* 2>/dev/null | wc -l) -lt 2 ]; then
      mv $D/job_B $D/job_B.$(cat $D/job_B).killed_before_log; rm -f $D/fleet_B.released
      J=$(cd $O && DCFT=$PWD sbatch --parsable $E/$ARM/sbatch/${ARM}_rl.sbatch) && { echo $J > $D/job_B; log "ARM B gone before writing a log ($st); resubmitted as $J"; } \
        || { log "ARM B resubmit failed ($st)"; touch $D/B_DONE; }
    else
      touch $D/B_DONE; log "B gone ($st)"
    fi
  fi
  [ -f $D/A_DONE ] && [ -f $D/B_DONE ] && break
  (( $(date +%s) % 600 < 60 )) && snapshot
  sleep 60
done
# ---------- C: export checkpoint 24, probe it ----------
CK=$E/$ARM/$ARM/checkpoints
if [ ! -f $D/C_EXPORTED ]; then
  latest=$(cat $CK/latest_ckpt_global_step.txt 2>/dev/null)
  if [ "$latest" != "24" ] || [ ! -f $CK/global_step_24/trainer_state.pt ]; then
    log "ARM ended without checkpoint 24 (latest=$latest); nothing exported"; touch $D/FAILED; exit 1
  fi
  if [ ! -f $E/exports/${ARM}_step24/model/model.safetensors.index.json ]; then
    J=$(cd $C && sbatch --parsable --export=ALL,RUN=$ARM,STEP=24 export_hf.sbatch) || { log "export sbatch failed"; exit 1; }
    log "EXPORT $J"; while ! job_gone $J; do sleep 20; done
    grep -E "tensor-name set|done$" $E/exports/logs/snowball_export_hf_$J.out | tail -2 | tee -a $LOG
  fi
  [ -f $E/exports/${ARM}_step24/model/model.safetensors.index.json ] || { log "EXPORT MISSING"; touch $D/FAILED; exit 1; }
  touch $D/C_EXPORTED; log "export ok: $E/exports/${ARM}_step24/model"
fi
if [ ! -f $D/C_DONE ]; then
  if [ ! -f $D/job_C ]; then
    bash $C/cont_s24_probe.sh $FINAL $E/exports/${ARM}_step24/model 1 2>&1 | tee -a $LOG
    [ -f $E/$FINAL/job_id ] || { log "final probe not submitted"; exit 1; }
    cp $E/$FINAL/job_id $D/job_C; log "PROBE C $(cat $D/job_C)"
  fi
  while ! job_gone $(cat $D/job_C); do
    if [ ! -f $D/fleet_C ] && [ ! -f $D/fleet_C.released ] && job_started $(cat $D/job_C); then
      log "PROBE C $(cat $D/job_C) is on nodes; bringing its fleet up"; fleet_submit C 16 03:45:00 || log "fleet C submit failed, retrying next poll"
    fi
    (( $(date +%s) % 600 < 60 )) && snapshot; sleep 60
  done
  fleet_release C; touch $D/C_DONE; log "C gone ($(sacct -j $(cat $D/job_C) -n -o State,Elapsed,NNodes -X | head -1))"
fi
# ---------- R: readout (probe_watch.sh tables each probe when its eval block lands) ----------
for p in $ANCHOR $FINAL; do
  for i in $(seq 1 60); do [ -f $E/$p/pass8_pass8_table.csv ] && break; sleep 30; done
  [ -f $E/$p/pass8_pass8_table.csv ] || { log "no pass8 table for $p after 30 min"; touch $D/FAILED; exit 1; }
done
{ echo "# $NAME readout $(date '+%F %T') (cluster time; PT = -9 h)"; echo; python3 $C/v2val_compare.py $ANCHOR $FINAL; echo; python3 $C/heldout_compare.py $ANCHOR $FINAL; } > $D/readout.md 2>&1
log "READOUT -> $D/readout.md"; head -40 $D/readout.md | tee -a $LOG
touch $D/DONE; log "=== DONE"
