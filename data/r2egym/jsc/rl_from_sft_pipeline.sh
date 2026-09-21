#!/bin/bash
# rl_from_sft_pipeline.sh — plain-GRPO R2E-Gym RL from an SFT export, end to end, from ONE Jupiter login-node tmux.
# Derived from cont_s24_pipeline.sh (2026-09-15/16) with the anchor and final val441 probes removed (Luke paused the
# R2E-Gym validation probes 2026-09-19; the readout is SWE-bench Verified random-100 + Terminal-Bench 2 trials):
#   B  arm: clone_newstack_arm.sh MODEL=<export> NODES=20 SPEC=1 (8 policy + 12 EAGLE-3 engines, 768 seats, the default
#      r2egym-tt-v2-train-basecurr-x16 tree, lr 5e-7 after a 3-step warmup, staleness 2, GRPO + sequence_mean), max_steps 24,
#      a 24-node JUWELS fleet brought up once the arm is on nodes and released by job id when it leaves the queue;
#      the arm is cancelled once raw checkpoint 24 is on disk and quiet (own job, max_steps reached)
#   C  export: export_hf.sbatch of the final checkpoint (max_steps, or $D/max_steps written by band_swap.sh; 1 node, ~4 min)
#   R  readout: TRIALS x (serve_snowball.sbatch 6 h + eval_chain.sh) on the step-24 export, tags <TAG>t<i>; waits for the
#      final_*.txt files, then prints them
# Markers in $D let a re-run resume where it stopped. Usage (login02, inside tmux):
#   NAME=rl_d517 ARM=snowball_ota3d517_rl_a MODEL=<hf export dir> TAG=rld517 TRIALS=3 bash rl_from_sft_pipeline.sh
set -uo pipefail
NAME=${NAME:?}; ARM=${ARM:?}; MODEL=${MODEL:?}; TAG=${TAG:?}; TRIALS=${TRIALS:-3}; MAX_STEPS=${MAX_STEPS:-24}   # MAX_STEPS: the clone arg and the ms() default
TRAIN_TREE=${TRAIN_TREE:-r2egym-tt-v2-train-basecurr-x16}; ARM_WALL=${ARM_WALL:-08:00:00}; ACCOUNT=${ACCOUNT:-laionize}
E=/e/fscratch/reformo/lee27/experiments; T=/e/fscratch/reformo/lee27/tasks; C=/e/project1/transfernetx/lee27/code/snowball
W=/e/project1/transfernetx/lee27/code/tb2; TB=$E/tb2
O=/e/project1/transfernetx/lee27/code/OpenThoughts-Agent; D=$E/$NAME; mkdir -p $D; LOG=$D/pipeline.log
SOCK=$HOME/.ssh/cm_juwels/bridge; JW="ssh -o BatchMode=yes -o ConnectTimeout=10 -S $SOCK juwels"
PORT=${PORT:-9930}; BRIDGE=http://10.128.1.2:$PORT
FLEET_DIR=/p/project1/synthlaion/lee27/fleet; WORKER=$FLEET_DIR/refresh_sparse_worker.py; FLOG=/p/scratch/synthlaion/lee27/dc_agent_eval/logs
export OMP_NUM_THREADS=1
ms(){ cat $D/max_steps 2>/dev/null || echo ${MAX_STEPS:-24}; }   # band_swap.sh rewrites $D/max_steps once the learnable band is sized
log(){ echo "$(date '+%F %T') $*" | tee -a $LOG; }
status(){ curl -s -m 5 http://127.0.0.1:$PORT/status; }
workers_alive(){ status | grep -q '"workers_alive": true'; }
fleet_submit(){ # <label> <nodes> <wall>: sbatch on JUWELS through the login-node master; id -> $D/fleet_<label>
  local label=$1 nodes=$2 wall=$3 id
  id=$($JW "cd $FLEET_DIR && sbatch --parsable --nodes=$nodes --time=$wall --export=ALL,HARBOR_SRC=/p/project1/synthlaion/lee27/harbor/src,WORKER_SCRIPT=$WORKER,WORKERS_PER_NODE=32,STAGING_BASE=/tmp/apptainer_staging,BRIDGE_LOGIN=jwlogin03i,BRIDGE_PORT=$PORT,MAX_CHAIN=0 -J ${NAME}_${label}_fleet juwels_workers.sbatch" 2>&1 | tail -1)
  [[ "$id" =~ ^[0-9]+$ ]] || { log "FLEET $label submit FAILED: $id"; return 1; }
  echo $id > $D/fleet_$label; log "FLEET $label $id ($nodes nodes x 32 workers, wall $wall)"
}
fleet_release(){ local id; id=$(cat $D/fleet_$1 2>/dev/null) || return 0; [ -n "$id" ] || return 0; $JW "scancel $id" && { mv $D/fleet_$1 $D/fleet_$1.released; log "RELEASE fleet $1 $id"; } || log "RELEASE fleet $1 $id FAILED (retry next poll)"; }
job_gone(){ # true only when squeue answered without the job AND sacct shows a terminal state (a slurmctld hiccup returns empty output)
  local out rc; out=$(squeue -h -j "$1" -o %T 2>&1); rc=$?
  if [ $rc -eq 0 ]; then [ -z "$out" ] || return 1; else echo "$out" | grep -q "Invalid job id" || return 1; fi
  sacct -j "$1" -X -n -o State 2>/dev/null | grep -qE "COMPLETED|CANCELLED|FAILED|TIMEOUT|NODE_FAIL|OUT_OF_MEMORY|PREEMPTED|DEADLINE|BOOT_FAIL"; }
job_started(){ squeue -h -j "$1" -o %T 2>/dev/null | grep -q -E "RUNNING|CONFIGURING|COMPLETING"; }
snapshot(){ log "queue: $(squeue -h -u $USER -o '%i:%j:%T:%M' | tr '\n' ' ') | fleets: $($JW "squeue -h -u lee27 -o '%i:%T:%M'" 2>/dev/null | tr '\n' ' ') | bridge: $(status | cut -c1-120)"; }

log "=== pipeline start NAME=$NAME ARM=$ARM MODEL=$MODEL TAG=$TAG TRIALS=$TRIALS tree=$TRAIN_TREE"
[ -f $D/DONE ] && { log "already DONE"; exit 0; }
status > /dev/null || { log "bridge $PORT not answering on 127.0.0.1"; exit 1; }
$JW "true" || { log "JUWELS master $SOCK dead"; exit 1; }
[ -d $T/$TRAIN_TREE ] && [ -f $MODEL/model.safetensors.index.json ] || { log "tree or export missing"; exit 1; }

# ---------- B: the arm; its fleet comes up once the arm is on nodes (09-16: an arm sat PENDING 8 h while its fleet idled) ----------
if [ ! -f $D/B_DONE ] && [ ! -f $D/job_B ]; then
  if [ ! -f $E/$ARM/sbatch/${ARM}_rl.sbatch ]; then
    NODES=20 SPEC=1 TRAIN_TREE=$TRAIN_TREE MODEL=$MODEL BRIDGE=$BRIDGE bash $C/clone_newstack_arm.sh $ARM 0 \
      "++terminal_bench_config.harbor.n_concurrent_trials=768" "trainer.max_steps=${MAX_STEPS:-24}" \
      "trainer.policy.optimizer_config.num_warmup_steps=3" 2>&1 | tee -a $LOG
    [ -f $E/$ARM/sbatch/${ARM}_rl.sbatch ] || { log "arm build failed"; exit 1; }
    # the clone hard-codes 12 h on reformo: 12 h jobs are deferred by a hidden reservation and reformo's fairshare has collapsed
    sed -i "s/^#SBATCH --time=12:00:00$/#SBATCH --time=$ARM_WALL/; s/^#SBATCH --account reformo$/#SBATCH --account $ACCOUNT/" $E/$ARM/sbatch/${ARM}_rl.sbatch
    grep -q "^#SBATCH --time=$ARM_WALL$" $E/$ARM/sbatch/${ARM}_rl.sbatch && grep -q "^#SBATCH --account $ACCOUNT$" $E/$ARM/sbatch/${ARM}_rl.sbatch || { log "sbatch wall/account sed failed"; exit 1; }
  fi
  J=$(cd $O && DCFT=$PWD sbatch --parsable $E/$ARM/sbatch/${ARM}_rl.sbatch) || { log "arm sbatch failed"; exit 1; }
  echo $J > $D/job_B; log "ARM B $J (wall $ARM_WALL, account $ACCOUNT)"
fi
while [ ! -f $D/B_DONE ]; do
  if [ ! -f $D/fleet_B ] && [ ! -f $D/fleet_B.released ] && job_started $(cat $D/job_B); then
    log "ARM B $(cat $D/job_B) is on nodes; bringing its fleet up"; fleet_submit B 24 05:30:00 || log "fleet B submit failed, retrying next poll"
  fi
  MS=$(ms)
  if [ ! -f $D/B_TRAINED ] && [ "$(cat $E/$ARM/$ARM/checkpoints/latest_ckpt_global_step.txt 2>/dev/null)" = "$MS" ] \
     && [ -f $E/$ARM/$ARM/checkpoints/global_step_$MS/trainer_state.pt ] && [ -z "$(find $E/$ARM/$ARM/checkpoints/global_step_$MS -mmin -2 2>/dev/null | head -1)" ]; then
    touch $D/B_TRAINED; log "ARM B checkpoint $MS complete and quiet; cancelling $(cat $D/job_B)"; scancel $(cat $D/job_B)
  fi
  if job_gone $(cat $D/job_B); then
    fleet_release B
    st=$(sacct -j $(cat $D/job_B) -n -o State%30,Elapsed,NNodes -X | head -1)
    # an arm Slurm kills before it writes a log (node-boot failure, 09-16) is resubmitted, at most twice in total
    if [ ! -f $D/B_TRAINED ] && [ -z "$(ls $E/$ARM/logs/*_$(cat $D/job_B).out 2>/dev/null)" ] && [ $(ls $D/job_B.* 2>/dev/null | wc -l) -lt 2 ]; then
      mv $D/job_B $D/job_B.$(cat $D/job_B).killed_before_log; rm -f $D/fleet_B.released
      J=$(cd $O && DCFT=$PWD sbatch --parsable $E/$ARM/sbatch/${ARM}_rl.sbatch) && { echo $J > $D/job_B; log "ARM B gone before writing a log ($st); resubmitted as $J"; } \
        || { log "ARM B resubmit failed ($st)"; touch $D/B_DONE; }
    else
      touch $D/B_DONE; log "B gone ($st)"
    fi
  fi
  [ -f $D/B_DONE ] && break
  (( $(date +%s) % 600 < 60 )) && snapshot
  sleep 60
done
# ---------- C: export the final checkpoint ----------
MS=$(ms); CK=$E/$ARM/$ARM/checkpoints; EXPORT=$E/exports/${ARM}_step$MS/model
if [ ! -f $D/C_EXPORTED ]; then
  latest=$(cat $CK/latest_ckpt_global_step.txt 2>/dev/null)
  if [ "$latest" != "$MS" ] || [ ! -f $CK/global_step_$MS/trainer_state.pt ]; then
    log "ARM ended without checkpoint $MS (latest=$latest); nothing exported"; touch $D/FAILED; exit 1
  fi
  if [ ! -f $EXPORT/model.safetensors.index.json ]; then
    J=$(cd $C && sbatch --parsable --export=ALL,RUN=$ARM,STEP=$MS export_hf.sbatch) || { log "export sbatch failed"; exit 1; }
    log "EXPORT $J"; while ! job_gone $J; do sleep 20; done
    grep -E "tensor-name set|done$" $E/exports/logs/snowball_export_hf_$J.out | tail -2 | tee -a $LOG
  fi
  [ -f $EXPORT/model.safetensors.index.json ] || { log "EXPORT MISSING"; touch $D/FAILED; exit 1; }
  touch $D/C_EXPORTED; log "export ok: $EXPORT"
fi
# ---------- R: SWE-bench random-100 + TB2, TRIALS independent serve nodes ----------
if [ ! -f $D/R_LAUNCHED ]; then
  for t in $(seq 1 $TRIALS); do
    J=$(MODEL=$EXPORT POLICY=trained sbatch --parsable --time=06:00:00 --account=$ACCOUNT $W/serve_snowball.sbatch) || { log "serve sbatch failed (trial $t)"; continue; }
    tmux new-session -d -s eval_${TAG}t$t "bash $W/eval_chain.sh $J ${TAG}t$t; sleep 600"
    log "TRIAL $t: serve $J, tmux eval_${TAG}t$t (runs ${TAG}t${t}swe_v01_* and ${TAG}t${t}_v01_*)"
    sleep 90
  done
  touch $D/R_LAUNCHED
fi
for i in $(seq 1 96); do   # up to 8 h: queue + 4.5 h trial pair + recovery pass
  n=$(ls $TB/final_${TAG}t[0-9]swe_v01_*.txt $TB/final_${TAG}t[0-9]_v01_*.txt 2>/dev/null | wc -l)
  [ "$n" -ge $((2*TRIALS)) ] && break; sleep 300
done
log "finals on disk: $n of $((2*TRIALS))"; grep -H "pass@1" $TB/final_${TAG}t*_v01_*.txt 2>/dev/null | tee -a $LOG
touch $D/DONE; log "=== DONE"
