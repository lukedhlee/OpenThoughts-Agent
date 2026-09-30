#!/bin/bash
# launch_eval_heldout.sh <serve job> <run name> [smoke] — Horizon login node. The held-out 300 CalibForge eval of one
# student checkpoint, as Jupiter ran it for the relay SFT arms (notes/relay/sft_launch_plan.md "Eval", the
# heldout6516_<arm>_20260928 runs): run_pilot.sh with RUN_KIND=heldout, ARMS=student_only (pass-through router,
# Terminus-2 strict, summarization off), 1 serve node, CONC=100, a 2.5 h deadline (Jupiter's CAP_NODE_H=2.5), 65,536 input / 16,384 output,
# CLOCK=wall (harbor's own 1x agent budget), one trial per task, harbor-relay @ 89098635.
#
# By diff from launch_driver.sh: the same driver job (driver.sbatch) and login-side tunnels, but none of the relay
# settings. launch_driver.sh fills every relay knob with `${X:-default}`, so an empty value cannot turn one off; this
# script instead unsets them and leaves run_pilot.sh's own defaults, which are the Jupiter eval's (balance=pinned,
# stagger 0, no latency gate, no context budget / row limit, no stop rule, verify_note 0).
#
#   S=$(STUDENT_MODEL=<snapshot> bash serve_submit.sh 1 1 03:00:00 eval_srv_<name>)
#   bash launch_eval_heldout.sh $S heldout_<name>
set -uo pipefail
SERVE_JOB=${1:?serve job id}; NAME=${2:?run name}; MODE=${3:-full}
HERE=$(cd "$(dirname "$0")" && pwd); OTA=$(cd $HERE/../../.. && pwd)
[ -f $HERE/serve_env.sh ] && source $HERE/serve_env.sh
S=${SCRATCH_DIR:-/scratch/11584/$USER}
export E=${E:-${RELAY_EXP_DIR:-$S/experiments/relay/pilot}}; LOGS=$E/logs; mkdir -p $LOGS $E/runs
[ -e $E/runs/$NAME ] && { echo "$E/runs/$NAME exists; pick a new run name"; exit 1; }
# relay-only knobs: leave them to run_pilot.sh's defaults (off)
unset CTX_BUDGET ROW_MAX ROW_RESERVE BALANCE STAGGER_SEC GATE_MIN VERIFY_NOTE STOP_AFTER STOP_OVF STOP_FMT STOP_HERR \
  HERR_EXCLUDE TARGET_FAIL TARGET_BASE ARM_GATES TAKEOVER_MIN TAKEOVER_MAX TAKEOVER_AFTER TAKEOVER_ACTION ACCEPT_FLAG \
  ACCEPT_STOP TASK_LIST SHUFFLE_SEED OVF_AFTER
export SERVE_JOB RUN_NAME=$NAME RUN_MODE=$MODE OTA
export TUNNEL_PORTS=${TUNNEL_PORTS:-18080,18081} STUDENT_TOKENIZER=${STUDENT_TOKENIZER:-meta}
export TREE=${TREE:-$S/tasks/calibforge_heldout300} NTASKS=300 RUN_KIND=heldout
# the Jupiter eval's settings (sft_launch_plan.md's run_pilot.sh line). CAP_NODE_H: Jupiter's 2.5 on one serve node gave a
# 2.5 h deadline (run_pilot.sh: start + (cap - reserved) / nodes - margin); here the driver node counts too and its
# VERIFY_WAIT tail (0.75 node-h) is reserved, so the same 2.5 h deadline needs 2 x 2.5 + 0.75 = 5.75
export ARMS=student_only NODES=1 CAP_NODE_H=${CAP_NODE_H:-5.75} CLOCK=wall MAX_INPUT=65536 MAX_OUTPUT=16384 CONC=100 N_ATTEMPTS=1
VW=${VERIFY_WAIT:-2700}
TMIN=${DRIVER_TIME:-$(awk -v c=$CAP_NODE_H -v n=$NODES -v w=$VW 'BEGIN{printf "%d", c/(n+1)*60 + w/60 + 60}')}
JOB=$(sbatch --parsable --export=ALL -J eval_drv_$NAME -t $TMIN -o $LOGS/%x_%j.out $HERE/driver.sbatch | tail -1)
[ -n "$JOB" ] && [ "$JOB" -eq "$JOB" ] 2>/dev/null || { echo "sbatch failed: $JOB"; exit 1; }
for p in ${TUNNEL_PORTS//,/ }; do
  setsid nohup bash $OTA/data/r2egym/horizon/tunnel.sh $JOB $p > $LOGS/tunnel_${JOB}_$p.log 2>&1 < /dev/null &
done
echo "[$(date -Is)] driver job $JOB ($TMIN min) for serve job $SERVE_JOB, held-out run $NAME ($MODE); tunnels $TUNNEL_PORTS"
echo "  driver log: $LOGS/eval_drv_${NAME}_$JOB.out; run dir: $E/runs/$NAME"
