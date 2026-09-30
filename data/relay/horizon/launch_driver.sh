#!/bin/bash
# launch_driver.sh <serve job> <run name> [smoke] — Horizon login node. Submits driver.sbatch for a relay run on the serve
# job serve_submit.sh started, and one login-side tunnel.sh per TUNNEL_PORTS port for it (setsid, outlives this shell,
# exits when the driver job ends). Replaces launch_relay.sh's Jupiter gating (check_decide.py / relay_plan.py on Jupiter
# run dirs) with explicit settings: the full relay runs' settings (launch_relay.sh, 2026-09-26/27) as defaults, each
# overridable from the environment, plus the finetuned-student gates (ARM_GATES). Nothing here judges whether to run.
#
#   S=$(STUDENT_MODEL=<arm A snapshot> bash serve_submit.sh 8 4 06:00:00 relay_srv_<name>)   # the serving side's script
#   NODES=8 CAP_NODE_H=45 TREE=... TASK_LIST=... bash launch_driver.sh $S <name>
# The router's student tokenizer comes from the serve job's endpoints .meta (STUDENT_TOKENIZER=meta), so it always matches
# the student that job actually serves.
#
# Required: NODES (serve nodes) and CAP_NODE_H (node-hour ceiling for serve + driver nodes together).
# Driver job time: CAP_NODE_H / (NODES + 1) h + the verify window + 1 h of queue slack for the serve job (DRIVER_TIME).
set -uo pipefail
SERVE_JOB=${1:?serve job id}; NAME=${2:?run name}; MODE=${3:-full}
HERE=$(cd "$(dirname "$0")" && pwd); OTA=$(cd $HERE/../../.. && pwd)
[ -f $HERE/serve_env.sh ] && source $HERE/serve_env.sh   # RELAY_EXP_DIR (the serve side's experiment dir)
S=${SCRATCH_DIR:-/scratch/11584/$USER}
export E=${E:-${RELAY_EXP_DIR:-$S/experiments/relay/pilot}}; LOGS=$E/logs; mkdir -p $LOGS $E/runs
[ -e $E/runs/$NAME ] && { echo "$E/runs/$NAME exists; pick a new run name"; exit 1; }
export SERVE_JOB RUN_NAME=$NAME RUN_MODE=$MODE OTA
export TUNNEL_PORTS=${TUNNEL_PORTS:-18080,18081}
export STUDENT_TOKENIZER=${STUDENT_TOKENIZER:-meta}   # meta: the served student's tokenizer.json, read from the serve job's .meta
[ "$STUDENT_TOKENIZER" = meta ] || [ -f $STUDENT_TOKENIZER ] || { echo "no tokenizer at $STUDENT_TOKENIZER"; exit 1; }
export TREE=${TREE:-$S/relay/calibforge_tree}
# the full runs' relay settings (launch_relay.sh's run_pilot.sh line)
export ARMS=${ARMS:-relay_repair} CLOCK=${CLOCK:-repair} CTX_BUDGET=${CTX_BUDGET:-32000} ROW_MAX=${ROW_MAX:-65536} ROW_RESERVE=${ROW_RESERVE:-8192}
export MAX_INPUT=${MAX_INPUT:-131072} TEACHER_MAX_TOKENS=${TEACHER_MAX_TOKENS:-32768} BALANCE=${BALANCE:-active} STAGGER_SEC=${STAGGER_SEC:-180}
export GATE_MIN=${GATE_MIN:-15} GATE_LAT=${GATE_LAT:-30} GATE_KV=${GATE_KV:-0.90} TEACHER_GUARD=${TEACHER_GUARD:-1} VERIFY_NOTE=${VERIFY_NOTE:-1}
export CONC=${CONC:-128} NODES=${NODES:?NODES = serve nodes} CAP_NODE_H=${CAP_NODE_H:?CAP_NODE_H = node-hour ceiling, serve + driver}
export N_ATTEMPTS=${N_ATTEMPTS:-2} TASK_LIST=${TASK_LIST:-} SHUFFLE_SEED=${SHUFFLE_SEED:-}
export STOP_AFTER=${STOP_AFTER:-300} STOP_OVF=${STOP_OVF:-0.30} STOP_FMT=${STOP_FMT:-0.98} STOP_HERR=${STOP_HERR:-0.10}
export HERR_EXCLUDE=${HERR_EXCLUDE:-} TARGET_FAIL=${TARGET_FAIL:-0} TARGET_BASE=${TARGET_BASE:-0}
# the finetuned student's gates (arm_gates.py; DRIVER.md has the reasoning): flag, never cancel, by default
export ARM_GATES=${ARM_GATES:-1} TAKEOVER_MIN=${TAKEOVER_MIN:-0.50} TAKEOVER_MAX=${TAKEOVER_MAX:-0.95} TAKEOVER_AFTER=${TAKEOVER_AFTER:-100}
export TAKEOVER_ACTION=${TAKEOVER_ACTION:-flag} ACCEPT_FLAG=${ACCEPT_FLAG:-1.8} ACCEPT_STOP=${ACCEPT_STOP:-0}
VW=${VERIFY_WAIT:-2700}
TMIN=${DRIVER_TIME:-$(awk -v c=$CAP_NODE_H -v n=$NODES -v w=$VW 'BEGIN{printf "%d", c/(n+1)*60 + w/60 + 60}')}
JOB=$(sbatch --parsable --export=ALL -J relay_drv_$NAME -t $TMIN -o $LOGS/%x_%j.out $HERE/driver.sbatch | tail -1)
[ -n "$JOB" ] && [ "$JOB" -eq "$JOB" ] 2>/dev/null || { echo "sbatch failed: $JOB"; exit 1; }
for p in ${TUNNEL_PORTS//,/ }; do
  setsid nohup bash $OTA/data/r2egym/horizon/tunnel.sh $JOB $p > $LOGS/tunnel_${JOB}_$p.log 2>&1 < /dev/null &
done
echo "[$(date -Is)] driver job $JOB ($TMIN min) for serve job $SERVE_JOB, run $NAME ($MODE); tunnels $TUNNEL_PORTS"
echo "  driver log: $LOGS/relay_drv_${NAME}_$JOB.out; run dir: $E/runs/$NAME; tunnels: $LOGS/tunnel_${JOB}_*.log"
