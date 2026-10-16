#!/bin/bash
# serve_submit.sh NODES N_STUDENT TIME [JOB_NAME] — submit data/relay/pilot/serve_relay.sbatch on Horizon: the first
# N_STUDENT nodes serve the student, the rest the teacher (4 x Qwen3.8 per node, ports 8000-8003). Prints the job id.
# Endpoint files then appear in $RELAY_EXP_DIR/endpoints/<jobid>.{student,teacher,meta} (or <jobid>.DEAD).
#   bash serve_submit.sh 2 1 01:00:00 relay_srv_test
set -euo pipefail
NODES=${1:?nodes}; NST=${2:?n_student}; TIME=${3:?time}; NAME=${4:-relay_srv}
# serve_env.sh would fall back to the old Stage-3 model and the router follows whatever is served, so a forgotten
# STUDENT_MODEL runs the wrong student with no error anywhere. Every caller names it; require that.
[ "$NST" -eq 0 ] || [ -n "${STUDENT_MODEL:-}" ] || { echo "set STUDENT_MODEL=<student HF dir> (e.g. \$SCRATCH/models/grug-datakit-sft-20260921)" >&2; exit 1; }
source "$(dirname "$0")/serve_env.sh"
mkdir -p $RELAY_EXP_DIR/logs $RELAY_EXP_DIR/endpoints
N_STUDENT=$NST sbatch --parsable --export=ALL -A ${SERVE_ACCOUNT:-CCR24067} -p ${SERVE_PARTITION:-debug} --exclude=${SERVE_EXCLUDE:-} \
  -o $RELAY_EXP_DIR/logs/%x_%j.out -J $NAME -N $NODES -t $TIME $RELAY_PILOT_DIR/serve_relay.sbatch | tail -1
