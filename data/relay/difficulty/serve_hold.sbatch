#!/bin/bash
#SBATCH -p debug
#SBATCH -A CCR24067
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH -t 02:00:00
#SBATCH -J sdiff
# serve_hold.sbatch — one student checkpoint served on one Horizon GB200 node for teacher-forced scoring:
# data/relay/pilot/serve_node.sh student = the MSA2 eval serve's arguments (TP1 x DP4 x EP, the model's template,
# temperature 1 / top_p 1 / top_k -1), with no EAGLE-3 draft (nothing is generated) and STUDENT_MAXLEN context.
#   STUDENT_MODEL=<dir> sbatch -o <exp>/logs/serve_%j.out serve_hold.sbatch
# Score from the login node with a step in this job (the server listens on the node's localhost:8000):
#   srun -p debug -A CCR24067 -t 00:30:00 --jobid=<job> --overlap -N1 -n1 <python> score.py --url http://localhost:8000/v1 ...
# The server stays up until the time limit or until <exp>/STOP_<job> exists.
set -eo pipefail
unset LD_PRELOAD                     # TACC XALT's NVML warning lands on stdout of every wrapped command
OTA=${OTA:-$HOME/snowball/ota-difficulty}
EXP=${EXP:-/scratch/11584/$USER/experiments/student_difficulty}
source $OTA/data/relay/horizon/serve_env.sh
export STUDENT_MODEL=${STUDENT_MODEL:?STUDENT_MODEL=<checkpoint dir>} STUDENT_DRAFT=none
export STUDENT_MAXLEN=${STUDENT_MAXLEN:-131072}
echo "serve_hold job=$SLURM_JOB_ID node=$(hostname) model=$STUDENT_MODEL maxlen=$STUDENT_MAXLEN ota=$(git -C $OTA rev-parse HEAD)"
setsid bash $OTA/data/relay/pilot/serve_node.sh student &
pg=$!
for _ in $(seq 1 90); do curl -sf --max-time 5 localhost:8000/health >/dev/null && break; sleep 10; done
curl -sf --max-time 5 localhost:8000/health >/dev/null && echo "SERVE_READY $(date -Is)" || echo "SERVE_NOT_READY $(date -Is)"
while kill -0 $pg 2>/dev/null && [ ! -f $EXP/STOP_$SLURM_JOB_ID ]; do sleep 20; done
echo "serve_hold stopping $(date -Is)"
kill -TERM -- -$pg 2>/dev/null || true
sleep 20
kill -KILL -- -$pg 2>/dev/null || true
