#!/bin/bash
# tblite_chain.sh <serve-jobid> <run-name> [smoke]: OpenThoughts-TBLite 2.0 (100 tasks x 1, 16 concurrent per server) of a
# Snowball checkpoint under the Marin eval policy of 2026-09-24 (tblite_marin_policy_0924.yaml, harbor 761fb516, Terminus-2).
# tb2p4_chain.sh with the TBLite tree and three knobs: waits for the serve endpoint, runs run_tb2.sh, waits for the harbor
# run to end, reruns the unscored (infrastructure) trials once ONLY if they exceed 10 % of the planned trials, writes
# $E/final_<name>.{json,txt,DONE}, then releases the serve node.
#   smoke          : run_tb2.sh's smoke (the first two tasks, 2 concurrent); no recovery pass.
#   SHARD=i/n      : this server takes every n-th task from i (run one chain per server; merge the job dirs to score:
#                    summarize_tb2.py reads one dir, so score the union with tblite_merge.py).
#   RELEASE=0      : keep the serve job up afterwards (a smoke before the full run on the same node).
#   EXCLUDE_TASKS  : tasks with no Daytona snapshot (default empty); they count as infrastructure losses.
# Run it in tmux tblite_<name> on the login node, never tb2_<name> (run_tb2.sh's own session). Log: $E/pipeline_<name>.log
set -o pipefail; JOB=${1:?serve job id}; NAME=${2:?run name}; MODE=${3:-full}
C=/e/project1/transfernetx/lee27/code; W=$C/tb2; E=/e/fscratch/reformo/lee27/experiments/tb2
J=/e/data1/mmlaion/lee27/experiments/tb2_jobs/$NAME; KEYF=/e/fscratch/reformo/lee27/keys/daytona_eval.env
PY=$C/envs/snowball-v2/bin/python; HARBOR=$C/envs/snowball-v2/bin/harbor
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export HARBOR_SRC=$C/harbor-p0924/src HARBOR_SHA=761fb516 POLICY_FILE=tblite_marin_policy_0924.yaml
export TASKS=/e/fscratch/reformo/lee27/tasks/openthoughts_tblite_2_0 NTASKS=100
export EXCLUDE_TASKS=${EXCLUDE_TASKS-}
log() { echo "$(date -Is) $*" | tee -a $E/pipeline_$NAME.log; }
release() { [ "${RELEASE:-1}" = 0 ] && { log "keeping serve job $JOB (RELEASE=0)"; return; }; log "releasing serve job $JOB"; scancel $JOB; }
log "waiting for endpoint of serve job $JOB (mode $MODE, shard '${SHARD:-}', policy $POLICY_FILE, harbor $HARBOR_SHA, exclude '$EXCLUDE_TASKS')"
while [ ! -f $E/endpoints/$JOB ]; do
  squeue -h -j $JOB -o %T | grep -qE "PENDING|RUNNING|CONFIGURING" || { log "FAILED: serve job $JOB left the queue before writing an endpoint"; exit 1; }
  sleep 30
done
log "endpoint $(cat $E/endpoints/$JOB)"
bash $W/run_tb2.sh $JOB $NAME $MODE 2>&1 | tee -a $E/pipeline_$NAME.log || { log "FAILED: launch"; release; exit 1; }
while ! grep -q RUN_DONE $E/logs/run_$NAME.log 2>/dev/null; do
  squeue -h -j $JOB -o %T | grep -q RUNNING || { log "FAILED: serve job $JOB ended before the run finished; no recovery pass, finish by hand"; exit 2; }
  sleep 60
done
log "run finished: $(grep RUN_DONE $E/logs/run_$NAME.log | tail -1)"
$PY $W/summarize_tb2.py $J --json > $E/final_${NAME}_pre.json
LOST=$($PY -c "import json; d=json.load(open('$E/final_${NAME}_pre.json')); n=len(json.load(open('$J/config.json'))['tasks']); print(n-d['scored'], n)")
log "unscored (infrastructure) trials / planned: $LOST"
if [ "$MODE" != smoke ] && [ "$(echo $LOST | awk '{print ($1 > 0.10*$2) ? 1 : 0}')" = 1 ]; then
  log "over 10 %: one recovery pass (harbor jobs resume re-runs only the unscored trials)"
  ( export PYTHONPATH=$HARBOR_SRC; set -a; source $KEYF; set +a; cd $W; $HARBOR jobs resume -p $J -f CancelledError 2>&1 | tail -5 ) | tee -a $E/pipeline_$NAME.log
fi
$PY $W/summarize_tb2.py $J --json > $E/final_$NAME.json; $PY $W/summarize_tb2.py $J | tee $E/final_$NAME.txt | tee -a $E/pipeline_$NAME.log
touch $E/final_$NAME.DONE $E/pipeline_$NAME.DONE
release
