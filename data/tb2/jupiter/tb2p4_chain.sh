#!/bin/bash
# tb2p4_chain.sh <serve-jobid> <run-name>: one Terminal-Bench 2 trial (89 tasks x 1, 16 concurrent) of a Snowball checkpoint
# under the Marin eval policy of 2026-09-24 (tb2_marin_policy_0924.yaml, harbor 761fb516), for pass@k studies that put each
# trial on its own one-node server. Waits for the serve endpoint, runs run_tb2.sh, waits for the harbor run to end, reruns
# the unscored (infrastructure) trials once ONLY if they exceed 10 % of the planned trials, writes
# $E/final_<name>.{json,txt,DONE}, then releases the serve node.
# EXCLUDE_TASKS (default train-fasttext: its Daytona image is gone and the org's 40 snapshot slots are full, 2026-09-28)
# are left out and count as infrastructure losses.
# Run it in tmux pl_<name> on the login node, never tb2_<name> (run_tb2.sh's own session). Log: $E/pipeline_<name>.log
# 2026-09-28 pass@4 study: launched with this script, then its waiting tail was handed to tb2p4_finish.sh and the per-run
# recovery replaced by one pooled wave (tb2p4_rerun.sh), which costs one server instead of one per run.
set -o pipefail; JOB=${1:?serve job id}; NAME=${2:?run name}
C=/e/project1/transfernetx/lee27/code; W=$C/tb2; E=/e/fscratch/reformo/lee27/experiments/tb2
J=/e/data1/mmlaion/lee27/experiments/tb2_jobs/$NAME; KEYF=/e/fscratch/reformo/lee27/keys/daytona_eval.env
PY=$C/envs/snowball-v2/bin/python; HARBOR=$C/envs/snowball-v2/bin/harbor
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export HARBOR_SRC=$C/harbor-p0924/src HARBOR_SHA=761fb516 POLICY_FILE=tb2_marin_policy_0924.yaml
export EXCLUDE_TASKS=${EXCLUDE_TASKS-train-fasttext}
log() { echo "$(date -Is) $*" | tee -a $E/pipeline_$NAME.log; }
release() { log "releasing serve job $JOB"; scancel $JOB; }
log "waiting for endpoint of serve job $JOB (policy $POLICY_FILE, harbor $HARBOR_SHA, exclude '$EXCLUDE_TASKS')"
while [ ! -f $E/endpoints/$JOB ]; do
  squeue -h -j $JOB -o %T | grep -qE "PENDING|RUNNING|CONFIGURING" || { log "FAILED: serve job $JOB left the queue before writing an endpoint"; exit 1; }
  sleep 30
done
log "endpoint $(cat $E/endpoints/$JOB)"
bash $W/run_tb2.sh $JOB $NAME 2>&1 | tee -a $E/pipeline_$NAME.log || { log "FAILED: launch"; release; exit 1; }
while ! grep -q RUN_DONE $E/logs/run_$NAME.log 2>/dev/null; do
  squeue -h -j $JOB -o %T | grep -q RUNNING || { log "FAILED: serve job $JOB ended before the run finished; no recovery pass, finish by hand"; exit 2; }
  sleep 60
done
log "run finished: $(grep RUN_DONE $E/logs/run_$NAME.log | tail -1)"
$PY $W/summarize_tb2.py $J --json > $E/final_${NAME}_pre.json
LOST=$($PY -c "import json; d=json.load(open('$E/final_${NAME}_pre.json')); n=len(json.load(open('$J/config.json'))['tasks']); print(n-d['scored'], n)")
log "unscored (infrastructure) trials / planned: $LOST"
if [ "$(echo $LOST | awk '{print ($1 > 0.10*$2) ? 1 : 0}')" = 1 ]; then
  log "over 10 %: one recovery pass (harbor jobs resume re-runs only the unscored trials)"
  ( export PYTHONPATH=$HARBOR_SRC; set -a; source $KEYF; set +a; cd $W; $HARBOR jobs resume -p $J -f CancelledError 2>&1 | tail -5 ) | tee -a $E/pipeline_$NAME.log
fi
$PY $W/summarize_tb2.py $J --json > $E/final_$NAME.json; $PY $W/summarize_tb2.py $J | tee $E/final_$NAME.txt | tee -a $E/pipeline_$NAME.log
touch $E/final_$NAME.DONE $E/pipeline_$NAME.DONE
release
