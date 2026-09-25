#!/bin/bash
# run_smoke.sh <serve-jobid> <run-name> <qwen38|snowball|scripted>
# Jupiter login node, inside tmux: wait for the serve job's endpoint, run mini-swe-agent-host on the CalibForge smoke
# tasks on Daytona, check the rendered prompt (Qwen), cancel the serve job the moment harbor exits, then write the
# readout. "scripted" runs against scripted_openai.py on this login node (no GPU; serve-jobid is ignored).
#
#   tmux new -d -s msa_<name> "bash run_smoke.sh <jobid> <name> qwen38"
#
# Harbor is a plain clone of marin-community/harbor lukedhlee/mini-swe-host-smoke (mini-swe-host + the setup-files
# hook) on PYTHONPATH, with mini-swe-agent 2.4.6 in a --no-deps side dir; the interpreter is snowball-v2's, as for
# the TB2 evals. Never reuse a run name.
set -uo pipefail
JOB=${1:?serve job id}; NAME=${2:?run name}; MODEL=${3:?qwen38|snowball|scripted}
C=/e/project1/transfernetx/lee27/code
HERE=$(cd "$(dirname "$0")" && pwd)
HARBOR_SRC=${HARBOR_SRC:-$C/harbor-mini-swe-host/src}
MSA=${MSA:-$C/envs/msa-2.4.6}
PY=$C/envs/snowball-v2/bin/python; HARBOR=$C/envs/snowball-v2/bin/harbor
KEYF=/e/fscratch/reformo/lee27/keys/daytona_eval.env
TASKS=${TASKS:-/e/fscratch/reformo/lee27/tasks/calibforge_smoke8}
E=/e/fscratch/reformo/lee27/experiments/mini_swe_host; JOBS=/e/data1/mmlaion/lee27/experiments/mini_swe_host_jobs
mkdir -p $E/logs $E/runs $JOBS
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1   # login-node pid cap
export PYTHONPATH=$HARBOR_SRC:$MSA
LOG=$E/logs/run_$NAME.log
exec > >(tee -a $LOG) 2>&1
echo "run_smoke $NAME model=$MODEL job=$JOB harbor=$(git -C ${HARBOR_SRC%/src} rev-parse --short=8 HEAD) start=$(date -Is)"
[ -d $JOBS/$NAME ] && { echo "$JOBS/$NAME exists; pick a new name"; exit 1; }
case $MODEL in
  qwen38)   ENDPOINTS=$E/endpoints; SERVED=qwen38; EXTRA='{}'; INTERLEAVED=true;;
  snowball) ENDPOINTS=/e/fscratch/reformo/lee27/experiments/tb2/endpoints; SERVED=snowball; EXTRA='{"skip_special_tokens": false}'; INTERLEAVED=false;;
  scripted) SERVED=scripted; EXTRA='{}'; INTERLEAVED=false;;
  *) echo "model must be qwen38, snowball or scripted"; exit 2;;
esac
cancel_serve() { [ $MODEL = scripted ] || { scancel $JOB 2>/dev/null; echo "scancel $JOB $(date -Is)"; }; }
# 1. wait for the endpoint (the serve job writes it after one real completion)
if [ $MODEL = scripted ]; then
  PORT=$((20000 + RANDOM % 20000))
  $PY $HERE/scripted_openai.py --port $PORT --log $E/logs/scripted_$NAME.jsonl &
  SCRIPTED_PID=$!; trap 'kill $SCRIPTED_PID 2>/dev/null' EXIT; sleep 2
  URL=http://127.0.0.1:$PORT/v1
else
  URL=""
  for i in $(seq 1 120); do
    [ -f $ENDPOINTS/$JOB ] && { URL=$(cat $ENDPOINTS/$JOB); break; }
    STATE=$(squeue -h -j $JOB -o %T 2>/dev/null)
    [ -z "$STATE" ] && { echo "serve job $JOB is gone before its endpoint appeared"; exit 1; }
    sleep 30
  done
  [ -n "$URL" ] || { echo "no endpoint after 60 min"; cancel_serve; exit 1; }
fi
echo "endpoint $URL ($(date -Is))"
curl -sf --max-time 20 $URL/models | grep -q "\"$SERVED\"" || { echo "no model $SERVED at $URL"; cancel_serve; exit 1; }
set -a; source $KEYF; set +a
curl -sf --max-time 20 -H "Authorization: Bearer $DAYTONA_API_KEY" https://app.daytona.io/api/api-keys/current >/dev/null || { echo "Daytona key rejected"; cancel_serve; exit 1; }
# 2. render the policy
CFG=$E/runs/$NAME.yaml
sed "s#__JOB_NAME__#$NAME#; s#__JOBS_DIR__#$JOBS#; s#__TREE__#$TASKS#; s#__MODEL_NAME__#hosted_vllm/$SERVED#; s#__API_BASE__#$URL#; s#__EXTRA_BODY__#$EXTRA#; s#__INTERLEAVED__#$INTERLEAVED#" \
  $HERE/calibforge_mini_swe.yaml > $CFG
$PY -c "import yaml; from harbor_config.models.job.config import JobConfig; JobConfig.model_validate(yaml.safe_load(open('$CFG'))); print('config validates')" || { cancel_serve; exit 1; }
# 3. run
$HARBOR jobs start --config $CFG
echo "HARBOR_EXIT $? $(date -Is)"
# 4. the prompt check needs the server; then release the node
[ $MODEL = qwen38 ] && $PY $HERE/prompt_check.py $URL $SERVED $JOBS/$NAME
cancel_serve
$PY $HERE/readout.py $JOBS/$NAME > $E/runs/$NAME.readout.jsonl
tail -1 $E/runs/$NAME.readout.jsonl
echo "RUN_DONE $(date -Is)"
