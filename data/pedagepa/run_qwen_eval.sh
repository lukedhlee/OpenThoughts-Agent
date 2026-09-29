#!/bin/bash
# run_qwen_eval.sh <serve-jobid> <run-name> <student run yaml> <harbor src> <harbor sha> <concurrency>
# Jupiter login node. Runs Qwen3.8-27B (served by data/relay/pilot/serve_relay.sbatch with N_STUDENT=0, endpoint
# file $EP/<jobid>.teacher) on exactly the tasks and harbor settings of an existing 09-21 run, so the two are same-task
# pairs: the student's rendered yaml is copied with only job_name, api_base, model_name (hosted_vllm/qwen38),
# the concurrency, and extra_body (skip_special_tokens is a Snowball setting) changed. Writes <jobs>/<run-name>.
set -uo pipefail
JOB=${1:?}; NAME=${2:?}; SRC_YAML=${3:?}; HARBOR_SRC=${4:?}; HARBOR_SHA=${5:?}; CONC=${6:-16}
C=/e/project1/transfernetx/lee27/code; PY=$C/envs/snowball-v2/bin/python; HARBOR=$C/envs/snowball-v2/bin/harbor
EP=/e/fscratch/reformo/lee27/experiments/relay/pilot/endpoints
JOBS=/e/data1/mmlaion/lee27/experiments/pedagepa/jobs; OUT=/e/data1/mmlaion/lee27/experiments/pedagepa/runs; mkdir -p $JOBS $OUT
KEYF=/e/fscratch/reformo/lee27/keys/daytona_eval.env
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
log() { echo "[$(date -Is)] $*" | tee -a $OUT/$NAME.log; }
[ -d $JOBS/$NAME ] && { log "$JOBS/$NAME exists"; exit 1; }
[ "$(git -C ${HARBOR_SRC%/src} rev-parse --short=8 HEAD)" = "${HARBOR_SHA:0:8}" ] || { log "harbor $HARBOR_SRC not at $HARBOR_SHA"; exit 1; }
log "waiting for serve job $JOB"
while [ ! -s $EP/$JOB.teacher ]; do
  [ -f $EP/$JOB.DEAD ] && { log "serve DEAD: $(cat $EP/$JOB.DEAD)"; exit 1; }
  squeue -h -j $JOB | grep -q . || { log "serve job gone"; exit 1; }
  sleep 30
done
URL=$(cut -d, -f1 $EP/$JOB.teacher); log "teacher at $URL"
CFG=$OUT/$NAME.yaml
$PY - "$SRC_YAML" "$CFG" "$NAME" "$URL" "$JOBS" "$CONC" <<'PY'
import sys, yaml
src, dst, name, url, jobs, conc = sys.argv[1:7]
c = yaml.safe_load(open(src))
c['job_name'] = name; c['jobs_dir'] = jobs; c['n_concurrent_trials'] = int(conc)
a = c['agents'][0]; a['model_name'] = 'hosted_vllm/qwen38'; a['kwargs']['api_base'] = url
a['kwargs'].pop('extra_body', None)
a['kwargs'].setdefault('trajectory_config', {})['raw_content'] = True
yaml.safe_dump(c, open(dst, 'w'), sort_keys=False)
print(len(c['tasks']), 'tasks', a)
PY
$PY -c "import yaml,sys; sys.path.insert(0,'$HARBOR_SRC'); from harbor_config.models.job.config import JobConfig; JobConfig.model_validate(yaml.safe_load(open('$CFG'))); print('policy validates')" || { log "config invalid"; exit 1; }
set -a; source $KEYF; set +a
log "harbor start $NAME (conc $CONC)"
( export PYTHONPATH=$HARBOR_SRC; cd $OUT; $HARBOR run -c $CFG -y ) >> $OUT/$NAME.harbor.log 2>&1
log "harbor exit $?"
touch $OUT/$NAME.DONE
