#!/bin/bash
# run_claimgate.sh <serve jobid> <run name> — claim gate driver on the Jupiter login node (research note
# ai_memory/active/snowball-sft/research/2026-09-28_opd_ref_evidence.md § "Claim gate test").
#
# The serve job is data/relay/pilot/serve_relay.sbatch with STUDENT_MODEL = the ns_c step-24 export: N_STUDENT student
# nodes, the rest Qwen3.8-27B (the judge). This script waits for its endpoint files, checks the judge answers a
# 1-token /completions with logprobs, starts the relay router in --mode student --judge-gate, runs one harbor job
# (claimgate.yaml, apptainer through bridge 9930; a JUWELS fleet must be up), then cancels the serve job and runs the
# readout. Every episode's arm, judge score and gate action is in $R/router/*.jsonl.
#
#   TASK_LIST (default: all 248 mid-pass-rate tasks)  N_ATTEMPTS (4)  CONC (96)  THRESH (-2.25)  TREAT_FRAC (0.5)
#   KEEP_SERVE=1 leaves the serve job up at the end (the stage C smoke runs first on the main run's serve job)
set -uo pipefail
JOB=${1:?serve jobid}; NAME=${2:?run name}
HERE=$(cd "$(dirname "$0")" && pwd); OTA=$(cd $HERE/../.. && pwd)
C=/e/project1/transfernetx/lee27/code
PY=$C/envs/snowball-v2/bin/python; HARBOR=$C/envs/snowball-v2/bin/harbor
export PYTHONPATH=${HARBOR_SRC:-$C/harbor-claimgate/src} OMP_NUM_THREADS=1
EP=/e/fscratch/reformo/lee27/experiments/relay/pilot/endpoints
TREE=/e/fscratch/reformo/lee27/tasks/claimgate_mid248
X=/e/data1/mmlaion/lee27/experiments/claim_gate
TASK_LIST=${TASK_LIST:-$X/TASKS.txt}; N_ATTEMPTS=${N_ATTEMPTS:-4}; CONC=${CONC:-96}
THRESH=${THRESH:--2.25}; TREAT_FRAC=${TREAT_FRAC:-0.5}; RPORT=${RPORT:-8300}
R=$X/runs/$NAME; mkdir -p $R/jobs
log() { echo "$(date '+%F %T') $*" | tee -a $R/driver.log; }
abort() { log "ABORT: $*"; [ -n "${RPID:-}" ] && kill $RPID 2>/dev/null; [ "${KEEP_SERVE:-0}" = 1 ] || scancel $JOB; exit 1; }
curl -s -m 5 localhost:9930/status | grep -q '"workers_alive": true' || abort "bridge 9930 has no live workers (start the fleet first)"
log "run_claimgate $NAME serve=$JOB tasks=$TASK_LIST x$N_ATTEMPTS conc=$CONC thresh=$THRESH treat_frac=$TREAT_FRAC harbor=$PYTHONPATH"

# 1. endpoints (serve_relay.sbatch writes them after its own smoke completions)
while [ ! -f $EP/$JOB.student ] || [ ! -f $EP/$JOB.teacher ]; do
  [ -f $EP/$JOB.DEAD ] && abort "serve job died: $(cat $EP/$JOB.DEAD)"
  squeue -h -j $JOB | grep -q . || abort "serve job $JOB left the queue before its endpoints were written"
  sleep 30
done
SURL=$(cat $EP/$JOB.student); TURL=$(cat $EP/$JOB.teacher)
log "endpoints student=$SURL judge=$TURL"
END=$(date -d "$(squeue -h -j $JOB -o %e)" +%s); DEADLINE=$((END - 900))   # episodes end 15 min before the serve wall
log "serve ends $(date -d @$END '+%T'); router deadline $(date -d @$DEADLINE '+%T')"

# 2. the judge answers the way the router will ask it
for u in ${TURL//,/ }; do
  out=$(curl -s -m 120 $u/completions -H 'Content-Type: application/json' \
    -d '{"model":"qwen38","prompt":"<|im_start|>user\nIs 2+2=4? Answer with one word, Yes or No.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n","max_tokens":1,"temperature":0,"logprobs":20}')
  echo "$out" | $PY -c 'import json,sys; r=json.load(sys.stdin); t=r["choices"][0]["logprobs"]["top_logprobs"][0]; print("JUDGE_SMOKE", r["choices"][0]["text"], sorted(t.items(), key=lambda x: -x[1])[:3])' \
    | tee -a $R/driver.log | grep -q JUDGE_SMOKE || abort "judge smoke failed at $u: ${out:0:300}"
done

# 3. router
$PY $OTA/data/relay/router/relay_router.py --mode student --arm claimgate --port $RPORT --log-dir $R/router \
  --student-url $SURL --student-model snowball --budget-mode off --balance active --engine-metrics \
  --deadline-epoch $DEADLINE --judge-gate --judge-url $TURL --judge-threshold $THRESH --judge-treat-frac $TREAT_FRAC \
  > $R/router.log 2>&1 &
RPID=$!
for i in $(seq 1 90); do grep -q RELAY_ROUTER_READY $R/router.log 2>/dev/null && break
  [ -f $R/router/FATAL ] && abort "router health check failed: $(head -3 $R/router/FATAL)"; sleep 5; done
grep -q RELAY_ROUTER_READY $R/router.log || abort "router not ready after 450 s"
log "router ready on $RPORT (pid $RPID)"

# 4. harbor
$PY - "$HERE/claimgate.yaml" "$R" "$NAME" "$TREE" "$TASK_LIST" "$N_ATTEMPTS" "$CONC" "http://127.0.0.1:$RPORT/v1" <<'PY'
import sys, yaml
src, r, name, tree, tl, n, conc, api = sys.argv[1:]
s = open(src).read()
for k, v in dict(JOB_NAME=name, JOBS_DIR=r + '/jobs', N_ATTEMPTS=n, CONC=conc, API_BASE=api).items():
    s = s.replace('__%s__' % k, v)
c = yaml.safe_load(s)
c['tasks'] = [{'path': '%s/%s' % (tree, t.strip().rsplit('/', 1)[-1])} for t in open(tl) if t.strip()]
yaml.safe_dump(c, open('%s/%s.yaml' % (r, name), 'w'), sort_keys=False)
print('tasks', len(c['tasks']))
PY
log "harbor start ($(grep -c . $TASK_LIST) tasks x $N_ATTEMPTS)"
$HARBOR jobs start --config $R/$NAME.yaml > $R/harbor.log 2>&1
log "harbor exit $?"
kill $RPID 2>/dev/null
if [ "${KEEP_SERVE:-0}" = 1 ]; then log "serve $JOB kept"; else scancel $JOB; log "serve $JOB cancelled"; fi
$PY $HERE/claimgate_readout.py --run $R > $R/readout.md 2>&1; cat $R/readout.md | tee -a $R/driver.log
