#!/bin/bash
# run_full.sh <core-jobid> <burst-jobid> <run-name>
# Jupiter login node, inside tmux. The full Terminus-2 relay run on CalibForge (spec: notes/relay/relay_full_t2.md).
#
# Servers (serve_relay.sbatch, two jobs so the second can be released on its own):
#   core   N_STUDENT=2, 6 nodes: 2 x 09-21 + 4 x Qwen3.8          (--time = the run's wall ceiling)
#   burst  N_STUDENT=0, 4 nodes: 4 x Qwen3.8                        (released when phase 1's baseline drains)
# Arms (one router each, both pinning every episode to one server per role):
#   control       Qwen from scratch, 1 rollout per task of the pool                         phase 1
#   relay_repair  the student with parse_error repairs + sticky takeovers, 1 rollout / task  phase 1
#                 + 2 more rollouts on each task control solved in phase 1                 phase 2 (<name>_relay_repair_p2)
# Settings are run 3's: thinking-on student, summarization off, 64k input, student clock paused on repairs, the
# calibrated triggers, the student's thinking stripped from the teacher's view.
#
#   1. pre-flight as run_pilot.sh (harbor pin, tree = full_pool.txt, disjoint from the held-out split, key, snapshots)
#   2. endpoints of both jobs; teacher list file = core + burst teachers
#   3. routers (control, relay_repair) with --teacher-url-file and the deadline (core start + CORE_WALL - margin)
#   4. harbor: control (CONC_BASE, default 400 = 50 per Qwen node) and relay_repair phase 1 (CONC_RELAY, default 200 =
#      100 per student node)
#   5. watch every 60 s: FATAL / DEAD / stall -> abort; early gate at 25 min (readout --gate early) and the latency
#      gate at 15 min (teacher p50 > LAT_WARN s: warn; > LAT_ABORT s: abort); node-hour cap CAP_NODE_H over both jobs
#      -> stop. When control is done: phase-2 task list (control passes), phase 2 starts once phase 1 relay has at
#      most P2_START trials running; the burst teachers are dropped from the list, drained (no in-flight requests,
#      or 10 min) and their job cancelled.
#   6. both relay jobs done -> routers stop, core released, sandboxes cleaned, readout + select_kept.py
set -uo pipefail
CORE=${1:?core serve job}; BURST=${2:?burst serve job}; NAME=${3:?run name}
C=/e/project1/transfernetx/lee27/code
HERE=$(cd "$(dirname "$0")" && pwd)
ROUTER=$HERE/../router/relay_router.py
PY=${PY:-$C/envs/snowball-v2/bin/python}; HARBOR=${HARBOR:-$C/envs/snowball-v2/bin/harbor}
HARBOR_SRC=${HARBOR_SRC:-$C/harbor-terminus2-relay/src}; HARBOR_SHA=${HARBOR_SHA:-89098635}
TREE=${TREE:-/e/fscratch/reformo/lee27/tasks/calibforge_full2043}
KEYF=${KEYF:-/e/fscratch/reformo/lee27/keys/daytona_eval.env}
E=/e/fscratch/reformo/lee27/experiments/relay/pilot; EP=$E/endpoints
R=$E/runs/$NAME
JOBS_ROOT=${JOBS_ROOT:-/e/data1/mmlaion/lee27/experiments/relay_full_jobs}
CONC_BASE=${CONC_BASE:-400}; CONC_RELAY=${CONC_RELAY:-200}; CONC_P2=${CONC_P2:-200}; P2_START=${P2_START:-20}
CAP_NODE_H=${CAP_NODE_H:-35}; DEADLINE_MARGIN=${DEADLINE_MARGIN:-600}
EARLY_MIN=${EARLY_MIN:-25}; LAT_MIN=${LAT_MIN:-15}; LAT_WARN=${LAT_WARN:-30}; LAT_ABORT=${LAT_ABORT:-90}
STALL_MIN=${STALL_MIN:-15}; UP_WAIT=${UP_WAIT:-2400}; VERIFY_WAIT=${VERIFY_WAIT:-2700}; DRAIN_WAIT=${DRAIN_WAIT:-600}
PORT0=${PORT0:-$((21000 + RANDOM % 8000))}; PB=$PORT0; PR=$((PORT0 + 1))
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
[ -d $R ] && { echo "$R exists"; exit 1; }
mkdir -p $R $JOBS_ROOT; ln -s $JOBS_ROOT $R/jobs
exec > >(tee -a $R/driver.log) 2>&1
log() { echo "[$(date -Is)] $*"; }
HB=""; HR=""; HP2=""; RB=""; RR=""; CORE_UP=1; BURST_UP=1
nodes_of() { sacct -j $1 -X -n -o NNodes 2>/dev/null | head -1 | tr -d ' '; }
elapsed_of() { sacct -j $1 -X -n -o ElapsedRaw 2>/dev/null | head -1 | tr -d ' '; }
node_h() { awk -v a="$(elapsed_of $CORE)" -v na="$(nodes_of $CORE)" -v b="$(elapsed_of $BURST)" -v nb="$(nodes_of $BURST)" \
             'BEGIN{printf "%.2f", (a*na + b*nb)/3600}'; }
release() { local j=$1; scancel $j; [ $j = $CORE ] && CORE_UP=0; [ $j = $BURST ] && BURST_UP=0; log "released job $j: $2"; }
stop_harbor() { for p in $HB $HR $HP2; do kill -INT $p 2>/dev/null; done; sleep 30; for p in $HB $HR $HP2; do kill -TERM $p 2>/dev/null; done; }
cleanup_sandboxes() { local j=(); for d in $JOBS_ROOT/${NAME}_*; do [ -d $d ] && j+=($d); done
                      [ ${#j[@]} -gt 0 ] && $PY $HERE/cleanup_sandboxes.py --key-file $KEYF --delete "${j[@]}" 2>&1 | tail -2; }
write_meta() { printf 'core=%s\nburst=%s\nnode_hours=%s\nharbor=%s\nota=%s\nend=%s\n' $CORE $BURST "$(node_h)" \
               "$(git -C ${HARBOR_SRC%/src} rev-parse --short=8 HEAD)" "$(git -C $HERE rev-parse --short=8 HEAD)" "$(date -Is)" > $R/run.meta; }
abort() { log "ABORT: $*"; echo "$*" > $R/ABORT; stop_harbor; kill -TERM $RB $RR 2>/dev/null
          [ $CORE_UP = 1 ] && release $CORE abort; [ $BURST_UP = 1 ] && release $BURST abort; cleanup_sandboxes; write_meta; exit 1; }
running() { local d=$JOBS_ROOT/${NAME}_$1; [ -d $d ] || { echo 0; return; }
            echo $(( $(find $d -mindepth 1 -maxdepth 1 -type d | wc -l) - $(find $d -mindepth 2 -maxdepth 2 -name result.json | wc -l) )); }
log "run_full $NAME core=$CORE burst=$BURST conc base/relay/p2=$CONC_BASE/$CONC_RELAY/$CONC_P2 cap=$CAP_NODE_H node-h"

# ---- 1. pre-flight ------------------------------------------------------------------------------------------------
[ "$(git -C ${HARBOR_SRC%/src} rev-parse --short=8 HEAD)" = "$HARBOR_SHA" ] || { log "harbor not at $HARBOR_SHA"; scancel $CORE $BURST; exit 1; }
diff -q <(sort $TREE/TASKS.txt) <(sort $HERE/full_pool.txt) >/dev/null && [ -f $TREE/router_tasks.json ] || { log "tree is not full_pool.txt"; scancel $CORE $BURST; exit 1; }
[ -z "$(comm -12 <(sort $TREE/TASKS.txt) <(sort $HERE/calibforge_heldout300.txt))" ] || { log "pool overlaps the held-out split"; scancel $CORE $BURST; exit 1; }
set -a; source $KEYF; set +a
curl -sf --max-time 20 -H "Authorization: Bearer $DAYTONA_API_KEY" https://app.daytona.io/api/api-keys/current >/dev/null || { log "Daytona key rejected"; scancel $CORE $BURST; exit 1; }
PYTHONPATH=$HARBOR_SRC $PY $HERE/../../calibforge/daytona/snapshot_census.py --key-file $KEYF --tree $TREE > $R/snapshot_census.txt 2>&1
grep -qE "tree hashes present in the org: 3 of 3" $R/snapshot_census.txt && ! grep "^owned" $R/snapshot_census.txt | grep -qv "| active |" \
  || { log "CalibForge snapshots not all ACTIVE"; scancel $CORE $BURST; exit 1; }
log "pre-flight ok"

# ---- 2. endpoints -------------------------------------------------------------------------------------------------
for j in $CORE $BURST; do
  RS=""
  while [ ! -f $EP/$j.teacher ]; do
    [ -f $EP/$j.DEAD ] && { log "job $j died: $(cat $EP/$j.DEAD)"; scancel $CORE $BURST; exit 1; }
    ST=$(squeue -h -j $j -o %T 2>/dev/null); [ -z "$ST" ] && { log "job $j gone"; scancel $CORE $BURST; exit 1; }
    [ "$ST" = RUNNING ] && { RS=${RS:-$(date +%s)}; [ $(( $(date +%s) - RS )) -gt $UP_WAIT ] && { log "job $j: no endpoints"; scancel $CORE $BURST; exit 1; }; }
    sleep 30
  done
done
SURL=$(cat $EP/$CORE.student); CT=$(cat $EP/$CORE.teacher); BT=$(cat $EP/$BURST.teacher)
TFILE=$R/teachers.txt; echo "$CT,$BT" | tr ',' '\n' | grep . > $TFILE
JSTART=$(date -d "$(squeue -h -j $CORE -o %S)" +%s); CORE_LIMIT=$(squeue -h -j $CORE -o %l)
WALL_S=$(echo $CORE_LIMIT | awk -F: '{ if (NF==3) print $1*3600+$2*60+$3; else print $1*60+$2 }')
DEADLINE=$((JSTART + WALL_S - DEADLINE_MARGIN))
log "students=$SURL teachers=$(tr '\n' ' ' < $TFILE) deadline=$(date -d @$DEADLINE -Is)"

# ---- 3. routers ---------------------------------------------------------------------------------------------------
TALL=$(paste -sd, $TFILE)
$PY $ROUTER --mode teacher --arm control --port $PB --log-dir $R/router_control --tasks $TREE/router_tasks.json \
  --budget-mode on --deadline-epoch $DEADLINE --teacher-url $TALL --teacher-model qwen38 --teacher-url-file $TFILE \
  > $R/router_control.log 2>&1 & RB=$!
$PY $ROUTER --mode relay --arm relay_repair --student-think strip --repair-on-parse-error \
  --terminus-parser $HARBOR_SRC/harbor/agents/terminus_2/terminus_json_plain_parser.py --port $PR \
  --log-dir $R/router_relay_repair --tasks $TREE/router_tasks.json --budget-mode on --deadline-epoch $DEADLINE \
  --student-url $SURL --student-model snowball --teacher-url $TALL --teacher-model qwen38 --teacher-url-file $TFILE \
  > $R/router_relay_repair.log 2>&1 & RR=$!
for arm in control relay_repair; do
  for i in $(seq 1 120); do grep -q RELAY_ROUTER_READY $R/router_$arm.log 2>/dev/null && break
    [ -f $R/router_$arm/FATAL ] && abort "router $arm health: $(head -2 $R/router_$arm/FATAL)"; sleep 5; done
  grep -q RELAY_ROUTER_READY $R/router_$arm.log || abort "router $arm not ready"
done

# ---- 4. harbor ----------------------------------------------------------------------------------------------------
export PYTHONPATH=$HARBOR_SRC
render() {  # render <job suffix> <port> <conc> <task list> <attempts>
  local CFG=$R/${NAME}_$1.yaml
  sed "s#__JOB_NAME__#${NAME}_$1#; s#__JOBS_DIR__#$JOBS_ROOT#; s#__API_BASE__#http://127.0.0.1:$2/v1#; s#__CONC__#$3#" $HERE/relay_pilot.yaml > $CFG
  $PY - "$CFG" "$TREE" "$4" "$5" <<'PY' || abort "config $1 does not validate"
import sys, yaml
p, tree, lst, att = sys.argv[1:5]
c = yaml.safe_load(open(p)); c['n_attempts'] = int(att)
c['tasks'] = [{'path': f'{tree}/{l.strip()}'} for l in open(lst) if l.strip()]
yaml.safe_dump(c, open(p, 'w'), sort_keys=False)
from harbor_config.models.job.config import JobConfig
JobConfig.model_validate(c); print(p, len(c['tasks']), 'tasks x', att)
PY
  echo $CFG
}
CB=$(render control $PB $CONC_BASE $TREE/TASKS.txt 1 | tail -1); $HARBOR jobs start --config $CB > $R/harbor_control.log 2>&1 & HB=$!
CR=$(render relay_repair $PR $CONC_RELAY $TREE/TASKS.txt 1 | tail -1); $HARBOR jobs start --config $CR > $R/harbor_relay_repair.log 2>&1 & HR=$!
T0=$(date +%s); LAST_N=0; LAST_CHANGE=$T0; EARLY=0; LAT=0; P2=0; RELEASED_AT=""
log "phase 1 started: control pid $HB, relay_repair pid $HR"

# ---- 5. watch -----------------------------------------------------------------------------------------------------
while :; do
  sleep 60; NOW=$(date +%s)
  for arm in control relay_repair; do [ -f $R/router_$arm/FATAL ] && abort "router $arm FATAL: $(head -2 $R/router_$arm/FATAL)"; done
  for j in $CORE $BURST; do
    { [ $j = $CORE ] && [ $CORE_UP = 1 ]; } || { [ $j = $BURST ] && [ $BURST_UP = 1 ]; } || continue
    [ -f $EP/$j.DEAD ] && abort "job $j DEAD: $(cat $EP/$j.DEAD)"
    squeue -h -j $j -o %T | grep -q RUNNING || abort "job $j not RUNNING"
  done
  N=$(cat $R/router_*/turns.jsonl 2>/dev/null | wc -l); [ $N -ne $LAST_N ] && { LAST_N=$N; LAST_CHANGE=$NOW; }
  NH=$(node_h)
  log "watch: node-h=$NH requests=$N running control/relay/p2=$(running control)/$(running relay_repair)/$(running relay_repair_p2) core=$CORE_UP burst=$BURST_UP"
  awk -v a=$NH -v b=$CAP_NODE_H 'BEGIN{exit !(a>=b)}' && abort "node-hour cap $CAP_NODE_H reached"
  [ $NOW -lt $DEADLINE ] && [ $((NOW - LAST_CHANGE)) -gt $((STALL_MIN*60)) ] && abort "no router traffic for $STALL_MIN min"
  if [ $LAT = 0 ] && [ $((NOW - T0)) -ge $((LAT_MIN*60)) ]; then LAT=1
    P50=$($PY - $R <<'PY'
import json, sys
xs = []
for arm in ('control', 'relay_repair'):
    for l in open(f'{sys.argv[1]}/router_{arm}/turns.jsonl'):
        try: r = json.loads(l)
        except Exception: continue
        if r.get('owner') == 'teacher' and r.get('upstream_status') == 200: xs.append(r.get('latency_sec') or 0)
xs.sort(); print(xs[len(xs)//2] if xs else 0)
PY
)
    log "latency gate: teacher p50 ${P50}s after $LAT_MIN min"
    awk -v a=$P50 -v b=$LAT_ABORT 'BEGIN{exit !(a>b)}' && abort "teacher p50 ${P50}s > ${LAT_ABORT}s"
    awk -v a=$P50 -v b=$LAT_WARN 'BEGIN{exit !(a>b)}' && log "WARNING: teacher p50 ${P50}s > ${LAT_WARN}s"
  fi
  if [ $EARLY = 0 ] && [ $((NOW - T0)) -ge $((EARLY_MIN*60)) ]; then EARLY=1
    $PY $HERE/readout.py --run-dir $R --name $NAME --gate early > $R/early_gate.json 2> $R/early_gate.txt \
      || abort "early gate: $(grep FAIL $R/early_gate.txt | tr '\n' ' ')"
    log "early gate passed"
  fi
  # control drained -> phase-2 list; burst drained and released
  if [ $P2 = 0 ] && ! kill -0 $HB 2>/dev/null; then
    $PY - $JOBS_ROOT/${NAME}_control $R/p2_tasks.txt $HERE <<'PY'
import glob, os, sys
sys.path.insert(0, sys.argv[3]); import readout
solved = sorted({t for t, r, e in (readout.read_outcome(p) for p in glob.glob(f'{sys.argv[1]}/*/result.json'))
                 if r is not None and r >= 1 and (e is None or e in readout.AGENT_ENDS)})
open(sys.argv[2], 'w').write(''.join(t + '\n' for t in solved)); print(len(solved), 'control-solved tasks -> phase 2')
PY
    P2=1; log "control done; phase 2 has $(grep -c . $R/p2_tasks.txt) tasks x 2"
    if [ $BURST_UP = 1 ]; then
      echo "$CT" | tr ',' '\n' | grep . > $TFILE.new && mv $TFILE.new $TFILE; DSTART=$NOW
      log "burst teachers dropped from the list; draining"
    fi
  fi
  if [ $P2 -ge 1 ] && [ $BURST_UP = 1 ]; then
    INF=$($PY - $PB $PR "$BT" <<'PY'
import json, sys, urllib.request
n = 0
for port in sys.argv[1:3]:
    try: e = json.load(urllib.request.urlopen(f'http://127.0.0.1:{port}/endpoints', timeout=10))['teacher']
    except Exception: e = {}
    n += sum(v['inflight'] for u, v in e.items() if u in sys.argv[3].split(','))
print(n)
PY
)
    { [ "$INF" = 0 ] || [ $((NOW - DSTART)) -gt $DRAIN_WAIT ]; } && release $BURST "control done, burst drained (in flight $INF)"
  fi
  if [ $P2 = 1 ] && [ -s $R/p2_tasks.txt ] && [ "$(running relay_repair)" -le $P2_START ]; then
    C2=$(render relay_repair_p2 $PR $CONC_P2 $R/p2_tasks.txt 2 | tail -1); $HARBOR jobs start --config $C2 > $R/harbor_relay_repair_p2.log 2>&1 & HP2=$!
    P2=2; log "phase 2 started (pid $HP2)"
  fi
  ALIVE=0; for p in $HB $HR $HP2; do kill -0 $p 2>/dev/null && ALIVE=$((ALIVE+1)); done
  if [ $ALIVE -eq 0 ] && { [ $P2 = 2 ] || [ ! -s $R/p2_tasks.txt -a $P2 = 1 ]; }; then break; fi
  if [ $NOW -ge $DEADLINE ] && [ $CORE_UP = 1 ] && [ $((NOW - LAST_CHANGE)) -ge 180 ]; then release $CORE "past the deadline, traffic stopped"; RELEASED_AT=$NOW; fi
  [ -n "$RELEASED_AT" ] && [ $((NOW - RELEASED_AT)) -gt $VERIFY_WAIT ] && { log "harbor still running after release; stopping"; stop_harbor; break; }
done

# ---- 6. finish ----------------------------------------------------------------------------------------------------
kill -TERM $RB $RR 2>/dev/null
[ $CORE_UP = 1 ] && release $CORE "done"; [ $BURST_UP = 1 ] && release $BURST "done"
sleep 20; write_meta; cleanup_sandboxes
$PY $HERE/readout.py --run-dir $R --name $NAME --gate final --json $R/readout.json > /dev/null 2> $R/readout.txt; cat $R/readout.txt
$PY $HERE/select_kept.py --run-dir $R --name $NAME --target 2000 --out $R/kept_manifest.jsonl | tail -20
log "RUN_DONE $(tr '\n' ' ' < $R/run.meta)"
