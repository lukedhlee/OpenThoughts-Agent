#!/bin/bash
# run_pilot.sh <serve-jobid> <run-name> [smoke]
# Jupiter login node, inside tmux. Drives one relay pilot on the two servers serve_relay.sbatch started.
#
# Arms (ARMS, default "control relay_repair"), each its own router and harbor job on the same 100 CalibForge tasks:
#   control       teacher from scratch (router --mode teacher)                                 = the baseline
#   relay_repair  student -> teacher: parse_error repairs (non-sticky, one teacher turn) plus the sticky takeovers
#                 done_claim / loop / no_progress_wait; the student's thinking stripped from the teacher's view
#   relay         sticky takeovers only, thinking stripped;  relay_keep  the same, thinking kept (--student-think keep)
#
#   1. pre-flight (no GPU time spent yet): harbor clone at the pinned commit, task tree + router task file, Daytona key,
#      the three CalibForge snapshots ACTIVE (a missing one would make harbor build a new snapshot into a 39/40 org),
#      none of the pilot tasks in the held-out eval split
#   2. waits for the endpoint files (queue time is free; once RUNNING the servers get UP_WAIT s, else scancel)
#   3. starts one router per arm, each health-checking the served names and one real completion (exit 3 -> cancel);
#      every router gets --deadline-epoch = job start + CAP_NODE_H/2 h - DEADLINE_MARGIN, after which each episode is
#      ended at its next request, so the run finishes inside the cap with its results written
#   4. renders relay_pilot.yaml per arm (longest agent budget first) and runs the harbor jobs on Daytona
#   5. every 60 s: router FATAL -> abort; serve job DEAD/gone before the deadline -> abort; no traffic for STALL_MIN
#      before the deadline -> abort; at EARLY_MIN the early gate (readout.py --gate early) or abort. Past the deadline,
#      once no LLM request has arrived for 3 min, the serve job is cancelled (no more node-hours) and harbor finishes
#      its verifiers on the login node; the node-hour cap also cancels it (Slurm's --time is the backstop)
#   6. harbor done -> stop routers, cancel the serve job if still up, remove leftover sandboxes of these jobs (by id),
#      write run.meta, run the final readout
# "smoke": 4 tasks per arm (shortest budgets), 4 concurrent, early gate at 10 min.
#
#   tmux new -d -s relay_<name> "bash run_pilot.sh <jobid> <name>"
set -uo pipefail
JOB=${1:?serve job id}; NAME=${2:?run name}; MODE=${3:-full}
C=/e/project1/transfernetx/lee27/code
HERE=$(cd "$(dirname "$0")" && pwd)
ROUTER=$HERE/../router/relay_router.py
PY=${PY:-$C/envs/snowball-v2/bin/python}; HARBOR=${HARBOR:-$C/envs/snowball-v2/bin/harbor}
HARBOR_SRC=${HARBOR_SRC:-$C/harbor-terminus2-relay/src}
HARBOR_SHA=${HARBOR_SHA:-89098635}          # marin-community/harbor lukedhlee/terminus2-relay
TREE=${TREE:-/e/fscratch/reformo/lee27/tasks/calibforge_relay100}
NTASKS=${NTASKS:-100}
RUN_KIND=${RUN_KIND:-pilot}
STUDENT_TOKENIZER=${STUDENT_TOKENIZER:-/e/data1/mmlaion/lee27/models/grug-datakit-sft-20260921/tokenizer.json}   # the cap on older teacher reasoning   # pilot: TASKS.txt must be disjoint from the held-out split; heldout: it must BE the split
KEYF=${KEYF:-/e/fscratch/reformo/lee27/keys/daytona_eval.env}
E=/e/fscratch/reformo/lee27/experiments/relay/pilot; EP=$E/endpoints
R=$E/runs/$NAME
JOBS_ROOT=${JOBS_ROOT:-/e/data1/mmlaion/lee27/experiments/relay_pilot_jobs}   # many small files -> mmlaion
ARMS=${ARMS:-control relay_repair}
CONC=${CONC:-100}; CAP_NODE_H=${CAP_NODE_H:-4.0}; NODES=${NODES:-3}; DEADLINE_MARGIN=${DEADLINE_MARGIN:-300}
EARLY_MIN=${EARLY_MIN:-25}; STALL_MIN=${STALL_MIN:-15}; UP_WAIT=${UP_WAIT:-2400}; MIN_EARLY_TURNS=${MIN_EARLY_TURNS:-20}
VERIFY_WAIT=${VERIFY_WAIT:-2700}   # after the serve job is released, how long harbor may keep verifying (CPU only)
PORT0=${PORT0:-$((21000 + RANDOM % 8000))}
[ "$MODE" = smoke ] && { CONC=4; EARLY_MIN=10; MIN_EARLY_TURNS=4; }
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1   # login-node pid cap (4,096 incl. threads)
[ -d $R ] && { echo "$R exists; pick a new run name"; exit 1; }
mkdir -p $R $JOBS_ROOT
ln -s $JOBS_ROOT $R/jobs
exec > >(tee -a $R/driver.log) 2>&1
log() { echo "[$(date -Is)] $*"; }
declare -A HPID RPID PORT
i=0; for arm in $ARMS; do PORT[$arm]=$((PORT0 + i)); i=$((i+1)); done
SERVE_RELEASED=0
release_serve() { [ $SERVE_RELEASED = 1 ] && return; scancel $JOB; SERVE_RELEASED=1; log "serve job $JOB cancelled: $*"; }
stop_harbor() { for arm in $ARMS; do [ -n "${HPID[$arm]:-}" ] && kill -INT ${HPID[$arm]} 2>/dev/null; done; sleep 30
                for arm in $ARMS; do [ -n "${HPID[$arm]:-}" ] && kill -TERM ${HPID[$arm]} 2>/dev/null; done; }
stop_routers() { for arm in $ARMS; do [ -n "${RPID[$arm]:-}" ] && kill -TERM ${RPID[$arm]} 2>/dev/null; done; }
write_meta() {
  local el; el=$(sacct -j $JOB -X -n -o ElapsedRaw 2>/dev/null | head -1 | tr -d ' ')
  printf 'serve_job=%s\nnode_hours=%s\nharbor=%s\nota=%s\narms=%s\nconc=%s\nmode=%s\ndeadline=%s\nend=%s\n' "$JOB" \
    "$(awk -v s="${el:-0}" -v n=$NODES 'BEGIN{printf "%.3f", n*s/3600}')" "$(git -C ${HARBOR_SRC%/src} rev-parse --short=8 HEAD)" \
    "$(git -C $HERE rev-parse --short=8 HEAD 2>/dev/null)" "$ARMS" "$CONC" "$MODE" "${DEADLINE:-}" "$(date -Is)" > $R/run.meta
}
cleanup_sandboxes() {
  local jobs=(); for arm in $ARMS; do [ -d $JOBS_ROOT/${NAME}_$arm ] && jobs+=($JOBS_ROOT/${NAME}_$arm); done
  [ ${#jobs[@]} -gt 0 ] && $PY $HERE/cleanup_sandboxes.py --key-file $KEYF --delete "${jobs[@]}" 2>&1 | tail -3
}
abort() { log "ABORT: $*"; echo "$*" > $R/ABORT; stop_harbor; stop_routers; release_serve "abort"; cleanup_sandboxes; write_meta; exit 1; }
log "run_pilot $NAME job=$JOB mode=$MODE arms=[$ARMS] conc=$CONC cap=${CAP_NODE_H} node-h"

# ---- 1. pre-flight --------------------------------------------------------------------------------------------------
[ "$(git -C ${HARBOR_SRC%/src} rev-parse --short=8 HEAD)" = "$HARBOR_SHA" ] || { log "harbor at ${HARBOR_SRC%/src} is not $HARBOR_SHA"; scancel $JOB; exit 1; }
[ -f $TREE/router_tasks.json ] && [ "$(find $TREE -mindepth 1 -maxdepth 1 -type d | wc -l)" = $NTASKS ] || { log "tree $TREE does not hold $NTASKS staged tasks (stage_tree.sh)"; scancel $JOB; exit 1; }
if [ $RUN_KIND = heldout ]; then
  diff -q <(sort $TREE/TASKS.txt) <(sort $HERE/calibforge_heldout300.txt) >/dev/null || { log "tree is not the held-out split"; scancel $JOB; exit 1; }
else
  [ -z "$(comm -12 <(sort $TREE/TASKS.txt) <(sort $HERE/calibforge_heldout300.txt))" ] || { log "pilot tasks overlap the held-out split"; scancel $JOB; exit 1; }
fi
$PY -c "import aiohttp, yaml" || { log "aiohttp/yaml missing in $PY"; scancel $JOB; exit 1; }
set -a; source $KEYF; set +a
curl -sf --max-time 20 -H "Authorization: Bearer $DAYTONA_API_KEY" https://app.daytona.io/api/api-keys/current >/dev/null || { log "Daytona key rejected"; scancel $JOB; exit 1; }
PYTHONPATH=$HARBOR_SRC $PY $HERE/../../calibforge/daytona/snapshot_census.py --key-file $KEYF --tree $TREE > $R/snapshot_census.txt 2>&1
$PY - $R/snapshot_census.txt <<'PY' || { log "CalibForge snapshots not all ACTIVE (see snapshot_census.txt; run the RECOVERY.md restore first)"; scancel $JOB; exit 1; }
import re, sys
txt = open(sys.argv[1]).read()
owned = [l.split(' | ') for l in txt.splitlines() if l.startswith('owned ')]
m = re.search(r'tree hashes present in the org: (\d+) of (\d+)', txt)
bad = [o[1] for o in owned if o[2].lower() != 'active']
print('owned snapshots:', [(o[1], o[2]) for o in owned], 'present:', m.groups() if m else None)
sys.exit(0 if m and m.group(1) == m.group(2) and owned and not bad else 1)
PY
log "pre-flight ok: harbor $HARBOR_SHA, tree $TREE, held-out disjoint, snapshots active"

# ---- 2. endpoints ---------------------------------------------------------------------------------------------------
RUNNING_SINCE=""
while [ ! -f $EP/$JOB.student ] || [ ! -f $EP/$JOB.teacher ]; do   # .teacher is empty on a student-only serve
  [ -f $EP/$JOB.DEAD ] && { log "serve job died: $(cat $EP/$JOB.DEAD)"; exit 1; }
  ST=$(squeue -h -j $JOB -o %T 2>/dev/null)
  [ -z "$ST" ] && { log "serve job $JOB gone before its endpoints appeared"; exit 1; }
  if [ "$ST" = RUNNING ]; then
    RUNNING_SINCE=${RUNNING_SINCE:-$(date +%s)}
    [ $(( $(date +%s) - RUNNING_SINCE )) -gt $UP_WAIT ] && { log "no endpoints $UP_WAIT s after start"; scancel $JOB; exit 1; }
  fi
  sleep 30
done
SURL=$(cat $EP/$JOB.student); TURL=$(cat $EP/$JOB.teacher)
JSTART=$(date -d "$(squeue -h -j $JOB -o %S)" +%s)
DEADLINE=$(awk -v s=$JSTART -v c=$CAP_NODE_H -v n=$NODES -v m=$DEADLINE_MARGIN 'BEGIN{printf "%d", s + c/n*3600 - m}')
log "endpoints student=$SURL teacher=$TURL; job start $(date -d @$JSTART -Is), deadline $(date -d @$DEADLINE -Is)"

# ---- 3. routers -----------------------------------------------------------------------------------------------------
for arm in $ARMS; do
  TARGS=(); [ -n "$TURL" ] && TARGS=(--teacher-url $TURL --teacher-model qwen38 --teacher-max-tokens ${TEACHER_MAX_TOKENS:-16384})
  case $arm in control) M=(--mode teacher);; student_only) M=(--mode student);; relay) M=(--mode relay --student-think strip);; relay_keep) M=(--mode relay --student-think keep);;
    relay_repair) M=(--mode relay --student-think strip --repair-on-parse-error --terminus-parser $HARBOR_SRC/harbor/agents/terminus_2/terminus_json_plain_parser.py
                  --autofix --student-tokenizer $STUDENT_TOKENIZER);;
    *) abort "unknown arm $arm";; esac
  $PY $ROUTER "${M[@]}" --arm $arm --port ${PORT[$arm]} --log-dir $R/router_$arm --tasks $TREE/router_tasks.json \
    --budget-mode on --deadline-epoch $DEADLINE --student-url $SURL --student-model snowball "${TARGS[@]}" \
    > $R/router_$arm.log 2>&1 &
  RPID[$arm]=$!
done
for arm in $ARMS; do
  for i in $(seq 1 90); do
    grep -q RELAY_ROUTER_READY $R/router_$arm.log 2>/dev/null && break
    [ -f $R/router_$arm/FATAL ] && abort "router $arm failed its health check: $(head -3 $R/router_$arm/FATAL)"
    sleep 5
  done
  grep -q RELAY_ROUTER_READY $R/router_$arm.log || abort "router $arm not ready"
done
log "routers ready: $(for arm in $ARMS; do printf '%s:%s ' $arm ${PORT[$arm]}; done)"

# ---- 4. harbor ------------------------------------------------------------------------------------------------------
export PYTHONPATH=$HARBOR_SRC
for arm in $ARMS; do
  CFG=$R/${NAME}_$arm.yaml
  sed "s#__JOB_NAME__#${NAME}_$arm#; s#__JOBS_DIR__#$JOBS_ROOT#; s#__API_BASE__#http://127.0.0.1:${PORT[$arm]}/v1#; s#__CONC__#$CONC#" $HERE/relay_pilot.yaml > $CFG
  $PY - "$CFG" "$TREE" "$MODE" <<'PY' || abort "config for $arm does not validate"
import os, re, sys, yaml
p, tree, mode = sys.argv[1:4]
c = yaml.safe_load(open(p))
ids = [l.strip() for l in open(f'{tree}/TASKS.txt') if l.strip()]
def budget(t):
    m = re.search(r'\[agent\][^\[]*?timeout_sec\s*=\s*([0-9.]+)', open(f'{tree}/{t}/task.toml').read())
    return float(m.group(1)) if m else 0.0
ids.sort(key=lambda t: (-budget(t), t))           # longest budget first
if mode == 'smoke':
    ids = sorted(ids, key=lambda t: (budget(t), t))[:4]
c['tasks'] = [{'path': f'{tree}/{t}'} for t in ids]
yaml.safe_dump(c, open(p, 'w'), sort_keys=False)
from harbor_config.models.job.config import JobConfig
JobConfig.model_validate(c)
print(f'{os.path.basename(p)}: {len(ids)} tasks, validates')
PY
  [ -e $JOBS_ROOT/${NAME}_$arm ] && abort "$JOBS_ROOT/${NAME}_$arm exists"
  $HARBOR jobs start --config $CFG > $R/harbor_$arm.log 2>&1 &
  HPID[$arm]=$!
  log "harbor $arm started (pid ${HPID[$arm]}) -> $JOBS_ROOT/${NAME}_$arm"
done
T0=$(date +%s); EARLY_DONE=0; LAST_N=0; LAST_CHANGE=$T0; RELEASED_AT=""

# ---- 5. watch -------------------------------------------------------------------------------------------------------
while :; do
  sleep 60
  NOW=$(date +%s)
  for arm in $ARMS; do [ -f $R/router_$arm/FATAL ] && abort "router $arm FATAL: $(head -3 $R/router_$arm/FATAL)"; done
  N=0; for arm in $ARMS; do N=$((N + $(wc -l < $R/router_$arm/turns.jsonl 2>/dev/null || echo 0))); done
  [ $N -ne $LAST_N ] && { LAST_N=$N; LAST_CHANGE=$NOW; }
  ALIVE=0; for arm in $ARMS; do kill -0 ${HPID[$arm]} 2>/dev/null && ALIVE=$((ALIVE+1)); done
  NH=$(awk -v s=$JSTART -v n=$NOW -v k=$NODES 'BEGIN{printf "%.2f", k*(n-s)/3600}')
  [ $SERVE_RELEASED = 1 ] && NH="$NH (released)"
  log "watch: node-h=$NH requests=$N harbor_alive=$ALIVE threads=$(ps -L -u $USER --no-headers 2>/dev/null | wc -l)"
  [ $ALIVE -eq 0 ] && break
  if [ $SERVE_RELEASED = 0 ]; then
    [ -f $EP/$JOB.DEAD ] && abort "serve job reports DEAD: $(cat $EP/$JOB.DEAD)"
    squeue -h -j $JOB -o %T | grep -q RUNNING || abort "serve job $JOB not RUNNING"
    awk -v a="${NH%% *}" -v b=$CAP_NODE_H 'BEGIN{exit !(a>=b)}' && { release_serve "node-hour cap $CAP_NODE_H reached"; RELEASED_AT=$NOW; }
    if [ $NOW -ge $DEADLINE ] && [ $((NOW - LAST_CHANGE)) -ge 180 ]; then release_serve "past the deadline, no LLM traffic for 3 min"; RELEASED_AT=$NOW; fi
    [ $NOW -lt $DEADLINE ] && [ $((NOW - LAST_CHANGE)) -gt $((STALL_MIN*60)) ] && abort "no router traffic for $STALL_MIN min"
  elif [ $((NOW - RELEASED_AT)) -gt $VERIFY_WAIT ]; then
    log "harbor still running $VERIFY_WAIT s after the servers were released; stopping it"; stop_harbor; break
  fi
  if [ $EARLY_DONE = 0 ] && [ $((NOW - T0)) -ge $((EARLY_MIN*60)) ]; then
    EARLY_DONE=1
    $PY $HERE/readout.py --run-dir $R --name $NAME --gate early > $R/early_gate.json 2> $R/early_gate.txt \
      || abort "early gate failed: $(grep FAIL $R/early_gate.txt | tr '\n' ' ')"
    for arm in $ARMS; do
      t=$($PY -c "import json; print(json.load(open('$R/early_gate.json'))['$arm']['main_turns'])")
      [ "$t" -ge $MIN_EARLY_TURNS ] || abort "early gate: only $t turns on $arm after $EARLY_MIN min"
    done
    log "early gate passed: $(grep -c PASS $R/early_gate.txt) checks"
  fi
done

# ---- 6. finish ------------------------------------------------------------------------------------------------------
stop_routers
release_serve "harbor done"
sleep 20; write_meta
cleanup_sandboxes
$PY $HERE/readout.py --run-dir $R --name $NAME --gate final --json $R/readout.json > /dev/null 2> $R/readout.txt
cat $R/readout.txt
log "RUN_DONE $(tr '\n' ' ' < $R/run.meta)"
