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
# Qwen's replies (every arm with a teacher): TEACHER_GUARD=1 (default) checks each agent turn with Terminus-2's parser
# (autofix, else up to TEACHER_RESAMPLES more samples, else passed through); VERIFY_NOTE=1 adds the verification note
# to Qwen's confirmation request (relay_router.py --teacher-format-guard / --verify-note).
# "smoke": 4 tasks per arm (shortest budgets), 4 concurrent, early gate at 10 min.
#
#   tmux new -d -s relay_<name> "bash run_pilot.sh <jobid> <name>"
#
# Other sites (TACC Horizon, 2026-09-29): every path is an env knob (PY HARBOR HARBOR_SRC TREE KEYF E JOBS_ROOT
# STUDENT_TOKENIZER TASK_LIST; E also from the serve side's RELAY_EXP_DIR; STUDENT_TOKENIZER=meta = the tokenizer.json of
# the student the serve job loaded, from its endpoints .meta), and the driver may run in its own compute-node job (data/relay/horizon/driver.sbatch):
# DRIVER_JOB / DRIVER_NODES make the node-hour cap, the deadline and run.meta count that job too (unset = Jupiter's
# accounting, unchanged). With HTTPS_PROXY set, the model hosts are added to NO_PROXY once the endpoints are known, and
# CLEANUP_PY names a sandbox cleanup that honors the proxy (the sync SDK's cleanup_sandboxes.py connects straight out).
# AGENT=msa (2026-10-02): mini-swe-agent tool mode instead of Terminus-2: harbor lukedhlee/mini-swe-relay (HARBOR_SRC /
# HARBOR_SHA defaults change with it; mini-swe-agent 2.4.6 from MSA_DIR on PYTHONPATH), the job template relay_msa.yaml,
# and every router with --harness msa (its parse checks are msa_tool.py's, so no Terminus-2 parser is passed).
# STUDENT_VIEW_STRIP=1 (msa): after a sticky takeover, 09-21's view (hard-end count, row) drops the student's thinking.
# ARM_GATES=1 runs data/relay/horizon/arm_gates.py every ARM_GATES_EVERY s (the finetuned-student gates): the takeover
# rate over finished relay episodes against [TAKEOVER_MIN, TAKEOVER_MAX] once TAKEOVER_AFTER are in (TAKEOVER_ACTION
# flag|stop), and the student's EAGLE-3 mean acceptance length from its servers' /metrics (flag below ACCEPT_FLAG, stop
# below ACCEPT_STOP, 0 = off). A flag is logged and appended to <run>/FLAGS; it never cancels anything.
set -uo pipefail
JOB=${1:?serve job id}; NAME=${2:?run name}; MODE=${3:-full}
C=/e/project1/transfernetx/lee27/code
HERE=$(cd "$(dirname "$0")" && pwd)
ROUTER=$HERE/../router/relay_router.py
PY=${PY:-$C/envs/snowball-v2/bin/python}; HARBOR=${HARBOR:-$C/envs/snowball-v2/bin/harbor}
AGENT=${AGENT:-terminus2}           # terminus2 | msa
if [ "$AGENT" = msa ]; then
  HARBOR_SRC=${HARBOR_SRC:-$C/harbor-mini-swe-relay/src}
  HARBOR_SHA=${HARBOR_SHA:?HARBOR_SHA = the lukedhlee/mini-swe-relay commit at HARBOR_SRC}   # marin-community/harbor
  MSA_DIR=${MSA_DIR:-$C/envs/msa-2.4.6}; JOB_TEMPLATE=relay_msa.yaml
else
  HARBOR_SRC=${HARBOR_SRC:-$C/harbor-terminus2-relay/src}
  HARBOR_SHA=${HARBOR_SHA:-89098635}          # marin-community/harbor lukedhlee/terminus2-relay
  MSA_DIR=; JOB_TEMPLATE=relay_pilot.yaml
fi
TREE=${TREE:-/e/fscratch/reformo/lee27/tasks/calibforge_relay100}
NTASKS=${NTASKS:-100}
TASK_LIST=${TASK_LIST:-}          # a subset of the tree to run (default: the tree's TASKS.txt), e.g. the solvable tasks; a task
                                  # listed k times runs k times (round by round), e.g. a top-up of unscored rollout slots
N_ATTEMPTS=${N_ATTEMPTS:-1}       # rollouts per task
MAX_INPUT=${MAX_INPUT:-65536}     # harbor's max_input_tokens; 131072 when Qwen serves 128k (the router reports it)
MAX_OUTPUT=${MAX_OUTPUT:-8192}    # harbor's max_output_tokens; 16384 under the 65k/16k eval policy (the SFT arms and their 09-21 reference)
MSA_INTERLEAVED=${MSA_INTERLEAVED:-true}  # msa: harbor re-sends earlier reasoning (true) or not (false, the no-refeed eval policy; student_only self-distillation)
TEACHER_MAX_TOKENS=${TEACHER_MAX_TOKENS:-32768}
CLOCK=${CLOCK:-paused}            # paused: the router's budgets, clock paused on model calls; wall: harbor's own 1x agent timeout;
                                  # repair: the router's budgets, the student's clock paused only while a teacher repair is in flight
CTX_BUDGET=${CTX_BUDGET:-}        # relay arms: context_budget takeover at this student-view size (e.g. 32000); empty = off
ROW_MAX=${ROW_MAX:-}; ROW_RESERVE=${ROW_RESERVE:-8192}   # relay arms: hard end once 09-21's view > ROW_MAX - ROW_RESERVE after a takeover
BALANCE=${BALANCE:-pinned}; STAGGER_SEC=${STAGGER_SEC:-0}   # STAGGER_SEC>0: the task list is split in two harbor jobs started that far apart
GATE_MIN=${GATE_MIN:-0}; GATE_LAT=${GATE_LAT:-30}; GATE_KV=${GATE_KV:-0.90}   # GATE_MIN>0: from then on, cancel on latency / KV saturation
TEACHER_GUARD=${TEACHER_GUARD:-1}  # 1: every Qwen agent turn is checked with Terminus-2's parser (autofix, else resample, else pass), every arm
TEACHER_RESAMPLES=${TEACHER_RESAMPLES:-2}
SHUFFLE_SEED=${SHUFFLE_SEED:-}     # set: tasks run in a uniform shuffled order (this seed), so a run cut early is an unbiased sample;
                                  # 'list': the task list's own order; empty: longest agent budget first (the default so far)
FAILOVER_5XX=${FAILOVER_5XX:-1}    # 1: an upstream 5xx (a crashed engine) fails over to another server like a connection error
VERIFY_NOTE=${VERIFY_NOTE:-0}      # 1: the verification note on Qwen's confirmation request (relay: done_claim takeover; control: the first)
[ $CLOCK = wall ] && AGENT_MULT=1.0 || AGENT_MULT=8.0
OVF_AFTER=${OVF_AFTER:-0}; OVF_MAX=${OVF_MAX:-0.20}   # relay backstop: cancel when overflow > OVF_MAX after OVF_AFTER episodes (0 = off)
STOP_AFTER=${STOP_AFTER:-0}; STOP_OVF=${STOP_OVF:-0.30}; STOP_FMT=${STOP_FMT:-0.98}; STOP_HERR=${STOP_HERR:-0.10}; STOP_EVERY=${STOP_EVERY:-900}
                                  # full run's stop rule (stop_rule.py): from STOP_AFTER scored relay episodes on, every STOP_EVERY s,
                                  # cancel on overflow > STOP_OVF, valid format < STOP_FMT or harness errors > STOP_HERR (0 = off)
HERR_EXCLUDE=${HERR_EXCLUDE:-}     # stop_rule.py --harness-exclude-tasks: judge harness errors only on tasks not in this list
TARGET_FAIL=${TARGET_FAIL:-0}; TARGET_BASE=${TARGET_BASE:-0}   # TARGET_FAIL>0: at each stop-rule check (from STOP_AFTER's first
                                  # call on), end the run normally (harbor stopped, servers released, final readout) once
                                  # TARGET_BASE (earlier attempts' real failures) + this run's real failures >= TARGET_FAIL
RUN_KIND=${RUN_KIND:-pilot}
STUDENT_TOKENIZER=${STUDENT_TOKENIZER:-/e/data1/mmlaion/lee27/models/grug-datakit-sft-20260921/tokenizer.json}   # the cap on older teacher reasoning   # pilot: TASKS.txt must be disjoint from the held-out split; heldout: it must BE the split
KEYF=${KEYF:-/e/fscratch/reformo/lee27/keys/daytona_eval.env}
E=${E:-${RELAY_EXP_DIR:-/e/fscratch/reformo/lee27/experiments/relay/pilot}}; EP=$E/endpoints
R=$E/runs/$NAME
JOBS_ROOT=${JOBS_ROOT:-/e/data1/mmlaion/lee27/experiments/relay_pilot_jobs}   # many small files -> mmlaion
ARMS=${ARMS:-control relay_repair}
CONC=${CONC:-100}; CAP_NODE_H=${CAP_NODE_H:-4.0}; NODES=${NODES:-3}; DEADLINE_MARGIN=${DEADLINE_MARGIN:-300}
EARLY_MIN=${EARLY_MIN:-25}; STALL_MIN=${STALL_MIN:-15}; UP_WAIT=${UP_WAIT:-2400}; MIN_EARLY_TURNS=${MIN_EARLY_TURNS:-20}
DRIVER_JOB=${DRIVER_JOB:-}; DRIVER_NODES=${DRIVER_NODES:-0}   # the driver's own Slurm job (Horizon), counted in the cap
ARM_GATES=${ARM_GATES:-0}; ARM_GATES_EVERY=${ARM_GATES_EVERY:-600}; ARM_GATES_FROM=${ARM_GATES_FROM:-10}   # minutes after harbor start
TAKEOVER_MIN=${TAKEOVER_MIN:-}; TAKEOVER_MAX=${TAKEOVER_MAX:-}; TAKEOVER_AFTER=${TAKEOVER_AFTER:-100}; TAKEOVER_ACTION=${TAKEOVER_ACTION:-flag}
ACCEPT_FLAG=${ACCEPT_FLAG:-0}; ACCEPT_STOP=${ACCEPT_STOP:-0}
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
# DRIVER_IN_SERVE=1: this driver is a step inside the serve job, so cancelling it would end the driver before the sandbox
# cleanup and the readout; the serve job is cancelled after RUN_DONE instead.
release_serve() { [ $SERVE_RELEASED = 1 ] && return; SERVE_RELEASED=1
                  if [ "${DRIVER_IN_SERVE:-0}" = 1 ]; then log "serve job $JOB kept until the readout (driver inside it): $*"; return; fi
                  scancel $JOB; log "serve job $JOB cancelled: $*"; }
stop_harbor() { for k in "${!HPID[@]}"; do kill -INT ${HPID[$k]} 2>/dev/null; done; pkill -INT -u $USER -f "harbor jobs start --config $R/" 2>/dev/null; sleep 30
                for k in "${!HPID[@]}"; do kill -TERM ${HPID[$k]} 2>/dev/null; done; pkill -TERM -u $USER -f "harbor jobs start --config $R/" 2>/dev/null; }
stop_routers() { for arm in $ARMS; do [ -n "${RPID[$arm]:-}" ] && kill -TERM ${RPID[$arm]} 2>/dev/null; done; }
write_meta() {
  local el del=0; el=$(sacct -j $JOB -X -n -o ElapsedRaw 2>/dev/null | head -1 | tr -d ' ')
  [ -n "$DRIVER_JOB" ] && del=$(sacct -j $DRIVER_JOB -X -n -o ElapsedRaw 2>/dev/null | head -1 | tr -d ' ')
  printf 'serve_job=%s\nnodes=%s\nnode_hours=%s\nharbor=%s\nota=%s\narms=%s\nconc=%s\nmode=%s\ndeadline=%s\nend=%s\n' "$JOB" "$NODES" \
    "$(awk -v s="${el:-0}" -v n=$NODES -v d="${del:-0}" -v k=$DRIVER_NODES 'BEGIN{printf "%.3f", (n*s + k*d)/3600}')" "$(git -C ${HARBOR_SRC%/src} rev-parse --short=8 HEAD)" \
    "$(git -C $HERE rev-parse --short=8 HEAD 2>/dev/null)" "$ARMS" "$CONC" "$MODE" "${DEADLINE:-}" "$(date -Is)" > $R/run.meta
  [ -n "$DRIVER_JOB" ] && printf 'serve_node_hours=%s\ndriver_job=%s\ndriver_nodes=%s\ndriver_node_hours=%s\n' \
    "$(awk -v s="${el:-0}" -v n=$NODES 'BEGIN{printf "%.3f", n*s/3600}')" "$DRIVER_JOB" "$DRIVER_NODES" \
    "$(awk -v d="${del:-0}" -v k=$DRIVER_NODES 'BEGIN{printf "%.3f", k*d/3600}')" >> $R/run.meta
}
cleanup_sandboxes() {
  local jobs=(); for arm in $ARMS; do for d in $JOBS_ROOT/${NAME}_$arm $JOBS_ROOT/${NAME}_${arm}_p2; do [ -d $d ] && jobs+=($d); done; done
  [ ${#jobs[@]} -gt 0 ] && $PY ${CLEANUP_PY:-$HERE/cleanup_sandboxes.py} --key-file $KEYF --delete "${jobs[@]}" 2>&1 | tail -3
}
abort() { log "ABORT: $*"; echo "$*" > $R/ABORT; stop_harbor; stop_routers; release_serve "abort"; cleanup_sandboxes; write_meta; exit 1; }
log "run_pilot $NAME agent=$AGENT job=$JOB mode=$MODE arms=[$ARMS] conc=$CONC cap=${CAP_NODE_H} node-h clock=$CLOCK ctx_budget=${CTX_BUDGET:-off} row_max=${ROW_MAX:-off} teacher_guard=$TEACHER_GUARD verify_note=$VERIFY_NOTE failover_5xx=$FAILOVER_5XX max_input=$MAX_INPUT max_output=$MAX_OUTPUT teacher_max_tokens=$TEACHER_MAX_TOKENS balance=$BALANCE stagger=$STAGGER_SEC gate=$GATE_MIN/$GATE_LAT/$GATE_KV tasks=${TASK_LIST:-$TREE/TASKS.txt} x$N_ATTEMPTS stop=${STOP_AFTER}:$STOP_OVF/$STOP_FMT/$STOP_HERR herr_exclude=${HERR_EXCLUDE:-none} target_fail=$TARGET_FAIL base=$TARGET_BASE shuffle=${SHUFFLE_SEED:-off}"

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
if [ "$STUDENT_TOKENIZER" = meta ]; then   # the served student's own tokenizer (serve_relay.sbatch writes student_model=)
  STUDENT_TOKENIZER=$(sed -n 's/^student_model=//p' $EP/$JOB.meta 2>/dev/null | head -1)/tokenizer.json
  [ -f "$STUDENT_TOKENIZER" ] || abort "STUDENT_TOKENIZER=meta: no tokenizer at '$STUDENT_TOKENIZER' (endpoints .meta)"
  log "student tokenizer from the serve job's meta: $STUDENT_TOKENIZER"
fi
JSTART=$(date -d "$(squeue -h -j $JOB -o %S)" +%s)
# the driver job's node-hours (0 without DRIVER_JOB): what it ran before the serve job started (DPRE), and the verify tail
# it keeps running after the servers are released (DTAIL = VERIFY_WAIT) are reserved, the rest is shared per hour
DSTART=$JSTART; [ -n "$DRIVER_JOB" ] && DSTART=$(date -d "$(squeue -h -j $DRIVER_JOB -o %S)" +%s)
DPRE=$(awk -v j=$JSTART -v d=$DSTART -v k=$DRIVER_NODES 'BEGIN{x=j-d; printf "%.4f", k*(x>0?x:0)/3600}')
DTAIL=$(awk -v w=$VERIFY_WAIT -v k=$DRIVER_NODES 'BEGIN{printf "%.4f", k*w/3600}')
DEADLINE=$(awk -v s=$JSTART -v c=$CAP_NODE_H -v n=$NODES -v m=$DEADLINE_MARGIN -v k=$DRIVER_NODES -v p=$DPRE -v t=$DTAIL \
  'BEGIN{printf "%d", s + (c-p-t)/(n+k)*3600 - m}')
log "endpoints student=$SURL teacher=$TURL; job start $(date -d @$JSTART -Is), deadline $(date -d @$DEADLINE -Is)$([ -n "$DRIVER_JOB" ] && echo "; driver job $DRIVER_JOB x$DRIVER_NODES from $(date -d @$DSTART -Is), reserved ${DPRE} + ${DTAIL} node-h")"
if [ -n "${HTTPS_PROXY:-}${https_proxy:-}" ]; then   # model traffic never goes through the Daytona proxy
  MH=$(echo "$SURL,$TURL" | tr ',' '\n' | sed -E 's#^[a-z]+://([^:/]+).*#\1#' | grep -v '^$' | sort -u | paste -sd, -)
  export NO_PROXY="${NO_PROXY:-localhost,127.0.0.1}${MH:+,$MH}"; export no_proxy="$NO_PROXY"; log "NO_PROXY=$NO_PROXY"
fi

# ---- 3. routers -----------------------------------------------------------------------------------------------------
PARSER=$HARBOR_SRC/harbor/agents/terminus_2/terminus_json_plain_parser.py
PARSER_ARGS=(--terminus-parser $PARSER); HARNESS_ARGS=()
[ "$AGENT" = msa ] && { PARSER_ARGS=(); HARNESS_ARGS=(--harness msa); }
for arm in $ARMS; do
  TARGS=(); [ -n "$TURL" ] && TARGS=(--teacher-url $TURL --teacher-model qwen38 --teacher-max-tokens $TEACHER_MAX_TOKENS)
  if [ -n "$TURL" ] && [ "$TEACHER_GUARD" = 1 ]; then TARGS+=(--teacher-format-guard --teacher-resamples $TEACHER_RESAMPLES)
    case $arm in relay_repair) ;; *) TARGS+=("${PARSER_ARGS[@]}");; esac; fi   # relay_repair passes the parser below
  [ -n "$TURL" ] && [ "$VERIFY_NOTE" = 1 ] && TARGS+=(--verify-note)
  [ -n "$TURL" ] && [ "$VERIFY_NOTE" = 1 ] && [ -n "${VERIFY_NOTE_TEXT:-}" ] && TARGS+=(--verify-note-text "$VERIFY_NOTE_TEXT")
  [ -n "$TURL" ] && [ -n "${TEACHER_HINT_TEXT:-}" ] && TARGS+=(--teacher-hint-text "$TEACHER_HINT_TEXT")   # msa: teacher-only instruction
  [ "$MAX_INPUT" != 65536 ] && TARGS+=(--report-max-model-len $MAX_INPUT)
  SARGS=(); [ -n "$SURL" ] && SARGS=(--student-url $SURL --student-model snowball)   # empty on a teacher-only serve
  case $arm in control) M=(--mode teacher);; student_only) M=(--mode student);; relay) M=(--mode relay --student-think strip);; relay_keep) M=(--mode relay --student-think keep);;
    relay_repair) M=(--mode relay --student-think strip --repair-on-parse-error "${PARSER_ARGS[@]}"
                  --autofix --student-tokenizer $STUDENT_TOKENIZER);;
    *) abort "unknown arm $arm";; esac
  case $arm in relay*)
    [ -n "$CTX_BUDGET" ] && M+=(--context-budget-tokens $CTX_BUDGET)
    [ -n "$ROW_MAX" ] && M+=(--student-row-max-tokens $ROW_MAX --student-row-reserve $ROW_RESERVE)
    [ "${STUDENT_VIEW_STRIP:-0}" = 1 ] && M+=(--student-view-after-takeover strip)
    [ "${ASSIST:-0}" = 1 ] && M+=(--assist --assist-ctx "${ASSIST_CTX:-20000,35000}" --assist-error-streak ${ASSIST_ERR:-4} --assist-gap ${ASSIST_GAP:-3} --assist-max ${ASSIST_MAX:-6});; esac   # msa: one-turn teacher assists
  case $CLOCK in wall) CARGS=(--budget-mode off);; repair) CARGS=(--budget-mode on);; paused) CARGS=(--budget-mode on --pause-model-calls);;
    *) abort "unknown CLOCK $CLOCK";; esac
  $PY $ROUTER "${M[@]}" "${HARNESS_ARGS[@]}" --arm $arm --port ${PORT[$arm]} --log-dir $R/router_$arm --tasks $TREE/router_tasks.json \
    "${CARGS[@]}" --balance $BALANCE \
    --engine-metrics --deadline-epoch $DEADLINE "${SARGS[@]}" "${TARGS[@]}" $([ "$FAILOVER_5XX" = 1 ] && echo --failover-5xx) \
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
export PYTHONPATH=$HARBOR_SRC${MSA_DIR:+:$MSA_DIR}
for arm in $ARMS; do
  CFG=$R/${NAME}_$arm.yaml
  sed "s#__JOB_NAME__#${NAME}_$arm#; s#__JOBS_DIR__#$JOBS_ROOT#; s#__API_BASE__#http://127.0.0.1:${PORT[$arm]}/v1#; s#__CONC__#$CONC#; s#__MAX_INPUT__#$MAX_INPUT#; s#__MAX_OUTPUT__#$MAX_OUTPUT#; s#__AGENT_MULT__#$AGENT_MULT#; s#interleaved_thinking: true#interleaved_thinking: $MSA_INTERLEAVED#" $HERE/$JOB_TEMPLATE > $CFG
  $PY - "$CFG" "$TREE" "$MODE" "${TASK_LIST:-$TREE/TASKS.txt}" "$N_ATTEMPTS" "$SHUFFLE_SEED" <<'PY' || abort "config for $arm does not validate"
import os, random, re, sys, yaml
p, tree, mode, lst, att, seed = sys.argv[1:7]
c = yaml.safe_load(open(p))
c['n_attempts'] = int(att)
ids = [l.strip() for l in open(lst) if l.strip()]
B = {}
def budget(t):
    if t not in B:
        m = re.search(r'\[agent\][^\[]*?timeout_sec\s*=\s*([0-9.]+)', open(f'{tree}/{t}/task.toml').read())
        B[t] = float(m.group(1)) if m else 0.0
    return B[t]
occ, seen = [], {}                                # a task listed k times (a top-up of unscored rollout slots) runs k
for t in ids:                                     # times, round by round like harbor's attempts: every task's first
    occ.append(seen.get(t, 0)); seen[t] = occ[-1] + 1   # copy, then every second copy, ...
if seed == 'list':                                # the list's own order within each round (a top-up continuing a shuffle)
    ids = [ids[i] for i in sorted(range(len(ids)), key=lambda i: (occ[i], i))]
elif seed:                                        # uniform shuffled order within each round
    rk = random.Random(int(seed)).sample(range(len(ids)), len(ids))
    ids = [ids[i] for i in sorted(range(len(ids)), key=lambda i: (occ[i], rk[i]))]
else:
    ids = [ids[i] for i in sorted(range(len(ids)), key=lambda i: (occ[i], -budget(ids[i]), ids[i]))]   # longest budget first
if mode == 'smoke':
    ids = sorted(ids, key=lambda t: (budget(t), t))[:4]
c['tasks'] = [{'path': f'{tree}/{t}'} for t in ids]
yaml.safe_dump(c, open(p, 'w'), sort_keys=False)
from harbor_config.models.job.config import JobConfig
JobConfig.model_validate(c)
print(f'{os.path.basename(p)}: {len(ids)} task entries ({len(seen)} distinct) x {att}, validates, order '
      f'{"shuffled, seed " + seed if seed else "longest budget first"}; first 3: {ids[:3]}')
PY
  [ -e $JOBS_ROOT/${NAME}_$arm ] && abort "$JOBS_ROOT/${NAME}_$arm exists"
  if [ $STAGGER_SEC -gt 0 ]; then
    # stagger the sandbox start wave: two harbor jobs (every other task, longest first in each), the second
    # STAGGER_SEC later; the readout merges <name>_<arm> and <name>_<arm>_p2
    $PY - "$CFG" <<'PY'
import sys, yaml
p = sys.argv[1]; c = yaml.safe_load(open(p)); t = c['tasks']; n = c['n_concurrent_trials']
a, b = dict(c, tasks=t[0::2], n_concurrent_trials=n // 2), dict(c, tasks=t[1::2], n_concurrent_trials=n - n // 2, job_name=c['job_name'] + '_p2')
yaml.safe_dump(a, open(p, 'w'), sort_keys=False); yaml.safe_dump(b, open(p.replace('.yaml', '_p2.yaml'), 'w'), sort_keys=False)
print('staggered:', len(a['tasks']), '+', len(b['tasks']))
PY
    $HARBOR jobs start --config $CFG > $R/harbor_$arm.log 2>&1 &
    HPID[$arm]=$!
    log "harbor $arm (first half) started (pid ${HPID[$arm]}); second half in ${STAGGER_SEC}s"
    ( sleep $STAGGER_SEC; $HARBOR jobs start --config ${CFG%.yaml}_p2.yaml > $R/harbor_${arm}_p2.log 2>&1 ) &
    HPID[${arm}_p2]=$!
  else
    $HARBOR jobs start --config $CFG > $R/harbor_$arm.log 2>&1 &
    HPID[$arm]=$!
  fi
  log "harbor $arm started (pid ${HPID[$arm]}) -> $JOBS_ROOT/${NAME}_$arm"
done
T0=$(date +%s); EARLY_DONE=0; LAST_N=0; LAST_CHANGE=$T0; RELEASED_AT=""; OVF_DONE=0; STOP_LAST=0; TARGET_HIT=0; AG_LAST=0

# ---- 5. watch -------------------------------------------------------------------------------------------------------
while :; do
  sleep 60
  NOW=$(date +%s)
  for arm in $ARMS; do [ -f $R/router_$arm/FATAL ] && abort "router $arm FATAL: $(head -3 $R/router_$arm/FATAL)"; done
  N=0; for arm in $ARMS; do N=$((N + $(wc -l < $R/router_$arm/turns.jsonl 2>/dev/null || echo 0))); done
  [ $N -ne $LAST_N ] && { LAST_N=$N; LAST_CHANGE=$NOW; }
  ALIVE=0; for k in "${!HPID[@]}"; do kill -0 ${HPID[$k]} 2>/dev/null && ALIVE=$((ALIVE+1)); done
  NH=$(awk -v s=$JSTART -v n=$NOW -v k=$NODES -v d=$DSTART -v j=$DRIVER_NODES 'BEGIN{printf "%.2f", (k*(n-s) + j*(n-d))/3600}')
  [ $SERVE_RELEASED = 1 ] && NH="$NH (released)"
  log "watch: node-h=$NH requests=$N harbor_alive=$ALIVE threads=$(ps -L -u $USER --no-headers 2>/dev/null | wc -l)"
  [ $ALIVE -eq 0 ] && break
  if [ $SERVE_RELEASED = 0 ]; then
    [ -f $EP/$JOB.DEAD ] && abort "serve job reports DEAD: $(cat $EP/$JOB.DEAD)"
    squeue -h -j $JOB -o %T | grep -q RUNNING || abort "serve job $JOB not RUNNING"
    awk -v a="${NH%% *}" -v b=$CAP_NODE_H -v t=$DTAIL 'BEGIN{exit !(a>=b-t)}' && { release_serve "node-hour cap $CAP_NODE_H reached"; RELEASED_AT=$NOW; }
    if [ $NOW -ge $DEADLINE ] && [ $((NOW - LAST_CHANGE)) -ge 180 ]; then release_serve "past the deadline, no LLM traffic for 3 min"; RELEASED_AT=$NOW; fi
    [ $NOW -lt $DEADLINE ] && [ $((NOW - LAST_CHANGE)) -gt $((STALL_MIN*60)) ] && abort "no router traffic for $STALL_MIN min"
  elif [ $((NOW - RELEASED_AT)) -gt $VERIFY_WAIT ]; then
    log "harbor still running $VERIFY_WAIT s after the servers were released; stopping it"; stop_harbor; break
  fi
  if [ $GATE_MIN -gt 0 ] && [ $((NOW - T0)) -ge $((GATE_MIN*60)) ]; then
    # auto-cancel: median teacher reply latency over the last 5 min > GATE_LAT s, or an engine with KV > GATE_KV and
    # waiting requests in every sample of the last 5 min
    G=$($PY - $R $GATE_LAT $GATE_KV <<'PY'
import glob, json, sys, time, collections
R, lat_max, kv_max = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]); now = time.time(); lat = []
for p in glob.glob(R + '/router_*/turns.jsonl'):
    for l in open(p):
        try: r = json.loads(l)
        except Exception: continue
        if r.get('owner') == 'teacher' and r.get('upstream_status') == 200 and now - r['ts'] <= 300: lat.append(r['latency_sec'])
eng = collections.defaultdict(list)
for p in glob.glob(R + '/router_*/engines.jsonl'):
    for l in open(p):
        try: e = json.loads(l)
        except Exception: continue
        if now - e['ts'] <= 300: eng[e['url']].append(e)
bad = [u for u, v in eng.items() if len(v) >= 4 and all((x.get('kv') or 0) > kv_max and (x.get('waiting') or 0) > 0 for x in v)]
lat.sort(); p50 = lat[len(lat) // 2] if lat else 0
print('%.1f %d %s' % (p50, len(lat), ','.join(bad) or '-'))
PY
)
    set -- $G; log "gate: teacher p50 $1 s over the last 5 min ($2 replies); saturated engines: $3"
    awk -v a=$1 -v b=$GATE_LAT 'BEGIN{exit !(a>b)}' && abort "gate: median teacher latency $1 s > $GATE_LAT s over the last 5 min"
    [ "$3" != - ] && abort "gate: engines at KV > $GATE_KV with waiting requests for 5 min: $3"
  fi
  if [ $OVF_AFTER -gt 0 ] && [ $OVF_DONE = 0 ]; then
    for arm in $ARMS; do case $arm in relay*) ;; *) continue;; esac
      OV=$($PY - $JOBS_ROOT/${NAME}_$arm $OVF_AFTER $HERE <<'PY'
import glob, sys
sys.path.insert(0, sys.argv[3]); import readout
out = [readout.read_outcome(p) for d in (sys.argv[1], sys.argv[1] + '_p2') for p in glob.glob(f'{d}/*/result.json')]
n = len(out)
if n < int(sys.argv[2]): print('wait'); sys.exit()
print('%d %.4f' % (n, sum(1 for _, _, e in out if e == 'ContextLengthExceededError') / n))
PY
)
      if [ "$OV" != wait ] && [ -n "$OV" ]; then OVF_DONE=1; set -- $OV
        log "overflow backstop: $2 of the first $1 $arm episodes overflowed (limit $OVF_MAX)"
        awk -v a=$2 -v b=$OVF_MAX 'BEGIN{exit !(a>b)}' && abort "relay context overflow $2 > $OVF_MAX after $1 episodes"
      fi
    done
  fi
  if [ $STOP_AFTER -gt 0 ] && [ $SERVE_RELEASED = 0 ] && [ $((NOW - STOP_LAST)) -ge $STOP_EVERY ]; then
    for arm in $ARMS; do case $arm in relay*) ;; *) continue;; esac
      SR=$(timeout 600 $PY $HERE/stop_rule.py $R $NAME $arm --after $STOP_AFTER --ovf-max $STOP_OVF --fmt-min $STOP_FMT --herr-max $STOP_HERR \
           $([ -n "$HERR_EXCLUDE" ] && echo --harness-exclude-tasks $HERR_EXCLUDE) 2>>$R/stop_rule.err)
      set -- ${SR:-error}
      if [ $TARGET_FAIL -gt 0 ] && [ "$1" != stop ] && [ $# -ge 3 ] && [ "${!#}" -eq "${!#}" ] 2>/dev/null; then
        TF=$((TARGET_BASE + ${!#})); log "failure target: $TF real failures ($TARGET_BASE earlier + ${!#} this run) of $TARGET_FAIL"
        if [ $TF -ge $TARGET_FAIL ]; then
          echo "$TF" > $R/TARGET_REACHED; log "failure target $TARGET_FAIL reached ($TF); stopping harbor and finishing the run"
          stop_harbor; TARGET_HIT=1; break
        fi
      fi
      case $1 in
        wait) STOP_LAST=$((NOW - STOP_EVERY + 120)); log "stop rule $arm: $2 scored episodes (< $STOP_AFTER)";;   # re-check in 2 min
        ok) STOP_LAST=$NOW; log "stop rule $arm: ok over $2 scored (overflow $3, format $4, harness errors $5 of $6 trials)";;
        stop) abort "stop rule $arm over $2 scored: overflow $3 (max $STOP_OVF), format $4 (min $STOP_FMT), harness errors $5 (max $STOP_HERR)";;
        *) STOP_LAST=$NOW; log "stop rule $arm: evaluation failed (${SR:-no output}; see stop_rule.err)";;
      esac
    done
    [ $TARGET_HIT = 1 ] && break
  fi
  if [ $ARM_GATES = 1 ] && [ $SERVE_RELEASED = 0 ] && [ $((NOW - T0)) -ge $((ARM_GATES_FROM*60)) ] && [ $((NOW - AG_LAST)) -ge $ARM_GATES_EVERY ]; then
    AG_LAST=$NOW
    for arm in $ARMS; do case $arm in relay*) ;; *) continue;; esac
      AG=$(timeout 300 $PY $HERE/../horizon/arm_gates.py $R $NAME $arm --student-urls "$SURL" --takeover-after $TAKEOVER_AFTER \
           ${TAKEOVER_MIN:+--takeover-min $TAKEOVER_MIN} ${TAKEOVER_MAX:+--takeover-max $TAKEOVER_MAX} --takeover-action $TAKEOVER_ACTION \
           --accept-flag $ACCEPT_FLAG --accept-stop $ACCEPT_STOP 2>>$R/arm_gates.err)
      log "arm gates $arm: ${AG:-evaluation failed (see arm_gates.err)}"
      case "${AG%% *}" in
        stop) abort "arm gate $arm: ${AG#stop }";;
        flag) echo "[$(date -Is)] $arm ${AG#flag }" >> $R/FLAGS;;
      esac
    done
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
if [ $TARGET_HIT = 1 ]; then   # let the interrupted harbor jobs write their results before the readout (at most 5 min)
  for i in $(seq 1 30); do A=0; for k in "${!HPID[@]}"; do kill -0 ${HPID[$k]} 2>/dev/null && A=1; done; [ $A = 0 ] && break; sleep 10; done
fi
stop_routers
release_serve "harbor done"
sleep 20; write_meta
cleanup_sandboxes
$PY $HERE/readout.py --run-dir $R --name $NAME --gate final --json $R/readout.json \
  ${TAKEOVER_MIN:+--takeover-min $TAKEOVER_MIN} ${TAKEOVER_MAX:+--takeover-max $TAKEOVER_MAX} > /dev/null 2> $R/readout.txt
cat $R/readout.txt
log "RUN_DONE $(tr '\n' ' ' < $R/run.meta)"
[ "${DRIVER_IN_SERVE:-0}" = 1 ] && { log "serve job $JOB cancelled after the readout"; scancel $JOB; }
