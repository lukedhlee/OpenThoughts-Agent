#!/bin/bash
# launch_relay.sh <check run name> <baseline run name> <relay run name>
# Jupiter login node, in tmux. Launches the full run's relay arm automatically when BOTH hold (Luke 2026-09-25 18:40 PT):
#   1. the check run passed its pre-registered rule (check_decide.py, C1-C5)
#   2. the baseline run has FINISHED (decision: wait for all of it; its tail is the long-budget tasks, which are also the
#      ones most worth relaying, and a complete solvable list keeps the plan's arithmetic exact)
# then plans it (relay_plan.py: solvable tasks, rollouts per task, cost within 42 - baseline node-hours) and, on LAUNCH,
# submits 4 x 09-21 + 4 x Qwen (--time = the relay ceiling / 8 nodes) and run_pilot.sh on the solvable list with the
# overflow backstop (cancel when > 20 % of the first 200 relay episodes overflow). On FAIL / HOLD it writes the reason
# and stops. State: $E/runs/<relay run name>.launch.log
set -uo pipefail
CHECK=${1:?check run}; BASE=${2:?baseline run}; NAME=${3:?relay run name}
C=/e/project1/transfernetx/lee27/code
HERE=$(cd "$(dirname "$0")" && pwd)
PY=$C/envs/snowball-v2/bin/python
E=/e/fscratch/reformo/lee27/experiments/relay/pilot; RUNS=$E/runs
TREE=/e/fscratch/reformo/lee27/tasks/calibforge_full2043
JOBS_ROOT=/e/data1/mmlaion/lee27/experiments/relay_full_jobs
L=$RUNS/$NAME.launch.log
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
log() { echo "[$(date -Is)] $*" | tee -a $L; }
log "launch_relay: waiting for check $CHECK and baseline $BASE"
until grep -qE "RUN_DONE|ABORT" $RUNS/$CHECK/driver.log 2>/dev/null; do sleep 120; done
if grep -q ABORT $RUNS/$CHECK/driver.log; then log "HOLD: the check run aborted: $(grep ABORT $RUNS/$CHECK/driver.log | tail -1)"; exit 1; fi
$PY $HERE/check_decide.py --check $RUNS/$CHECK --run3 $RUNS/relay_run3b_20260925 --rule ${RULE:-check} > $RUNS/$CHECK/check_decision.json
DEC=$($PY -c "import json; print(json.load(open('$RUNS/$CHECK/check_decision.json'))['decision'])")
log "check run decision: $DEC ($($PY -c "import json; d=json.load(open('$RUNS/$CHECK/check_decision.json')); print([(c['check'][:3], c['ok']) for c in d['checks']])"))"
[ "$DEC" = PASS ] || { log "HOLD: the check run failed its rule; relay not launched"; exit 1; }
until grep -qE "RUN_DONE|ABORT" $RUNS/$BASE/driver.log 2>/dev/null; do sleep 120; done
grep -q ABORT $RUNS/$BASE/driver.log && log "note: the baseline aborted ($(grep ABORT $RUNS/$BASE/driver.log | tail -1)); planning from what finished"
# node-hours charged to the full run before the relay: the runs finished so far (BASE, CHECK and EXTRA_RUNS, from their
# run.meta) plus RESERVE, the ceiling of a job still running (the baseline rerun, 21:30 PT)
SPENT=$($PY -c "
import sys
t = float(sys.argv[1])
for r in sys.argv[2:]:
    t += float(dict(l.strip().split('=', 1) for l in open(r + '/run.meta') if '=' in l)['node_hours'])
print(round(t, 3))" ${RESERVE:-0} $RUNS/$BASE $RUNS/$CHECK $(for x in ${EXTRA_RUNS:-}; do echo $RUNS/$x; done))
log "charged to the full run before the relay: $SPENT node-h ($BASE, $CHECK, ${EXTRA_RUNS:-}, reserve ${RESERVE:-0})"
$PY $HERE/relay_plan.py --baseline $RUNS/$BASE --check $RUNS/$CHECK --check-decision $RUNS/$CHECK/check_decision.json \
  --spent $SPENT --total-ceiling ${TOTAL_CEILING:-42} ${ROLLOUTS:+--rollouts $ROLLOUTS} --out-tasks $RUNS/$NAME.solvable.txt > $RUNS/$NAME.plan.json
PD=$($PY -c "import json; d=json.load(open('$RUNS/$NAME.plan.json')); print(d['decision'], d['rollouts_per_task'], d['relay_time_h'], d['relay_ceiling_node_h'], d['solvable_tasks'])")
set -- $PD; PDEC=$1; R=$2; TH=$3; CEIL=$4; NSOLV=$5
log "relay plan: $PDEC, $NSOLV solvable tasks x $R, --time ${TH} h, ceiling $CEIL node-h ($(tr -d '\n' < $RUNS/$NAME.plan.json | cut -c1-600))"
[ "$PDEC" = LAUNCH ] || { log "HOLD: relay plan does not fit; not launched"; exit 1; }
TMIN=$($PY -c "print(int(float('$TH') * 60))")
JOB=$(RELAY_PILOT_DIR=$HERE N_STUDENT=4 TEACHER_MAXLEN=${TEACHER_MAXLEN:-131072} sbatch --parsable --export=ALL --nodes=8 --time=$TMIN --job-name=relay_full_relay $HERE/serve_relay.sbatch) \
  || { log "sbatch failed"; exit 1; }
log "relay serve job $JOB submitted (8 nodes, $TMIN min)"
tmux new -d -s relay_full_relay "MAX_INPUT=${TEACHER_MAXLEN:-131072} TEACHER_MAX_TOKENS=32768 ARMS=relay_repair NODES=8 CAP_NODE_H=$CEIL CONC=400 NTASKS=2043 TREE=$TREE TASK_LIST=$RUNS/$NAME.solvable.txt N_ATTEMPTS=$R OVF_AFTER=200 OVF_MAX=0.20 JOBS_ROOT=$JOBS_ROOT bash $HERE/run_pilot.sh $JOB $NAME"
log "LAUNCHED relay driver in tmux relay_full_relay (run $NAME)"
