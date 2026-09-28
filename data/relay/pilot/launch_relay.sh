#!/bin/bash
# launch_relay.sh <relay run name>
# Jupiter login node. Launches the full run's relay arm (Luke 2026-09-26 15:00 PT) with the context-budget check's exact
# settings (relay_ctxb2_20260926: run_pilot.sh, CLOCK=repair, context_budget 32k, the 64k row hard end, Qwen at 128k with
# one server per GPU, the teacher format guard, the verify note) on the baseline-solvable tasks x ROLLOUTS, on
# 4 x 09-21 + 4 x Qwen.
#
# Cost: a flat CAP_NODE_H (30) for this run. It replaces the old 42 / 47 node-hour accounting that charged the baseline,
# the checks and reserves. The serve job's --time is CAP_NODE_H / 8 nodes, so Slurm stops it at the cap; the routers'
# deadline ends episodes 5 min before that, and deadline-ended episodes are censored (dropped) by the readout. Harbor
# runs rollout 1 of every task before rollout 2 (attempts are the outer loop), so a tail cut by the cap loses late
# rollouts, not whole tasks.
#
# Launch rule (pre-registered before the ctxb2 result): the check PASSES, or it fails only C2 with overflow <= 25 %
# (overflowed episodes are scored failures that the keep filter drops; they cost compute, not data quality).
# In-run stop rule: after >= 300 scored relay episodes, cancel on overflow > 30 %, valid format < 98 % or harness
# errors > 10 % (run_pilot.sh STOP_*, stop_rule.py; re-evaluated every 15 min).
# State: $E/runs/<name>.launch.log, the plan in $E/runs/<name>.plan.json.
# TOPUP=<list> (attempt 3 on, Luke 2026-09-26 16:20 PT): run only the (task, rollout) slots the earlier attempts did not
# score: the list (merge_runs.py --remaining) holds each solvable task once per unscored slot, run_pilot.sh runs it round
# by round at 1 rollout per entry, so no slot is paid for twice. CAP_NODE_H is then what the 30 leaves.
# Attempt 5 on (Luke 2026-09-27, 2,000 kept per arm): ROLLOUTS=7 (tries 5-7 via merge_runs.py --per-task 7 --remaining),
# HERR_EXCLUDE (stop_rule.py --harness-exclude-tasks) and TARGET_FAIL / TARGET_BASE (end the run on the merged arm's
# real-failure count) pass through to run_pilot.sh. Attempt 6: SHUFFLE_SEED passes through (list = the TOPUP list's own
# order: two rounds of the tasks without relay trials, shuffled with seed 20260927), and TASKS_CHECK=pool checks the
# top-up against full_pool.txt instead of the solvable list.
set -uo pipefail
NAME=${1:?relay run name}
C=/e/project1/transfernetx/lee27/code
HERE=$(cd "$(dirname "$0")" && pwd)
PY=$C/envs/snowball-v2/bin/python
E=/e/fscratch/reformo/lee27/experiments/relay/pilot; RUNS=$E/runs
CHECK=${CHECK:-relay_ctxb2_20260926}; BASE=${BASE:-relay_full_baseline_20260925}
TASKS=${TASKS:-$RUNS/$BASE/solvable_tasks.txt}
TREE=/e/fscratch/reformo/lee27/tasks/calibforge_full2043
JOBS_ROOT=/e/data1/mmlaion/lee27/experiments/relay_full_jobs
CAP_NODE_H=${CAP_NODE_H:-30}; NODES=8; ROLLOUTS=${ROLLOUTS:-4}; CONC=${CONC:-192}   # 12 episodes per Qwen GPU (16 GPUs)
L=$RUNS/$NAME.launch.log
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
log() { echo "[$(date -Is)] $*" | tee -a $L; }
[ -e $RUNS/$NAME ] && { log "$RUNS/$NAME exists"; exit 1; }
$PY $HERE/check_decide.py --check $RUNS/$CHECK --run3 $RUNS/relay_run3b_20260925 --rule ctxbudget > $RUNS/$CHECK/check_decision.json
DEC=$($PY - $RUNS/$CHECK/check_decision.json <<'PY'
import json, sys
d = json.load(open(sys.argv[1])); bad = [c for c in d['checks'] if not c['ok']]
ok = d['decision'] == 'PASS' or (len(bad) == 1 and bad[0]['check'].startswith('C2') and bad[0]['detail']['rate'] <= 0.25)
print('LAUNCH' if ok else 'HOLD', d['decision'], [(c['check'][:2], c['detail'].get('rate') if isinstance(c['detail'], dict) else None) for c in bad])
PY
)
log "check $CHECK: $DEC"
[ "${DEC%% *}" = LAUNCH ] || { log "HOLD: the check is outside the launch rule; relay not launched"; exit 1; }
$PY $HERE/relay_plan.py --baseline $RUNS/$BASE --check $RUNS/$CHECK --check-decision $RUNS/$CHECK/check_decision.json \
  --spent 0 --total-ceiling $CAP_NODE_H --conc $CONC --rollouts $ROLLOUTS --out-tasks $RUNS/$NAME.solvable_recomputed.txt > $RUNS/$NAME.plan.json
diff -q <(sort $TASKS) <(sort $RUNS/$NAME.solvable_recomputed.txt) >/dev/null || { log "solvable list $TASKS differs from the baseline's passes"; exit 1; }
log "plan (projection only; the cap is the ceiling): $(tr -d '\n' < $RUNS/$NAME.plan.json | cut -c1-700)"
if [ -n "${TOPUP:-}" ]; then
  [ "${TASKS_CHECK:-solvable}" = pool ] && TASKS=$HERE/full_pool.txt
  $PY - $TOPUP $TASKS $ROLLOUTS <<'PY' || { log "top-up list $TOPUP is not a subset of the solvable slots"; exit 1; }
import collections, sys
top = collections.Counter(l.strip() for l in open(sys.argv[1]) if l.strip()); sol = {l.strip() for l in open(sys.argv[2]) if l.strip()}
bad = [t for t, k in top.items() if t not in sol or k > int(sys.argv[3])]
print('top-up: %d slots over %d tasks, bad %d' % (sum(top.values()), len(top), len(bad))); sys.exit(1 if bad or not top else 0)
PY
  log "top-up $TOPUP: $(wc -l < $TOPUP) unscored slots over $(sort -u $TOPUP | wc -l) tasks, 1 rollout per entry"
  TASKS=$TOPUP; ROLLOUTS=1
fi
TMIN=$($PY -c "print(int($CAP_NODE_H / $NODES * 60))")
JOB=$(RELAY_PILOT_DIR=$HERE N_STUDENT=4 PER_GPU=1 TEACHER_MAXLEN=131072 sbatch --parsable --export=ALL --nodes=$NODES --time=$TMIN --job-name=relay_full_relay $HERE/serve_relay.sbatch) \
  || { log "sbatch failed"; exit 1; }
log "serve job $JOB submitted ($NODES nodes: 4 x 09-21 + 4 x Qwen per-GPU 128k, $TMIN min = $CAP_NODE_H node-h)"
tmux new -d -s relay_full_relay "ARMS=relay_repair CLOCK=repair CTX_BUDGET=32000 ROW_MAX=65536 ROW_RESERVE=8192 MAX_INPUT=131072 \
TEACHER_MAX_TOKENS=32768 BALANCE=active STAGGER_SEC=180 GATE_MIN=15 GATE_LAT=30 GATE_KV=0.90 TEACHER_GUARD=1 VERIFY_NOTE=1 \
CONC=$CONC NODES=$NODES CAP_NODE_H=$CAP_NODE_H NTASKS=2043 TREE=$TREE TASK_LIST=$TASKS N_ATTEMPTS=$ROLLOUTS \
STOP_AFTER=300 STOP_OVF=0.30 STOP_FMT=0.98 STOP_HERR=0.10 HERR_EXCLUDE=${HERR_EXCLUDE:-} TARGET_FAIL=${TARGET_FAIL:-0} TARGET_BASE=${TARGET_BASE:-0} SHUFFLE_SEED=${SHUFFLE_SEED:-} \
JOBS_ROOT=$JOBS_ROOT bash $HERE/run_pilot.sh $JOB $NAME"
log "LAUNCHED: driver in tmux relay_full_relay (run $NAME, serve job $JOB)"
