#!/bin/bash
# run_tb2.sh <serve-jobid> <run-name> [smoke]
# On a Jupiter login node: point harbor (marin-community/harbor @ 7b18505a, the v0.1 pin) at the vLLM server that
# serve_snowball.sbatch started, and run Terminal-Bench 2 on Daytona in a tmux session under marin's policy
# (tb2_marin_policy.yaml). "smoke" runs two tasks once at 2 concurrent (marin's tb2-lite shape) instead of the
# full 89 x 3. Never start the same run name twice: two launches into one job dir corrupt it; resume instead
# (harbor jobs resume -p <jobs_dir>/<name>, after `set -a; source $KEYF; set +a`).
set -euo pipefail
JOB=${1:?serve job id}; NAME=${2:?run name (model, policy, date)}; MODE=${3:-full}
C=/e/project1/transfernetx/lee27/code; W=$C/tb2
E=/e/fscratch/reformo/lee27/experiments/tb2; JOBS=/e/data1/mmlaion/lee27/experiments/tb2_jobs
HARBOR_SRC=/e/fscratch/reformo/lee27/code/harbor-v01/src   # marin-community/harbor @ 7b18505a, the v0.1 pin
PY=$C/envs/snowball-v2/bin/python; HARBOR=$C/envs/snowball-v2/bin/harbor
KEYF=/e/fscratch/reformo/lee27/keys/daytona_eval.env
TASKS=${TASKS:-/e/fscratch/reformo/lee27/tasks/terminal_bench_2}   # TASKS=<dir> for another benchmark (swebench_verified_random100)
NTASKS=${NTASKS:-89}
# SHARD=i/n splits the task list across servers (same 16-wide concurrency per server; merge the job dirs to score)
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1   # login-node pid cap
[ "$(git -C ${HARBOR_SRC%/src} rev-parse --short=8 HEAD)" = 7b18505a ] || { echo "harbor-v01 is not at the v0.1 pin 7b18505a"; exit 1; }
[ "$(ls -d $TASKS/*/ | wc -l)" = "$NTASKS" ] || { echo "task tree at $TASKS does not have $NTASKS tasks"; exit 1; }
URL=$(cat $E/endpoints/$JOB 2>/dev/null) || { echo "no endpoint file for job $JOB (server not up, or gone)"; exit 1; }
squeue -h -j $JOB -o %T | grep -q RUNNING || { echo "serve job $JOB is not RUNNING"; exit 1; }
[ -d $JOBS/$NAME ] && { echo "$JOBS/$NAME exists; pick a new name or resume"; exit 1; }
mkdir -p $JOBS $W/runs
# 1. the server answers from here, with the served name and the think markers intact
curl -sf --max-time 20 $URL/models | grep -q '"snowball"' || { echo "no model 'snowball' at $URL"; exit 1; }
curl -sf --max-time 300 $URL/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"snowball","messages":[{"role":"user","content":"Print hello in bash."}],"max_tokens":300,"skip_special_tokens":false}' \
  | $PY -c 'import json,sys; r=json.load(sys.stdin); print("server ok:", r["usage"], repr((r["choices"][0]["message"].get("content") or "")[:160]))'
# 2. the sandbox backend: Daytona (key + the global TB2 snapshots) or the apptainer bridge the serve job started
BURL=""
if grep -q "type: apptainer" $W/${POLICY_FILE:-tb2_marin_policy.yaml}; then
  BURL=$(cat $E/endpoints/$JOB.bridge 2>/dev/null) || { echo "no bridge file for job $JOB (use ifv2_apptainer.sbatch)"; exit 1; }
  curl -s --max-time 20 $BURL/ >/dev/null || { echo "bridge $BURL not answering"; exit 1; }
else
  set -a; source $KEYF; set +a
  curl -sf --max-time 20 -H "Authorization: Bearer $DAYTONA_API_KEY" https://app.daytona.io/api/api-keys/current >/dev/null || { echo "Daytona key rejected"; exit 1; }
fi
# 3. render the policy for this run
CFG=$W/runs/$NAME.yaml
sed "s#__JOB_NAME__#$NAME#; s#__API_BASE__#$URL#; s#__BRIDGE_URL__#$BURL#" $W/${POLICY_FILE:-tb2_marin_policy.yaml} > $CFG
if [ "$MODE" = smoke ]; then
  $PY - "$CFG" "$TASKS" <<'PY'
import sys, yaml
p, tasks = sys.argv[1:3]; c = yaml.safe_load(open(p))
c["n_attempts"] = 1; c["n_concurrent_trials"] = 2; c.pop("datasets", None)
import os; first = sorted(d for d in os.listdir(tasks) if os.path.isfile(f"{tasks}/{d}/task.toml"))[:2]
c["tasks"] = [{"path": f"{tasks}/{t}"} for t in first]
yaml.safe_dump(c, open(p, "w"), sort_keys=False); print("smoke: 2 tasks x 1 attempt")
PY
fi
if [ "$MODE" != smoke ]; then
  # Longest agent timeout first: harbor runs trials in config order, and a 12,000 s task started last holds a whole
  # node for 3 h after everything else is done (Luke 2026-09-17). Same tasks, same settings, only the order.
  $PY - "$CFG" "$TASKS" <<'PY'
import sys, yaml, re, os
p, tasks = sys.argv[1:3]; c = yaml.safe_load(open(p))
def agent_timeout(t):
    s = open(f"{tasks}/{t}/task.toml").read(); m = re.search(r"\[agent\][^\[]*?timeout_sec\s*=\s*([0-9.]+)", s)
    return float(m.group(1)) if m else 0.0
names = sorted((d for d in os.listdir(tasks) if os.path.isfile(f"{tasks}/{d}/task.toml")), key=lambda t: (-agent_timeout(t), t))
shard = os.environ.get("SHARD")   # SHARD=i/n: take every n-th task starting at i (interleaved so shards are alike)
if shard:
    i, n = map(int, shard.split("/")); names = names[i::n]
c.pop("datasets", None); c["tasks"] = [{"path": f"{tasks}/{t}"} for t in names]
yaml.safe_dump(c, open(p, "w"), sort_keys=False); print(f"{len(names)} tasks, longest agent timeout first ({names[0]} {agent_timeout(names[0]):.0f}s ... {names[-1]} {agent_timeout(names[-1]):.0f}s)")
PY
fi
$PY -c "import yaml,sys; sys.path.insert(0,'$HARBOR_SRC'); from harbor_config.models.job.config import JobConfig; JobConfig.model_validate(yaml.safe_load(open('$CFG'))); print('policy validates')"
# 4. run in tmux; the log is the tmux pane plus harbor's own job dir
LOG=$E/logs/run_${NAME}.log; mkdir -p $E/logs
tmux new -d -s tb2_$NAME "export OMP_NUM_THREADS=1 PYTHONPATH=$HARBOR_SRC; set -a; source $KEYF; set +a; cd $W; $HARBOR run -c $CFG -y 2>&1 | tee $LOG; echo RUN_DONE exit=\${PIPESTATUS[0]} >> $LOG"
echo "started tmux tb2_$NAME on $(hostname); log $LOG; results $JOBS/$NAME"
echo "watch: python3 $W/summarize_tb2.py $JOBS/$NAME"
