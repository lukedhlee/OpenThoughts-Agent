#!/bin/bash
# run_tb2.sh <serve job> <run name> [smoke] — Horizon login node. A harbor eval (Terminal-Bench 2 / 2.1, SWE-bench Verified
# random-100, TB-lite) on Daytona against the student server of a serve_relay.sbatch job, under a Jupiter policy file.
# By diff from data/tb2/jupiter/run_tb2.sh: the same checks (harbor pin, task count, Daytona key), the same policy
# rendering (data/tb2/jupiter/$POLICY_FILE, jobs_dir swapped to Horizon's), the same task order (longest agent timeout
# first), SHARD=i/n and EXCLUDE_TASKS; but harbor runs in its own compute-node job (tb2_driver.sbatch) behind the
# login-side tunnels, because Horizon compute nodes have no internet and the login node is shared.
#
# The model server: serve_submit.sh 1 1 <time> <name> (serve_relay.sbatch, N_STUDENT=1), whose student command is
# serve_snowball.sbatch's POLICY=trained serve (EAGLE-3 draft, TP1 x DP4 x EP, 65,536 context, 32 seqs per replica).
#
#   HARBOR_SRC=$HOME/snowball/harbor-p0924/src HARBOR_SHA=761fb516 POLICY_FILE=tb2_marin_policy_0924_65k16k.yaml \
#     TASKS=/scratch/11584/lukedhlee/tasks/terminal_bench_2_1 NTASKS=89 EXCLUDE_TASKS=train-fasttext \
#     bash run_tb2.sh <serve job> tb21_hz_A_20260930
set -euo pipefail
JOB=${1:?serve job id}; NAME=${2:?run name}; MODE=${3:-full}
HERE=$(cd "$(dirname "$0")" && pwd); OTA=$(cd $HERE/../../.. && pwd); JP=$OTA/data/tb2/jupiter
source $OTA/data/relay/horizon/serve_env.sh   # RELAY_EXP_DIR: where serve_relay.sbatch writes its endpoint files
S=${SCRATCH_DIR:-/scratch/11584/$USER}
E=${TB2_EXP_DIR:-$S/experiments/tb2}; JOBS=${JOBS:-$S/experiments/tb2_jobs}; mkdir -p $E/runs $E/logs $JOBS
PY=${PY:-$HOME/snowball/envs/snowball/bin/python}
KEYF=${KEYF:-$HOME/.config/otagent/daytona_eval.env}
HARBOR_SRC=${HARBOR_SRC:?HARBOR_SRC=<harbor clone>/src}; HARBOR_SHA=${HARBOR_SHA:?HARBOR_SHA=<8+ hex>}
POLICY_FILE=${POLICY_FILE:?POLICY_FILE=<file in data/tb2/jupiter>}
TASKS=${TASKS:?TASKS=<task tree>}; NTASKS=${NTASKS:?NTASKS=<task count>}
export OMP_NUM_THREADS=1
[ "$(git -C ${HARBOR_SRC%/src} rev-parse --short=8 HEAD)" = "${HARBOR_SHA:0:8}" ] || { echo "$HARBOR_SRC is not at $HARBOR_SHA"; exit 1; }
[ "$(find $TASKS -mindepth 2 -maxdepth 2 -name task.toml | wc -l)" = "$NTASKS" ] || { echo "task tree at $TASKS does not have $NTASKS tasks"; exit 1; }
URL=$(cut -d, -f1 $RELAY_EXP_DIR/endpoints/$JOB.student 2>/dev/null) || { echo "no student endpoint for job $JOB (server not up, or gone)"; exit 1; }
squeue -h -j $JOB -o %T | grep -q RUNNING || { echo "serve job $JOB is not RUNNING"; exit 1; }
[ -d $JOBS/$NAME ] && { echo "$JOBS/$NAME exists; pick a new name"; exit 1; }
grep -q "type: daytona" $JP/$POLICY_FILE || { echo "$POLICY_FILE is not a Daytona policy"; exit 1; }
set -a; source $KEYF; set +a
curl -sf --max-time 20 -H "Authorization: Bearer $DAYTONA_API_KEY" https://app.daytona.io/api/api-keys/current >/dev/null || { echo "Daytona key rejected"; exit 1; }
unset DAYTONA_API_KEY
# render the policy for this run (Horizon's jobs dir; the rest verbatim)
CFG=$E/runs/$NAME.yaml
sed "s#__JOB_NAME__#$NAME#; s#__API_BASE__#$URL#; s#^jobs_dir: .*#jobs_dir: $JOBS#" $JP/$POLICY_FILE > $CFG
MODE=$MODE $PY - "$CFG" "$TASKS" <<'PY'
import os, re, sys, yaml
p, tasks = sys.argv[1:3]; c = yaml.safe_load(open(p))
def agent_timeout(t):
    s = open(f"{tasks}/{t}/task.toml").read(); m = re.search(r"\[agent\][^\[]*?timeout_sec\s*=\s*([0-9.]+)", s)
    return float(m.group(1)) if m else 0.0
names = sorted((d for d in os.listdir(tasks) if os.path.isfile(f"{tasks}/{d}/task.toml")), key=lambda t: (-agent_timeout(t), t))
skip = [t for t in os.environ.get("EXCLUDE_TASKS", "").split(",") if t]   # tasks that cannot run (infra losses)
if skip:
    names = [t for t in names if t not in skip]; print(f"excluded: {skip}")
shard = os.environ.get("SHARD")   # SHARD=i/n: every n-th task starting at i (interleaved so shards are alike)
if shard:
    i, n = map(int, shard.split("/")); names = names[i::n]
if os.environ["MODE"] == "smoke":   # two shortest tasks once at 2 concurrent
    names = sorted(names, key=lambda t: (agent_timeout(t), t))[:2]; c["n_concurrent_trials"] = 2
c["n_attempts"] = 1
c.pop("datasets", None); c["tasks"] = [{"path": f"{tasks}/{t}"} for t in names]
yaml.safe_dump(c, open(p, "w"), sort_keys=False)
print(f"{len(names)} tasks at {c['n_concurrent_trials']} concurrent, longest agent timeout first ({names[0]} {agent_timeout(names[0]):.0f}s ... {names[-1]} {agent_timeout(names[-1]):.0f}s)")
PY
PYTHONPATH=$HARBOR_SRC $PY -c "import yaml; from harbor_config.models.job.config import JobConfig; JobConfig.model_validate(yaml.safe_load(open('$CFG'))); print('policy validates')"
# the driver job and its tunnels
T=${DRIVER_TIME:-08:00:00}
export TUNNEL_PORTS=${TUNNEL_PORTS:-18080,18081}
DJ=$(CFG=$CFG HARBOR_SRC=$HARBOR_SRC SERVE_JOB=$JOB RUN_NAME=$NAME JOBS=$JOBS OTA=$OTA \
  sbatch --parsable --export=ALL -J tb2_drv_$NAME -t $T -o $E/logs/%x_%j.out $HERE/tb2_driver.sbatch | tail -1)
[ -n "$DJ" ] && [ "$DJ" -eq "$DJ" ] 2>/dev/null || { echo "sbatch failed: $DJ"; exit 1; }
export TUNNEL_PORTS=${TUNNEL_PORTS:-18080,18081}
for p in ${TUNNEL_PORTS//,/ }; do
  setsid nohup bash $OTA/data/r2egym/horizon/tunnel.sh $DJ $p > $E/logs/tunnel_${DJ}_$p.log 2>&1 < /dev/null &
done
echo "[$(date -Is)] driver job $DJ ($T) for serve job $JOB, run $NAME ($MODE); log $E/logs/tb2_drv_${NAME}_$DJ.out; results $JOBS/$NAME"
