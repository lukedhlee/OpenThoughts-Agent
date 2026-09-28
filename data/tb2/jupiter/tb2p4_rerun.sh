#!/bin/bash
# tb2p4_rerun.sh <run-name> [<run-name> ...]: one pooled recovery wave for several finished TB2 runs of one checkpoint (the
# tb2p4 pass@k study), on ONE fresh server, rerunning only the trials lost to infrastructure.
#  1. waits for every run's $E/tb2p4_rundone_<name> (tb2p4_finish.sh);
#  2. lists the unscored trials (no verifier result) per run; planned-but-excluded tasks (EXCLUDED per run) count as lost;
#     if all losses are 10 % of the planned trials or fewer it stops (the policy's tolerance);
#  3. otherwise submits one serve job (MODEL, 1 node, wall sized to the waves) and starts, per run, a new harbor job
#     <name>_rec rendered from that run's own config ($W/runs/<name>.yaml) with only its unscored tasks, the new api_base, and
#     a share of the reference 16 concurrent slots proportional to its reruns (the server never carries more than 16);
#     SKIP_TASKS (default filter-js-from-html: its Selenium verifier outlives the policy's 300 s verifier limit in every run)
#     are not rerun and stay infrastructure losses;
#  4. waits for all of them, writes $E/final_<name>_rec.txt, releases the server. The analysis merges <name>_rec into <name>
#     (a rerun trial replaces that run's unscored trial of the same task).
# Run in tmux rec_<tag>. Log: $E/tb2p4_rerun.log
set -o pipefail; NAMES="$*"
C=/e/project1/transfernetx/lee27/code; W=$C/tb2; E=/e/fscratch/reformo/lee27/experiments/tb2; JD=/e/data1/mmlaion/lee27/experiments/tb2_jobs
KEYF=/e/fscratch/reformo/lee27/keys/daytona_eval.env; PY=$C/envs/snowball-v2/bin/python; HARBOR=$C/envs/snowball-v2/bin/harbor
MODEL=${MODEL:-/e/data1/mmlaion/lee27/models/grug-datakit-sft-20260921}; ACCT=${ACCT:-transfernetx}
EXCLUDED=${EXCLUDED:-1}; SKIP_TASKS=${SKIP_TASKS-filter-js-from-html}
export OMP_NUM_THREADS=1 PYTHONPATH=$C/harbor-p0924/src
log() { echo "$(date -Is) $*" | tee -a $E/tb2p4_rerun.log; }
log "waiting for rundone markers of: $NAMES"
for N in $NAMES; do while [ ! -f $E/tb2p4_rundone_$N ]; do sleep 60; done; done
PLAN=$E/tb2p4_rerun_plan.json
$PY - $JD $EXCLUDED "$SKIP_TASKS" $PLAN $NAMES <<'PY' | tee -a $E/tb2p4_rerun.log
import json, sys, glob
jd, excl, skip, out, names = sys.argv[1], int(sys.argv[2]), set(filter(None, sys.argv[3].split(","))), sys.argv[4], sys.argv[5:]
plan, lost, planned = {}, 0, 0
for n in names:
    tasks = [t["path"] for t in json.load(open(f"{jd}/{n}/config.json"))["tasks"]]; planned += len(tasks) + excl; lost += excl
    scored = set()
    for p in glob.glob(f"{jd}/{n}/*__*/result.json"):
        try: r = json.load(open(p))
        except Exception: continue
        if r.get("verifier_result") is not None: scored.add(r["task_name"])
    miss = [t for t in tasks if t.rstrip("/").split("/")[-1] not in scored]; lost += len(miss)
    plan[n] = [t for t in miss if t.rstrip("/").split("/")[-1] not in skip]
tot = sum(len(v) for v in plan.values())
alloc = {n: len(v) for n, v in plan.items() if v}
if tot > 16:
    alloc = {n: max(1, (16 * v) // tot) for n, v in alloc.items()}
    while sum(alloc.values()) > 16: alloc[max(alloc, key=alloc.get)] -= 1
    while sum(alloc.values()) < 16: alloc[max(alloc, key=lambda n: len(plan[n]) / alloc[n])] += 1
waves = max((len(plan[n]) + alloc[n] - 1) // alloc[n] for n in alloc) if alloc else 0
json.dump({"plan": plan, "alloc": alloc, "lost": lost, "planned": planned, "reruns": tot, "waves": waves}, open(out, "w"), indent=1)
print(f"lost {lost} of {planned} planned ({100*lost/planned:.1f} %); reruns {tot} (skipped {sorted(skip)}); concurrency {alloc}; waves {waves}")
PY
read -r LOST PLANNED WAVES < <($PY -c "import json; d=json.load(open('$PLAN')); print(d['lost'], d['planned'], d['waves'])")
if [ "$(echo $LOST $PLANNED | awk '{print ($1 > 0.10*$2) ? 1 : 0}')" != 1 ]; then log "within the 10 % tolerance: no recovery wave"; exit 0; fi
[ "$WAVES" -gt 0 ] || { log "nothing rerunnable"; exit 0; }
MIN=$(( 15 + 40 * WAVES )); [ $MIN -gt 95 ] && MIN=95; WALL=$(printf "%02d:%02d:00" $((MIN / 60)) $((MIN % 60)))
cd $W
JOB=$(MODEL=$MODEL POLICY=trained sbatch --parsable -A $ACCT --job-name=tb2p4_0921_rec --time=$WALL serve_snowball.sbatch)
log "recovery serve job $JOB ($ACCT, wall $WALL, $WAVES wave(s))"
while [ ! -f $E/endpoints/$JOB ]; do
  squeue -h -j $JOB -o %T | grep -qE "PENDING|RUNNING|CONFIGURING" || { log "FAILED: serve job $JOB left the queue before writing an endpoint"; exit 1; }
  sleep 30
done
URL=$(cat $E/endpoints/$JOB); log "endpoint $URL"
for i in $(seq 1 30); do curl -sf --max-time 20 $URL/models | grep -q '"snowball"' && break; sleep 10; done
PIDS=()
for N in $($PY -c "import json; print(' '.join(json.load(open('$PLAN'))['alloc']))"); do
  CFG=$W/runs/${N}_rec.yaml
  $PY - $W/runs/$N.yaml $CFG $PLAN $N $URL <<'PY' || { log "FAILED: render $N"; continue; }
import json, sys, yaml
src, dst, plan, n, url = sys.argv[1:6]; c = yaml.safe_load(open(src)); p = json.load(open(plan))
c["job_name"] = f"{n}_rec"; c["n_concurrent_trials"] = p["alloc"][n]; c["tasks"] = [{"path": t} for t in p["plan"][n]]
for a in c["agents"]: a["kwargs"]["api_base"] = url
yaml.safe_dump(c, open(dst, "w"), sort_keys=False)
from harbor_config.models.job.config import JobConfig; JobConfig.model_validate(c)
print(f"{n}_rec: {len(c['tasks'])} tasks at {c['n_concurrent_trials']} concurrent")
PY
  log "rerun $N -> ${N}_rec ($CFG)"
  ( set -a; source $KEYF; set +a; $HARBOR run -c $CFG -y > $E/logs/run_${N}_rec.log 2>&1; echo "RUN_DONE exit=$?" >> $E/logs/run_${N}_rec.log ) &
  PIDS+=($!)
done
wait "${PIDS[@]}"
for N in $NAMES; do [ -d $JD/${N}_rec ] && $PY $W/summarize_tb2.py $JD/${N}_rec | tee $E/final_${N}_rec.txt | tee -a $E/tb2p4_rerun.log; done
touch $E/tb2p4_rerun.DONE; log "releasing recovery serve job $JOB"; scancel $JOB
