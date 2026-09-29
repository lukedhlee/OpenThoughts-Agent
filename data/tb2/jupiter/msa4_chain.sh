#!/bin/bash
# msa4_chain.sh <serve-jobid> <run-name> <smoke|full> [n_attempts]
# Terminal-Bench 2 with mini-swe-agent-host in TOOL mode on one Snowball server (serve_snowball.sbatch), from a Jupiter login
# node: wait for the serve endpoint, check the server and the Daytona key, render tb2_msa_toolmode_0924.yaml, run harbor in
# the foreground of THIS tmux session, summarize, then scancel the serve job. Run it in tmux msa4_<...> (never tb2_<name>:
# that is run_tb2.sh's session name). Never reuse a run name.
#   smoke: the tasks in SMOKE_TASKS (comma list), n_attempts 1.
#   full : every TB2 task except EXCLUDE_TASKS (default train-fasttext: its Daytona image is gone and the org's snapshot
#          slots are full, 2026-09-28; counted as an infrastructure loss), optionally SHARD=i/n (every n-th task from i).
# Harbor: a clean clone of marin-community/harbor at HARBOR_SHA (default 6feb3759 = lukedhlee/mini-swe-host-exec-isolation,
# cdd5b008 + the exec-isolation fix; the first smoke ran on cdd5b008) in
# HARBOR_DIR, with mini-swe-agent 2.4.6 from the --no-deps side dir envs/msa-2.4.6; interpreter snowball-v2's.
set -o pipefail
JOB=${1:?serve job id}; NAME=${2:?run name}; MODE=${3:?smoke|full}; NATT=${4:-1}
C=/e/project1/transfernetx/lee27/code; W=$C/tb2; E=/e/fscratch/reformo/lee27/experiments/tb2
JOBS=/e/data1/mmlaion/lee27/experiments/tb2_jobs; KEYF=/e/fscratch/reformo/lee27/keys/daytona_eval.env
HARBOR_DIR=${HARBOR_DIR:-$C/harbor-msa4}; HARBOR_SRC=$HARBOR_DIR/src; HARBOR_SHA=${HARBOR_SHA:-6feb3759}
MSA=$C/envs/msa-2.4.6; PY=$C/envs/snowball-v2/bin/python; HARBOR=$C/envs/snowball-v2/bin/harbor
TASKS=${TASKS:-/e/fscratch/reformo/lee27/tasks/terminal_bench_2}; NTASKS=${NTASKS:-89}   # TBLite: TASKS=<tasks>/openthoughts_tblite_2_0 NTASKS=100
EXPECT_MODEL=${EXPECT_MODEL:-grug-datakit-sft-20260921}   # the checkpoint the serve job must hold (a substring of its model path)
POLICY_FILE=${POLICY_FILE:-$W/msa4/tb2_msa_toolmode_0924.yaml}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1   # login-node pid cap
export PYTHONPATH=$HARBOR_SRC:$MSA
export EXCLUDE_TASKS=${EXCLUDE_TASKS-train-fasttext}   # a TB2 name; harmless on other trees
mkdir -p $E/logs $W/msa4/runs
LOG=$E/logs/msa4_$NAME.log
exec > >(tee -a $LOG) 2>&1
log() { echo "$(date -Is) $*"; }
release() { [ "${RELEASE:-1}" = 0 ] && { log "keeping serve job $JOB (RELEASE=0)"; return; }; log "releasing serve job $JOB"; scancel $JOB; }   # RELEASE=0: the node serves another run next
log "msa4_chain $NAME mode=$MODE n_attempts=$NATT serve=$JOB harbor=$HARBOR_DIR policy=$POLICY_FILE"
[ "$(git -C $HARBOR_DIR rev-parse --short=8 HEAD)" = "${HARBOR_SHA:0:8}" ] || { log "FAILED: $HARBOR_DIR is not at $HARBOR_SHA"; release; exit 1; }
[ -z "$(git -C $HARBOR_DIR status --porcelain --untracked-files=no)" ] || { log "FAILED: $HARBOR_DIR has local changes"; release; exit 1; }
[ "$(ls -d $TASKS/*/ | wc -l)" = "$NTASKS" ] || { log "FAILED: $TASKS does not have $NTASKS tasks"; release; exit 1; }
[ -d $JOBS/$NAME ] && { log "FAILED: $JOBS/$NAME exists; pick a new name or resume"; exit 1; }
# 1. endpoint: queue time is free; once RUNNING the server gets 40 min to write its endpoint
RUNNING_SINCE=""
while [ ! -f $E/endpoints/$JOB ]; do
  STATE=$(squeue -h -j $JOB -o %T 2>/dev/null)
  [ -z "$STATE" ] && { log "FAILED: serve job $JOB left the queue before writing an endpoint"; exit 1; }
  if [ "$STATE" = RUNNING ]; then
    RUNNING_SINCE=${RUNNING_SINCE:-$(date +%s)}
    [ $(( $(date +%s) - RUNNING_SINCE )) -gt 2400 ] && { log "FAILED: no endpoint 40 min after start"; release; exit 1; }
  fi
  sleep 30
done
URL=$(cat $E/endpoints/$JOB); log "endpoint $URL"; cat $E/endpoints/$JOB.meta
grep -q "$EXPECT_MODEL" $E/endpoints/$JOB.meta && grep -q '"temperature":1.0' $E/endpoints/$JOB.meta \
  || { log "FAILED: serve job $JOB is not $EXPECT_MODEL at temperature 1.0"; release; exit 1; }
for i in $(seq 1 30); do curl -sf --max-time 20 $URL/models | grep -q '"snowball"' && break; sleep 10; done
curl -sf --max-time 20 $URL/models | grep -q '"snowball"' || { log "FAILED: no model snowball at $URL"; release; exit 1; }
# the bash tool renders and a reply comes back through the tool_choice "none" path the agent uses
curl -sf --max-time 300 $URL/chat/completions -H 'Content-Type: application/json' -d '{"model":"snowball","messages":[{"role":"user","content":"List the files in /tmp."}],"tools":[{"type":"function","function":{"name":"bash","description":"Execute a bash command","parameters":{"type":"object","properties":{"command":{"type":"string"}},"required":["command"]}}}],"tool_choice":"none","max_tokens":600,"skip_special_tokens":false}' \
  | $PY -c 'import json,sys; r=json.load(sys.stdin); print("server ok:", r["usage"], repr((r["choices"][0]["message"].get("content") or "")[-300:]))' \
  || { log "FAILED: tool-mode probe request"; release; exit 1; }
set -a; source $KEYF; set +a
curl -sf --max-time 20 -H "Authorization: Bearer $DAYTONA_API_KEY" https://app.daytona.io/api/api-keys/current >/dev/null || { log "FAILED: Daytona key rejected"; release; exit 1; }
# 2. render the policy and the task list
CFG=$W/msa4/runs/$NAME.yaml
sed "s#__JOB_NAME__#$NAME#; s#__API_BASE__#$URL#" $POLICY_FILE > $CFG
MODE=$MODE NATT=$NATT $PY - "$CFG" "$TASKS" <<'PY'
import os, sys, yaml
p, tasks = sys.argv[1:3]; c = yaml.safe_load(open(p))
names = sorted(d for d in os.listdir(tasks) if os.path.isfile(f"{tasks}/{d}/task.toml"))
if os.environ["MODE"] == "smoke":
    want = [t for t in os.environ["SMOKE_TASKS"].split(",") if t]
    missing = [t for t in want if t not in names]
    assert not missing, f"smoke tasks not in the tree: {missing}"
    names = want
else:
    skip = [t for t in os.environ.get("EXCLUDE_TASKS", "").split(",") if t]
    names = [t for t in names if t not in skip]
    shard = os.environ.get("SHARD")
    if shard:
        i, n = map(int, shard.split("/")); names = names[i::n]
    print(f"excluded {skip}; shard {shard}")
c["n_attempts"] = int(os.environ["NATT"]); c.pop("datasets", None)
c["tasks"] = [{"path": f"{tasks}/{t}"} for t in names]
yaml.safe_dump(c, open(p, "w"), sort_keys=False)
print(f"{len(names)} tasks x {c['n_attempts']} attempts, {c['n_concurrent_trials']} concurrent: {' '.join(names) if len(names) <= 12 else names[0] + ' ... ' + names[-1]}")
PY
[ $? = 0 ] || { log "FAILED: rendering $CFG"; release; exit 1; }
$PY -c "import yaml; from harbor_config.models.job.config import JobConfig; JobConfig.model_validate(yaml.safe_load(open('$CFG'))); import harbor.agents.mini_swe_agent_host.agent as a, minisweagent; print('policy validates; harbor', a.__file__, 'mini-swe-agent', minisweagent.__version__)" \
  || { log "FAILED: config/import check"; release; exit 1; }
# 3. run (foreground, this session)
log "harbor start $JOBS/$NAME"
cd $W/msa4 && $HARBOR run -c $CFG -y
log "HARBOR_EXIT $?"
release
$PY $W/summarize_tb2.py $JOBS/$NAME --json > $E/final_msa4_$NAME.json; $PY $W/summarize_tb2.py $JOBS/$NAME | tee $E/final_msa4_$NAME.txt
touch $E/final_msa4_$NAME.DONE
log "RUN_DONE"
