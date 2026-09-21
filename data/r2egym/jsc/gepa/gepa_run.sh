#!/bin/bash
# gepa_run.sh <run-name> <endpoint-url> <tree-dir> <task-list-file> [conc] [attempts]
# One `harbor run` for ONE shard of ONE GEPA candidate, against one server of the standing serve job, on Daytona.
# Adapted by diff from data/tb2/jupiter/run_tb2.sh: same preflight (the server answers with the served name and the
# think markers intact, the Daytona key is accepted), same policy rendering, same tmux + log convention, same refusal
# to start a run name twice.
#
# Deltas: the task list is an EXPLICIT file of dir names inside the wave tree (one candidate's dirs, one shard of
# them) instead of SHARD=i/n over a whole benchmark tree, because a GEPA tree holds every candidate's copies side by
# side and a run must see only its own; and HARBOR_SRC defaults to the setup-files-hook clone, which the R2E-Gym
# Daytona tasks need (each task dir carries setup_files/setup.sh).
#
# Called by gepa_runner.py; runnable by hand for one shard. DRY=1 renders and validates the policy, starts nothing.
set -uo pipefail
NAME=${1:?run name}; URL=${2:?endpoint url}; TREE=${3:?wave tree dir}; LIST=${4:?file of task dir names}
CONC=${5:-32}; ATTEMPTS=${6:-1}
C=/e/project1/transfernetx/lee27/code
G=${GEPA_CODE:-$C/snowball/gepa}
E=${GEPA_DIR:-/e/fscratch/reformo/lee27/experiments/gepa}
JOBS=${GEPA_JOBS:-/e/data1/mmlaion/lee27/experiments/gepa_jobs}
HARBOR_SRC=${HARBOR_SRC:-$C/harbor-hook/src}   # the setup-files hook; R2E-Gym daytona tasks run setup_files/setup.sh
PY=$C/envs/snowball-v2/bin/python; HARBOR=$C/envs/snowball-v2/bin/harbor
KEYF=${DAYTONA_KEYF:-/e/fscratch/reformo/lee27/keys/daytona_eval.env}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1   # login-node pid cap
mkdir -p "$E/runs" "$E/logs" "$JOBS"
[ -d "$TREE" ] || { echo "no wave tree at $TREE"; exit 1; }
[ -f "$LIST" ] || { echo "no task list at $LIST"; exit 1; }
[ -f "$HARBOR_SRC/harbor/trial/trial.py" ] || { echo "no harbor at $HARBOR_SRC"; exit 1; }
grep -q "_run_setup_script" "$HARBOR_SRC/harbor/trial/trial.py" || { echo "$HARBOR_SRC has no setup_files hook"; exit 1; }
[ -d "$JOBS/$NAME" ] && { echo "$JOBS/$NAME exists; pick a new name or resume (harbor jobs resume -p $JOBS/$NAME)"; exit 1; }

if [ "${DRY:-0}" != 1 ]; then
  # 1. the server answers from here, with the served name and the think markers intact
  curl -sf --max-time 20 "$URL/models" | grep -q '"snowball"' || { echo "no model 'snowball' at $URL"; exit 1; }
  curl -sf --max-time 300 "$URL/chat/completions" -H 'Content-Type: application/json' \
    -d '{"model":"snowball","messages":[{"role":"user","content":"Print hello in bash."}],"max_tokens":300,"skip_special_tokens":false}' \
    | $PY -c 'import json,sys; r=json.load(sys.stdin); print("server ok:", r["usage"], repr((r["choices"][0]["message"].get("content") or "")[:120]))' \
    || { echo "server at $URL did not complete a request"; exit 1; }

  # 2. the sandbox backend: Daytona, eval org key
  set -a; source "$KEYF"; set +a
  curl -sf --max-time 20 -H "Authorization: Bearer $DAYTONA_API_KEY" https://app.daytona.io/api/api-keys/current >/dev/null \
    || { echo "Daytona key rejected"; exit 1; }
else
  echo "DRY: skipping the server and Daytona preflight"
  [ -f "$KEYF" ] || { echo "no Daytona key file at $KEYF"; exit 1; }
fi

# 3. render the policy for this shard
CFG=$E/runs/$NAME.yaml
sed "s#__JOB_NAME__#$NAME#; s#__API_BASE__#$URL#; s#__CONC__#$CONC#; s#__ATTEMPTS__#$ATTEMPTS#" "$G/gepa_policy.yaml" > "$CFG"
$PY - "$CFG" "$TREE" "$LIST" <<'PY' || exit 1
import os, sys, yaml
p, tree, lst = sys.argv[1:4]
c = yaml.safe_load(open(p))
names = [l.strip() for l in open(lst) if l.strip() and not l.startswith("#")]
missing = [n for n in names if not os.path.isfile(os.path.join(tree, n, "task.toml"))]
if missing:
    sys.exit("%d task dirs missing from %s, e.g. %s" % (len(missing), tree, missing[:3]))
c.pop("datasets", None)
c["tasks"] = [{"path": os.path.join(tree, n)} for n in names]
yaml.safe_dump(c, open(p, "w"), sort_keys=False)
print("%d tasks, %s attempts, conc %s" % (len(names), c["n_attempts"], c["n_concurrent_trials"]))
PY
$PY -c "import yaml,sys; sys.path.insert(0,'$HARBOR_SRC'); from harbor_config.models.job.config import JobConfig; JobConfig.model_validate(yaml.safe_load(open('$CFG'))); print('policy validates')" || exit 1
if [ "${DRY:-0}" = 1 ]; then echo "DRY -- rendered and validated $CFG, started nothing"; exit 0; fi

# 4. run in tmux; the runner polls the log for RUN_DONE
LOG=$E/logs/run_${NAME}.log
tmux new -d -s gepa_$NAME "export OMP_NUM_THREADS=1 PYTHONPATH=$HARBOR_SRC; set -a; source $KEYF; set +a; cd $E; $HARBOR run -c $CFG -y 2>&1 | tee $LOG; echo RUN_DONE exit=\${PIPESTATUS[0]} >> $LOG"
echo "started tmux gepa_$NAME on $(hostname); log $LOG; results $JOBS/$NAME"
