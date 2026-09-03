#!/bin/bash
# swesmith_gate.sh <task_dir> <pristine|oracle>
#
# Model-free environment gate for one raw SWE-smith task (data/swesmith/build_raw.py output).
# Mirrors harbor's apptainer worker "base-image SIF" path (worker.py) step by step:
#   swesmith_base_<sha256(FROM image)[:12]>.sif, writable ext3 overlay, --no-home --home /tmp/fakehome,
#   host dir pre-copied from the SIF's /testbed and bound over /testbed (WORKDIR bind), per-instance
#   /tmp, /tests + /logs/* + /workspace + /setup_files binds, agent_tools bind + tool binds,
#   /etc/gitconfig insteadOf rewrite to the offline mirrors, the worker's own offline-pip patch on
#   test.sh (sliced out of the live worker.py), then the Dockerfile RUN lines replayed with
#   `apptainer exec --pwd /testbed instance://X bash -c <cmd>` exactly as the worker does.
#
#   pristine -> nothing else touches /testbed. Expect 0.0.
#   oracle   -> solution/solve.sh (reverse-apply the bug patch). Expect 1.0.
# pristine 0 AND oracle 1 on the same task == the environment measures code state.
#
# Also reports the things a first look must establish: origin URL after the rewrite, HEAD branch
# after the RUN replay, which python an interactive shell sees (the .bashrc conda activation),
# and the tail of the verifier output.
set -uo pipefail

TASK_DIR="${1:?usage: swesmith_gate.sh <task_dir> <pristine|oracle>}"
MODE="${2:-pristine}"
TASK=$(basename "$TASK_DIR")
ROOT=/p/scratch/synthlaion/lee27
export APPTAINER_CACHEDIR=$ROOT/apptainer_cache
export APPTAINER_TMPDIR=$ROOT/apptainer_tmp
SIF_CACHE=${SIF_CACHE:-$ROOT/swesmith_sif}
MIRRORS=${BRIDGE_GIT_MIRRORS:-$ROOT/git_mirrors_swesmith}
export BRIDGE_AGENT_TOOLS=${BRIDGE_AGENT_TOOLS:-$ROOT/agent_tools}
HARBOR_LIVE=${HARBOR_LIVE:-/p/project1/synthlaion/lee27/harbor}
WORKER=${WORKER:-$HARBOR_LIVE/src/harbor/environments/apptainer/worker.py}
RUNS=${SWG_RUNS:-$ROOT/swesmith_gate_runs}
mkdir -p "$RUNS"

DF="$TASK_DIR/environment/Dockerfile"
[[ -f "$DF" && -f "$TASK_DIR/tests/test.sh" && -f "$TASK_DIR/tests/config.json" ]] \
  || { echo "{\"task\":\"$TASK\",\"mode\":\"$MODE\",\"error\":\"not_staged\"}"; exit 0; }
IMG=$(grep -i '^FROM ' "$DF" | head -1 | awk '{print $2}')
H=$(printf '%s' "$IMG" | sha256sum | cut -c1-12)
SIF=$SIF_CACHE/swesmith_base_$H.sif
[[ -f "$SIF" ]] || { echo "{\"task\":\"$TASK\",\"mode\":\"$MODE\",\"error\":\"no_sif\",\"sif\":\"$SIF\"}"; exit 0; }
ID=$(python3 -c "import json;print(json.load(open('$TASK_DIR/tests/config.json'))['instance_id'])")

W=$(mktemp -d "$RUNS/g.XXXXXXXX")
INST="sg_$(echo "${TASK}_${MODE}_$$" | tr -cd 'a-zA-Z0-9_')"
mkdir -p $W/tests $W/logs/verifier $W/logs/agent $W/logs/artifacts $W/ws $W/tmp $W/setup_files $W/workdir
chmod 777 $W/workdir
cp "$TASK_DIR"/tests/* $W/tests/
# The whole environment dir lands in /workspace (harbor sends it as files_b64), not just the Dockerfile.
cp -r "$TASK_DIR"/environment/. $W/ws/
[[ -f "$TASK_DIR/solution/solve.sh" ]] && cp "$TASK_DIR/solution/solve.sh" $W/tmp/solve.sh

# harbor's OWN offline-pip patch, sliced from the live worker.py (no reimplementation to drift).
PATCHED=$(python3 - "$WORKER" "$W/tests" <<'PY' 2>&1
import os, re, shlex, sys
worker, target = sys.argv[1], sys.argv[2]
src = open(worker).read()
i = src.index("def _patch_test_sh_for_offline_pip")
m = re.search(r"^(?:def |class )", src[i + 10:], re.M)
body = src[i:i + 10 + m.start()] if m else src[i:]
g = {"os": os, "shlex": shlex, "print": print}
exec(body, g)
g["_patch_test_sh_for_offline_pip"](target)
print("PATCH_OK" if "BRIDGE OFFLINE-PIP PATCH" in open(os.path.join(target, "test.sh")).read() else "PATCH_MISSING")
PY
)
case "$PATCHED" in *PATCH_OK*) PATCH_STATUS=ok ;; *) PATCH_STATUS="FAILED: $(echo "$PATCHED" | tail -1 | tr -d '"')" ;; esac

# /etc/gitconfig rewrite, byte-for-byte what worker.py writes.
printf '[url "/git_mirrors/"]\n\tinsteadOf = https://github.com/\n\tinsteadOf = http://github.com/\n\tinsteadOf = git@github.com:\n[safe]\n\tdirectory = *\n' > $W/gitconfig

apptainer overlay create --size 4096 $W/ov.img >/dev/null 2>&1 \
  || { echo "{\"task\":\"$TASK\",\"mode\":\"$MODE\",\"error\":\"overlay_create_failed\"}"; exit 0; }

# WORKDIR bind: pre-copy the SIF's /testbed into a host dir (worker.py "Bind a writable per-instance dir").
apptainer exec --bind $W/workdir:/_workdir_init:rw "$SIF" sh -c \
  'if [ -d /testbed ] && [ -n "$(ls -A /testbed 2>/dev/null)" ]; then cp -a /testbed/. /_workdir_init/ 2>/dev/null || true; fi' \
  >$W/precopy.log 2>&1
PRECOPY_N=$(ls -A $W/workdir | wc -l)

# PATH the worker forces for base-image SIF tasks (_base_sif_command_prefix + start-time --env PATH).
AT=$BRIDGE_AGENT_TOOLS
HPATH="$AT/bin:$AT/uv_env/.venv/bin:/root/.local/bin:/testbed/.venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
TOOLBINDS=()
for t in uv tmux asciinema rg opencode; do
  [[ -x "$AT/bin/$t" ]] && TOOLBINDS+=(--bind "$AT/bin/$t:/usr/local/bin/$t:ro")
done
LIBENV=()
[[ -d "$AT/lib" ]] && LIBENV+=(--env "LD_LIBRARY_PATH=$AT/lib")

cleanup() { apptainer instance stop "$INST" >/dev/null 2>&1; [[ "${SWG_KEEP:-0}" = 1 ]] || rm -rf "$W"; }
trap cleanup EXIT

apptainer instance start --cleanenv --overlay $W/ov.img \
  --no-home --home /tmp/fakehome \
  --bind $W/ws:/workspace:rw \
  --bind $W/logs/verifier:/logs/verifier:rw \
  --bind $W/logs/agent:/logs/agent:rw \
  --bind $W/logs/artifacts:/logs/artifacts:rw \
  --bind $W/tests:/tests:rw \
  --bind $W/setup_files:/setup_files:rw \
  --bind $W/workdir:/testbed:rw \
  --bind $W/tmp:/tmp:rw \
  --bind $AT:$AT:ro \
  "${TOOLBINDS[@]}" \
  --bind $MIRRORS:/git_mirrors:ro \
  --bind $W/gitconfig:/etc/gitconfig:ro \
  "${LIBENV[@]}" \
  --env "PATH=$HPATH" --env 'PS1=\$ ' --env PROMPT_COMMAND= --env LANG=C.UTF-8 --env LC_ALL=C.UTF-8 \
  "$SIF" "$INST" >$W/start.log 2>&1
if [[ $? -ne 0 ]]; then
  echo "{\"task\":\"$TASK\",\"mode\":\"$MODE\",\"error\":\"instance_start_failed\",\"detail\":\"$(tail -1 $W/start.log | tr -d '"\\')\"}"
  exit 0
fi

X() { apptainer exec --cleanenv --env "PATH=$HPATH" --pwd /testbed "instance://$INST" bash -c "$1" 2>&1; }

ORIGIN0=$(X 'git remote get-url origin')
HEAD0=$(X 'git rev-parse --abbrev-ref HEAD')

# Replay the Dockerfile RUN lines exactly as worker._extract_run_commands + the exec loop do.
RUN_RC=""
while IFS= read -r line; do
  s=$(echo "$line" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
  [[ -z "$s" || "$s" == \#* ]] && continue
  [[ "${s^^}" == RUN\ * ]] || continue
  cmd=${s:4}
  skip=0; for bc in "apt-get" "curl -LsSf" "mkdir -p /output" "mkdir -p /logs"; do [[ "$cmd" == "$bc"* ]] && skip=1; done
  [[ $skip -eq 1 ]] && continue
  timeout -k 5 300 apptainer exec --pwd /testbed "instance://$INST" bash -c "$cmd" >$W/run.log 2>&1
  rc=$?; RUN_RC="${RUN_RC}${rc},"
  [[ $rc -ne 0 ]] && { echo "RUN rc=$rc: $cmd" >> $W/run_fail.log; tail -3 $W/run.log >> $W/run_fail.log; }
done < "$DF"

HEAD1=$(X 'git rev-parse --abbrev-ref HEAD')
PYWHICH=$(apptainer exec --cleanenv --env "PATH=$HPATH" --pwd /testbed "instance://$INST" bash -c \
  'export HOME=/tmp/fakehome; bash -ic "which python; python --version" 2>&1 | tail -2 | tr "\n" " "')
PYTEST=$(X 'ls /opt/miniconda3/envs/testbed/bin/pytest 2>&1 | tail -1')

if [[ "$MODE" == shell ]]; then
  # Leave the instance running for interactive use (swesmith_shell.sh exec/verify/stop).
  trap - EXIT
  echo "{\"task\":\"$TASK\",\"mode\":\"shell\",\"instance_id\":\"$ID\",\"inst\":\"$INST\",\"w\":\"$W\",\"node\":\"$(hostname -s)\",\"head_after_run\":\"$HEAD1\",\"run_rcs\":\"$RUN_RC\",\"interactive_python\":\"$PYWHICH\",\"hpath\":\"$HPATH\"}"
  exit 0
fi

if [[ "$MODE" == oracle ]]; then
  ORACLE=$(apptainer exec --cleanenv --env "PATH=$HPATH" --pwd /testbed "instance://$INST" /usr/bin/bash -lc \
    'export HOME=/tmp/fakehome; chmod +x /tmp/solve.sh; /tmp/solve.sh 2>&1 | tail -3 | tr "\n" " "; echo rc=${PIPESTATUS[0]}')
else
  ORACLE=""
fi

timeout -k 10 "${SWG_TIMEOUT:-2100}" \
  apptainer exec --cleanenv --env "PATH=$HPATH" --env LANG=C.UTF-8 --env LC_ALL=C.UTF-8 --pwd /testbed \
  "instance://$INST" /usr/bin/bash -lc \
  "export HOME=/tmp/fakehome XDG_DATA_HOME=/tmp/fakehome/.local/share XDG_CACHE_HOME=/tmp/fakehome/.cache XDG_CONFIG_HOME=/tmp/fakehome/.config; \
   export PATH=$HPATH; chmod +x /tests/test.sh 2>/dev/null; (/tests/test.sh) 2>&1" >$W/out.txt 2>&1
RC=$?
REWARD=$(cat $W/logs/verifier/reward.txt 2>/dev/null | tr -d '\n\r ')
GRADE=$(grep -E '^grade:' $W/out.txt | head -2 | tr '\n' ' ')
PYSUM=$(grep -oE '[0-9]+ (passed|failed|error)[^=]*' $W/logs/test_output.log 2>/dev/null | tail -1)
TAIL=$(tail -c 500 $W/out.txt 2>/dev/null)
RUNFAIL=$(cat $W/run_fail.log 2>/dev/null | tail -c 300)

python3 - "$TASK" "$MODE" "$ID" "${REWARD:-null}" "$RC" "$ORIGIN0" "$HEAD0" "$HEAD1" "$PYWHICH" "$PYTEST" "$RUN_RC" "$PRECOPY_N" "$PATCH_STATUS" "$ORACLE" "$GRADE" "$PYSUM" "$RUNFAIL" "$TAIL" <<'PY'
import json, sys
(task, mode, iid, reward, rc, origin, head0, head1, pywhich, pytest_bin, run_rc, precopy, patch, oracle, grade, pysum, runfail, tail) = sys.argv[1:19]
try: r = float(reward)
except Exception: r = None
print(json.dumps({"task": task, "mode": mode, "instance_id": iid, "reward": r, "exit": int(rc),
  "origin": origin.strip(), "head_before_run": head0.strip(), "head_after_run": head1.strip(),
  "checkout_ok": head1.strip() == iid, "interactive_python": pywhich.strip(), "pytest_bin": pytest_bin.strip(),
  "run_rcs": run_rc, "testbed_precopy_entries": int(precopy), "offline_patch": patch, "oracle": oracle.strip(),
  "grade": grade.strip(), "pytest_summary": pysum.strip(), "run_fail": runfail.strip(), "tail": tail[-500:]}))
PY
