#!/bin/bash
# swebench_gate.sh <task_dir> <mode>    mode = nop | oracle
#
# Model-free environment gate for ONE SWE-bench Verified task.
#
# Unlike r2egym (where the gold fix is `git checkout <base_commit> -- .` because
# /testbed sits at base_commit^), the swebench adapter ships the gold patch as
# solution/solve.sh -- `patch --fuzz=5 -p1` of the dataset `patch` field. So the
# two arms are simply:
#   nop    -> run tests/test.sh only.                    Expect reward 0.
#   oracle -> run solution/solve.sh, then tests/test.sh. Expect reward 1.
#
# Both arms on the same task producing DIFFERENT rewards is the gate: it proves
# the verifier measures code state rather than returning a per-task constant.
# (A per-task constant reward is the exact historical bug this project hit with
# r2egym, where /testbed was never populated.)
#
# Mirrors harbor's apptainer worker: instance start + exec instance://, writable
# ext3 overlay, /tests + /logs/verifier binds, $BRIDGE_AGENT_TOOLS bound :ro,
# and test.sh rewritten by harbor's OWN _patch_test_sh_for_offline_pip() --
# extracted from the live worker.py, never reimplemented. That patch is
# load-bearing here: swebench's test.sh runs `uv run parser.py` with PEP-723
# deps (swebench==4.0.3, datasets==2.16.1, fastcore<1.11) and monkey-patches
# make_test_spec's GitHub fetch, none of which can work on an offline node.
set -uo pipefail

TASK="${1:?usage: swebench_gate.sh <task_name> <nop|oracle>}"
MODE="${2:-nop}"
ROOT=/p/scratch/synthlaion/lee27
export APPTAINER_CACHEDIR=$ROOT/apptainer_cache
export APPTAINER_TMPDIR=$ROOT/apptainer_tmp
TASKS=${TASKS:-$ROOT/tasks/swebench-verified}
SIF_CACHE=${SIF_CACHE:-$ROOT/swebench_sif}
STAGE=$TASKS/$TASK

export BRIDGE_AGENT_TOOLS=${BRIDGE_AGENT_TOOLS:-$ROOT/agent_tools}
# Extract the offline patch from the worker the LIVE FLEET actually loaded.
# Verified 2026-08-06: job 15500584 runs
#   /p/project1/synthlaion/lee27/harbor/src/harbor/environments/apptainer/jureca_workers.sbatch
# i.e. a real checkout (branch lukedhlee/apptainer-opencode-bridge, d64180d), NOT the pinned
# apptainer_bridge/<sha>/ copies. The pinned 9c31e931 copy is OLDER and lacks the astropy
# DeprecationWarning shim and the /etc/gitconfig safe.directory write, so extracting from it
# would measure a patch the fleet does not use.
HARBOR_LIVE=${HARBOR_LIVE:-/p/project1/synthlaion/lee27/harbor}
WORKER=${WORKER:-$HARBOR_LIVE/src/harbor/environments/apptainer/worker.py}

emit_err() { echo "{\"task\":\"$TASK\",\"mode\":\"$MODE\",\"error\":\"$1\"}"; exit 0; }

[[ -d "$STAGE" ]] || emit_err not_staged
[[ -f "$STAGE/tests/test.sh" ]] || emit_err no_test_sh
[[ -f "$STAGE/solution/solve.sh" ]] || emit_err no_solve_sh

# SIF name is build_<task>-<sha256(Dockerfile)[:12]>.sif -- the convention
# prebuild_sifs.sh and worker.py both use.
DF=$STAGE/environment/Dockerfile
HASH=$(sha256sum "$DF" | cut -c1-12)
SIF=$SIF_CACHE/build_${TASK}-${HASH}.sif
[[ -f "$SIF" ]] || SIF=$(ls $SIF_CACHE/build_${TASK}-*.sif 2>/dev/null | head -1)
[[ -n "$SIF" && -f "$SIF" ]] || emit_err no_sif

W=$(mktemp -d $ROOT/sg.XXXXXXXX)
INST="sg_$(echo "${TASK}_${MODE}_$$" | tr -cd 'a-zA-Z0-9_')"
mkdir -p $W/tests $W/logs/verifier $W/sol
# Copy the WHOLE tests dir, as harbor uploads every test source dir to /tests
# (verifier.py:182). Cherry-picking files silently drops anything else the task
# ships.
cp -a $STAGE/tests/. $W/tests/
cp $STAGE/solution/solve.sh $W/sol/solve.sh
chmod +x $W/tests/test.sh $W/sol/solve.sh

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
print("PATCH_OK" if "BRIDGE OFFLINE-PIP PATCH" in open(
    os.path.join(target, "test.sh")).read() else "PATCH_MISSING")
PY
)
case "$PATCHED" in
  *PATCH_OK*) PATCH_STATUS=ok ;;
  *) PATCH_STATUS="FAILED: $(echo "$PATCHED" | tr -d '"' | tail -1)" ;;
esac

apptainer overlay create --size 4096 $W/ov.img >/dev/null 2>&1 \
  || emit_err overlay_create_failed
cleanup() { apptainer instance stop "$INST" >/dev/null 2>&1; rm -rf "$W"; }
trap cleanup EXIT

HPATH="$BRIDGE_AGENT_TOOLS/bin:$BRIDGE_AGENT_TOOLS/uv_env/.venv/bin:/root/.local/bin:/testbed/.venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
TOOLBINDS=()
for t in uv tmux asciinema rg opencode; do
  [[ -x "$BRIDGE_AGENT_TOOLS/bin/$t" ]] && TOOLBINDS+=(--bind "$BRIDGE_AGENT_TOOLS/bin/$t:/usr/local/bin/$t:ro")
done
mkdir -p $W/tmp $W/logs/agent $W/logs/artifacts $W/setup_files $W/ws

# --cleanenv (worker.py:559 at start, :1131 on EVERY exec -- the start-time flag
# does NOT carry over) keeps Slurm/PMI launcher state out of the container;
# harbor notes old Python runtimes can fail to spawn subprocesses with it
# present. --no-home stops the host $HOME shadowing the image's /root + conda.
# NB: /testbed was measured drwxrwxrwx and owned by our own uid inside these
# SIFs, so harbor's writable-workdir bind is NOT needed here.
apptainer instance start --cleanenv --overlay $W/ov.img \
  --no-home \
  --bind $W/tests:/tests:rw \
  --bind $W/logs/verifier:/logs/verifier:rw \
  --bind $W/logs/agent:/logs/agent:rw \
  --bind $W/logs/artifacts:/logs/artifacts:rw \
  --bind $W/setup_files:/setup_files:rw \
  --bind $W/tmp:/tmp:rw \
  --bind $W/ws:/workspace:rw \
  --bind $W/sol:/solution:ro \
  --bind $BRIDGE_AGENT_TOOLS:$BRIDGE_AGENT_TOOLS:ro \
  "${TOOLBINDS[@]}" \
  --env "PATH=$HPATH" --env LANG=C.UTF-8 --env LC_ALL=C.UTF-8 \
  "$SIF" "$INST" >$W/start.log 2>&1 \
  || emit_err "instance_start_failed"

PRE="true"
# harbor's oracle agent runs (/solution/solve.sh) directly via its shebang
# (agents/oracle.py:88-102), not `bash <path>`.
[[ "$MODE" == "oracle" ]] && PRE="chmod +x /solution/solve.sh 2>/dev/null; (/solution/solve.sh)"

# Bound it ourselves so an srun wall-clock kill can never leave us with no
# record; on timeout the tail still shows where it stalled. HOME is exported
# inside the exec because Apptainer rejects HOME in --env for instance execs
# on JURECA (worker.py:1076).
timeout -k 10 "${SG_TIMEOUT:-${BRIDGE_EXEC_TIMEOUT:-2100}}" \
  apptainer exec --cleanenv \
  --env "PATH=$HPATH" --env LANG=C.UTF-8 --env LC_ALL=C.UTF-8 \
  --pwd /testbed \
  instance://$INST /usr/bin/bash -lc \
  "export HOME=/root XDG_DATA_HOME=/root/.local/share XDG_CACHE_HOME=/root/.cache XDG_CONFIG_HOME=/root/.config; \
   export PATH=$HPATH; cd /testbed && { $PRE; } && chmod +x /tests/test.sh 2>/dev/null; (/tests/test.sh) 2>&1" \
  >$W/out.txt 2>&1
RC=$?
TIMED_OUT=false
[[ $RC -eq 124 || $RC -eq 137 ]] && TIMED_OUT=true

REWARD=$(cat $W/logs/verifier/reward.txt 2>/dev/null | tr -d '\n\r ')
RESOLVED=$(grep -oE "^(PASSED|FAILED)$" $W/out.txt 2>/dev/null | tail -1)
TAIL=$(tail -c 700 $W/out.txt 2>/dev/null)

python3 - "$TASK" "$MODE" "${REWARD:-null}" "$RC" "$PATCH_STATUS" "${RESOLVED:-}" "${TAIL:-}" "$TIMED_OUT" <<'PY'
import json, sys
task, mode, reward, rc, patch, resolved, tail, timed_out = sys.argv[1:9]
try: r = float(reward)
except Exception: r = None
print(json.dumps({"task": task, "mode": mode, "reward": r, "raw_reward": reward,
                  "exit": int(rc), "offline_patch": patch, "resolved": resolved,
                  "timed_out": timed_out == "true", "tail": tail[-700:]}))
PY
