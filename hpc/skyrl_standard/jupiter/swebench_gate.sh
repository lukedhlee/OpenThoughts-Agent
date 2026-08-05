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
WORKER=${WORKER:-$ROOT/apptainer_bridge/9c31e931/worker.py}

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
cp $STAGE/tests/test.sh $W/tests/test.sh
cp $STAGE/tests/config.json $W/tests/config.json
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

apptainer overlay create --size 4096 $W/ov.img >/dev/null 2>&1
cleanup() { apptainer instance stop "$INST" >/dev/null 2>&1; rm -rf "$W"; }
trap cleanup EXIT

apptainer instance start --overlay $W/ov.img \
  --bind $W/tests:/tests:rw \
  --bind $W/logs/verifier:/logs/verifier:rw \
  --bind $W/sol:/sol:ro \
  --bind $BRIDGE_AGENT_TOOLS:$BRIDGE_AGENT_TOOLS:ro \
  "$SIF" "$INST" >$W/start.log 2>&1 \
  || emit_err "instance_start_failed"

PRE="true"
[[ "$MODE" == "oracle" ]] && PRE="bash /sol/solve.sh"

apptainer exec instance://$INST bash -c \
  "cd /testbed && $PRE && bash /tests/test.sh" >$W/out.txt 2>&1
RC=$?

REWARD=$(cat $W/logs/verifier/reward.txt 2>/dev/null | tr -d '\n\r ')
RESOLVED=$(grep -oE "^(PASSED|FAILED)$" $W/out.txt 2>/dev/null | tail -1)
TAIL=$(tail -c 700 $W/out.txt 2>/dev/null)

python3 - "$TASK" "$MODE" "${REWARD:-null}" "$RC" "$PATCH_STATUS" "${RESOLVED:-}" "${TAIL:-}" <<'PY'
import json, sys
task, mode, reward, rc, patch, resolved, tail = sys.argv[1:8]
try: r = float(reward)
except Exception: r = None
print(json.dumps({"task": task, "mode": mode, "reward": r, "raw_reward": reward,
                  "exit": int(rc), "offline_patch": patch, "resolved": resolved,
                  "tail": tail[-700:]}))
PY
