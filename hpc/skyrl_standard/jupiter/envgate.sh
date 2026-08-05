#!/bin/bash
# envgate.sh <task> <mode>   mode = pristine | oracle
#
# Model-independent environment gate for one r2egym task. Mirrors harbor's
# apptainer worker: instance start + exec instance://, writable ext3 overlay,
# /tests + /logs/verifier + /workspace binds.
#
#   pristine -> /testbed stays at base_commit^ (the buggy parent). Expect 0.0.
#   oracle   -> `git checkout <base_commit> -- .` applies the GOLD patch.
#               Expect 1.0.
#
# A pristine 0.0 AND an oracle 1.0 on the same task proves the reward can take
# BOTH values, i.e. the environment measures code state -- no model involved.
#
# --- 2026-08-06 v2 -----------------------------------------------------------
# v1 returned reward=null / exit=2 on EVERY task and mode. That was a bug in
# THIS harness, not in the environments: the staged r2egym test.sh runs
#   uv pip install chardet
# under `set -e`, and JURECA compute nodes have no PyPI access, so test.sh died
# before it ever ran a test. Harbor does not hit this because worker.py
#   (a) bind-mounts $BRIDGE_AGENT_TOOLS into the container read-only, and
#   (b) rewrites every test.sh with _patch_test_sh_for_offline_pip(), injecting
#       UV_NO_INDEX / UV_FIND_LINKS / PIP_NO_INDEX / PIP_FIND_LINKS pointing at
#       the local wheel cache (87 wheels, incl. chardet).
# v2 does both. The patch function is EXTRACTED FROM THE LIVE worker.py rather
# than reimplemented here, so this gate cannot silently drift from what the real
# rollouts do -- the whole point of the gate is that it measures the same
# environment the trainer sees.
# v2 also emits a tail of the container output in the JSON, because v1's
# "reward: null" told me nothing about why.
set -uo pipefail

TASK="${1:?usage: envgate.sh <task> <pristine|oracle>}"
MODE="${2:-pristine}"
ROOT=/p/scratch/synthlaion/lee27
export APPTAINER_CACHEDIR=$ROOT/apptainer_cache
export APPTAINER_TMPDIR=$ROOT/apptainer_tmp
STAGE=$ROOT/envgate/$TASK

# Same values harbor's worker uses.
export BRIDGE_AGENT_TOOLS=${BRIDGE_AGENT_TOOLS:-$ROOT/agent_tools}
WORKER=${WORKER:-$ROOT/apptainer_bridge/9c31e931/worker.py}

SIF=$(ls $ROOT/r2egym_sif/build_${TASK}-*.sif 2>/dev/null | head -1)
if [[ -z "$SIF" ]]; then echo "{\"task\":\"$TASK\",\"mode\":\"$MODE\",\"error\":\"no_sif\"}"; exit 0; fi
if [[ ! -f "$STAGE/test.sh" || ! -f "$STAGE/metadata.json" ]]; then
  echo "{\"task\":\"$TASK\",\"mode\":\"$MODE\",\"error\":\"not_staged\"}"; exit 0
fi

BASE=$(python3 -c "import json;print(json.load(open('$STAGE/metadata.json'))['base_commit'])" 2>/dev/null)
if [[ -z "$BASE" ]]; then echo "{\"task\":\"$TASK\",\"mode\":\"$MODE\",\"error\":\"no_base_commit\"}"; exit 0; fi

W=$(mktemp -d $ROOT/eg.XXXXXXXX)
INST="eg_$(echo "${TASK}_${MODE}_$$" | tr -cd 'a-zA-Z0-9_')"
mkdir -p $W/tests $W/logs/verifier $W/ws
cp $STAGE/test.sh $W/tests/test.sh
cp $STAGE/metadata.json $W/ws/metadata.json

# Apply harbor's OWN offline-pip patch to the staged test.sh. We slice the
# function out of the live worker.py and exec it, so this is literally harbor's
# code path -- no reimplementation to drift.
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
  *PATCH_OK*) PATCH_STATUS="ok" ;;
  *) PATCH_STATUS="FAILED: $(echo "$PATCHED" | tr -d '"' | tail -1)" ;;
esac

apptainer overlay create --size 2048 $W/ov.img >/dev/null 2>&1

cleanup() { apptainer instance stop "$INST" >/dev/null 2>&1; rm -rf "$W"; }
trap cleanup EXIT

# NOTE the agent_tools bind: without it the wheel cache the patch points at
# does not exist inside the container and we are back to exit 2.
# --no-home is REQUIRED, not optional. Apptainer mounts the host $HOME by
# default; r2egym's test.sh opens with `source ~/.bashrc`, so without this the
# container sources the CLUSTER login bashrc (module init, broken conda) instead
# of the image's /root/.bashrc. worker.py:584 does the same thing and says why:
# "Prevent host home mount - it causes quota issues and overwrites container's
# /root/.bashrc with the host's (which has broken conda)."
# This is the prime suspect for the v2 smoke stalling out the full 25-min wall.
apptainer instance start --overlay $W/ov.img \
  --no-home \
  --bind $W/tests:/tests:rw \
  --bind $W/logs/verifier:/logs/verifier:rw \
  --bind $W/ws:/workspace:rw \
  --bind $BRIDGE_AGENT_TOOLS:$BRIDGE_AGENT_TOOLS:ro \
  "$SIF" "$INST" >$W/start.log 2>&1
if [[ $? -ne 0 ]]; then
  echo "{\"task\":\"$TASK\",\"mode\":\"$MODE\",\"error\":\"instance_start_failed\",\"detail\":\"$(tail -1 $W/start.log | tr -d '"'"'"'"\\')\"}"
  exit 0
fi

# HEAD before we touch anything, and the expected buggy-parent value.
HEAD0=$(apptainer exec instance://$INST git -C /testbed rev-parse HEAD 2>/dev/null)
PARENT=$(apptainer exec instance://$INST git -C /testbed rev-parse ${BASE}^ 2>/dev/null)

PRE="true"
if [[ "$MODE" == "oracle" ]]; then
  # Gold patch = the diff between the buggy parent and base_commit.
  PRE="git -C /testbed checkout $BASE -- . && git -C /testbed status --porcelain | head -5"
fi

# Bound the run OURSELVES. If the srun wall kills us instead, the script dies
# before emitting anything and a hang is indistinguishable from slow tests --
# which is exactly what happened on the first v2 smoke (r2egym-2514 ate the full
# 25 min srun limit and produced zero output, while real rollout trials finish in
# ~3.4 min median). With our own timeout we always emit JSON, and `tail` shows
# the last thing the container printed before it stalled.
timeout -k 10 "${EG_TIMEOUT:-1500}" \
  apptainer exec instance://$INST bash -c \
  "export HOME=/root; $PRE; cd /root 2>/dev/null || cd /; bash /tests/test.sh" \
  >$W/out.txt 2>&1
RC=$?
TIMED_OUT=false
[[ $RC -eq 124 || $RC -eq 137 ]] && TIMED_OUT=true

REWARD=$(cat $W/logs/verifier/reward.txt 2>/dev/null | tr -d '\n\r ')
SUMMARY=$(grep -oE "[0-9]+ failed,? ?[0-9]* ?p?a?s?s?e?d?|[0-9]+ passed" $W/out.txt 2>/dev/null | tail -1)
TAIL=$(tail -c 600 $W/out.txt 2>/dev/null)

python3 - "$TASK" "$MODE" "${REWARD:-null}" "${SUMMARY:-}" "$RC" "${HEAD0:-}" "${PARENT:-}" "$BASE" "$PATCH_STATUS" "${TAIL:-}" "$TIMED_OUT" <<'PY'
import json, sys
task, mode, reward, summary, rc, head0, parent, base, patch, tail, timed_out = sys.argv[1:12]
try:
    r = float(reward)
except Exception:
    r = None
print(json.dumps({
    "task": task, "mode": mode, "reward": r, "raw_reward": reward,
    "pytest": summary, "exit": int(rc), "offline_patch": patch,
    "head_before": head0[:12], "buggy_parent": parent[:12],
    "base_commit": base[:12],
    "head_was_buggy_parent": bool(head0) and head0 == parent,
    "timed_out": timed_out == "true",
    "tail": tail[-600:],
}))
PY
