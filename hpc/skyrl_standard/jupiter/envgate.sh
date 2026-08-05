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
set -uo pipefail

TASK="${1:?usage: envgate.sh <task> <pristine|oracle>}"
MODE="${2:-pristine}"
ROOT=/p/scratch/synthlaion/lee27
export APPTAINER_CACHEDIR=$ROOT/apptainer_cache
export APPTAINER_TMPDIR=$ROOT/apptainer_tmp
STAGE=$ROOT/envgate/$TASK

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
apptainer overlay create --size 2048 $W/ov.img >/dev/null 2>&1

cleanup() { apptainer instance stop "$INST" >/dev/null 2>&1; rm -rf "$W"; }
trap cleanup EXIT

apptainer instance start --overlay $W/ov.img \
  --bind $W/tests:/tests:rw \
  --bind $W/logs/verifier:/logs/verifier:rw \
  --bind $W/ws:/workspace:rw \
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

apptainer exec instance://$INST bash -c \
  "$PRE; cd /root 2>/dev/null || cd /; bash /tests/test.sh" >$W/out.txt 2>&1
RC=$?

REWARD=$(cat $W/logs/verifier/reward.txt 2>/dev/null | tr -d '\n\r ')
SUMMARY=$(grep -oE "[0-9]+ failed,? ?[0-9]* ?p?a?s?s?e?d?|[0-9]+ passed" $W/out.txt 2>/dev/null | tail -1)
CHANGED=$(grep -c "" /dev/null)  # placeholder, unused

python3 - "$TASK" "$MODE" "${REWARD:-null}" "${SUMMARY:-}" "$RC" "${HEAD0:-}" "${PARENT:-}" "$BASE" <<'PY'
import json, sys
task, mode, reward, summary, rc, head0, parent, base = sys.argv[1:9]
try:
    r = float(reward)
except Exception:
    r = None
print(json.dumps({
    "task": task, "mode": mode, "reward": r, "raw_reward": reward,
    "pytest": summary, "exit": int(rc),
    "head_before": head0[:12], "buggy_parent": parent[:12],
    "base_commit": base[:12],
    "head_was_buggy_parent": bool(head0) and head0 == parent,
}))
PY
