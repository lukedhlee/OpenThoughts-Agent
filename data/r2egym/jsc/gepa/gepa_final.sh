#!/bin/bash
# gepa_final.sh — the ONE run on the test set: the winning block against the control, paired, on 500 tasks the loop
# has never scored, plus the OOD-repo check. Three phases.
#
#   bash gepa_final.sh build <cand> <cands.json>    # build the test tree; enqueue nothing
#   CONFIRM=1 bash gepa_final.sh build <cand> ...   # build AND enqueue both arms onto the standing serve job
#   bash gepa_final.sh readout                      # paired delta with a bootstrap CI, per bucket and OOD vs ID
#
# Enqueueing normally needs no new go -- the serve job is already paid for. This one does, and CONFIRM=1 is it,
# because it spends something that cannot be bought back: the test set is scored ONCE. If the readout disappoints,
# that is the result. Re-running the test set after seeing it turns the held-out estimate into another dev set, and
# a second honest number then needs a new test split, not a second look at this one.
set -uo pipefail
PHASE=${1:?build | readout}
C=/e/project1/transfernetx/lee27/code
G=${GEPA_CODE:-$C/snowball/gepa}
E=${GEPA_DIR:-/e/fscratch/reformo/lee27/experiments/gepa}
PY=${PY:-python3}
WAVE=${WAVE:-final}
K=${K:-1}
export OMP_NUM_THREADS=1

if [ "$PHASE" = build ]; then
  CAND=${2:?the winning candidate id}; CANDS=${3:?the cands.json holding its block text}
  for s in queue running done; do
    if [ "$(ls -1 "$E/$s/$WAVE."*.json 2>/dev/null | wc -l)" -gt 0 ]; then
      echo "wave $WAVE already has $s entries -- the test set is scored ONCE. Read $E/$WAVE/summary.md, or start"
      echo "a new test split if a second honest number is really needed."
      exit 1
    fi
  done
  # one file with exactly the winner, so the test tree cannot accidentally carry the rest of the population
  $PY - "$CANDS" "$CAND" "$E/${WAVE}_cands.json" <<'PY' || exit 1
import json, sys
src, cand, dst = sys.argv[1:4]
d = json.load(open(src))
assert cand in d["blocks"], "%s is not in %s" % (cand, src)
json.dump({"delim": d["delim"], "blocks": {cand: d["blocks"][cand]}}, open(dst, "w"), indent=1)
print("final candidate %s written to %s" % (cand, dst))
PY
  $PY $G/gepa_tree.py --wave "$WAVE" --candidates "$E/${WAVE}_cands.json" --split test --verify 40 || exit 1
  NT=$(wc -l < "$E/split_test.txt")
  echo
  echo "FINAL: $NT test tasks x 2 arms ($CAND and ctl) x k=$K = $((NT * 2 * K)) attempts on the standing serve job."
  echo "       At ~100 attempts/node-h that is ~$(( NT * 2 * K / 100 )) node-hours of the allocation already held."
  echo "       The test set is scored ONCE. This is the only enqueue in the loop that needs a fresh go."
  if [ "${CONFIRM:-0}" != 1 ]; then
    echo
    echo "NOT ENQUEUED. Confirm with Luke, then:"
    echo "  CONFIRM=1 bash $G/gepa_final.sh build $CAND $CANDS"
    exit 0
  fi
  bash $G/gepa_queue.sh add-all "$WAVE" "$K" || exit 1
  echo "enqueued; watch with: bash $G/gepa_serve.sh status"
  exit 0
fi

if [ "$PHASE" = readout ]; then
  CTL=${CTL:-ctl}
  $PY $G/gepa_score.py "$WAVE" --strata "$E/split.tsv" --ctl "$CTL" || exit 1
  echo
  echo "Read $E/$WAVE/summary.md. The headline is the paired pass delta vs $CTL with its bootstrap 95 % interval,"
  echo "on $(wc -l < "$E/split_test.txt") tasks scored once. The OOD-repo row is the transfer check: a block that only"
  echo "helps ID repos is a band-specific trick, not a procedure, and should be reported as one."
  exit 0
fi
echo "unknown phase $PHASE (build | readout)"; exit 1
