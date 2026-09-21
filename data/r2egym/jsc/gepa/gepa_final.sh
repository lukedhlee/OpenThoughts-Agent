#!/bin/bash
# gepa_final.sh — the ONE run on the test set: the winning block against the control, paired, on 500 tasks the loop
# has never scored, plus the OOD-repo check. Two phases.
#
#   bash gepa_final.sh build <cand> <cands.json>    # tree + probe config + cost line; submits nothing
#   SUBMIT=1 bash gepa_final.sh build <cand> <cands.json>
#   bash gepa_final.sh readout                      # paired delta with a bootstrap CI, per bucket and OOD vs ID
#
# The test set is untouched until this runs, and it runs ONCE per loop. If the readout is disappointing, that is the
# result; re-running the test set after seeing it turns the held-out estimate into another dev set. A second final run
# needs a new test split, not a second look at this one.
#
# HARD RULE: the build phase prints the cost line. Show it to Luke, get the go, then re-run with SUBMIT=1.
set -uo pipefail
PHASE=${1:?build | readout}
C=/e/project1/transfernetx/lee27/code/snowball
G=$C/gepa
E=/e/fscratch/reformo/lee27/experiments
PY=${PY:-python3}
WAVE=${WAVE:-final}
export OMP_NUM_THREADS=1

if [ "$PHASE" = build ]; then
  CAND=${2:?the winning candidate id}; CANDS=${3:?the cands.json holding its block text}
  K=${K:-1}; NODES=${NODES:-8}
  # one file with exactly the winner, so the test tree cannot accidentally carry the rest of the population
  $PY - "$CANDS" "$CAND" "$E/gepa/${WAVE}_cands.json" <<'PY' || exit 1
import json, sys
src, cand, dst = sys.argv[1:4]
d = json.load(open(src))
assert cand in d["blocks"], "%s is not in %s" % (cand, src)
json.dump({"delim": d["delim"], "blocks": {cand: d["blocks"][cand]}}, open(dst, "w"), indent=1)
print("final candidate %s written to %s" % (cand, dst))
PY
  $PY $G/gepa_tree.py --wave "$WAVE" --candidates "$E/gepa/${WAVE}_cands.json" --split test --verify 40 || exit 1
  K=$K NODES=$NODES bash $G/gepa_wave.sh "$WAVE" "$K" "$NODES"
  exit $?
fi

if [ "$PHASE" = readout ]; then
  CTL=${CTL:-ctl}
  $PY $G/gepa_score.py "$WAVE" --strata $E/gepa/split.tsv --ctl "$CTL" || exit 1
  echo
  echo "Read $E/gepa/$WAVE/summary.md. The headline is the paired pass delta vs $CTL with its bootstrap 95 % interval,"
  echo "on 500 tasks scored once. The OOD-repo row is the transfer check: a block that only helps ID repos is a"
  echo "band-specific trick, not a procedure."
  exit 0
fi
echo "unknown phase $PHASE (build | readout)"; exit 1
