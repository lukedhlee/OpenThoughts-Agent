#!/bin/bash
# gepa_final.sh — the ONE run on the test set: the winning block against the control, paired, on 500 tasks the loop
# has never scored, plus the OOD-repo check. Three phases.
#
#   bash gepa_final.sh confirm <cand> <cands.json>  # re-run the winner and ctl on DEV at k=2, then judge it
#   bash gepa_final.sh verdict                      # score the confirm wave and write the pass/fail marker
#   bash gepa_final.sh build <cand> <cands.json>    # build the test tree; enqueue nothing
#   CONFIRM=1 bash gepa_final.sh build <cand> ...   # build AND enqueue both arms onto the standing serve job
#   bash gepa_final.sh readout                      # paired delta with a bootstrap CI, per bucket and OOD vs ID
#
# The CONFIRM step exists because the winner was chosen by looking at dev many times. Selecting a maximum over a
# dozen candidates on one 500-task set inflates it: the winner is partly whichever candidate got the friendlier
# noise. So before the sealed test is spent, the winner is re-run against the control on dev at k=2 -- fresh
# samples, same tasks -- and must clear a paired +0.05 with a bootstrap 95 % interval strictly above zero. A
# winner that cannot reproduce its own dev margin will not survive the test set either, and finding that out on
# dev costs 2,000 attempts instead of the one held-out number we get.
#
# Enqueueing normally needs no new go -- the serve job is already paid for. This one does, and CONFIRM=1 is it,
# because it spends something that cannot be bought back: the test set is scored ONCE. If the readout disappoints,
# that is the result. Re-running the test set after seeing it turns the held-out estimate into another dev set, and
# a second honest number then needs a new test split, not a second look at this one.
set -uo pipefail
PHASE=${1:?confirm | verdict | build | readout}
C=/e/project1/transfernetx/lee27/code
G=${GEPA_CODE:-$C/snowball/gepa}
E=${GEPA_DIR:-/e/fscratch/reformo/lee27/experiments/gepa}
PY=${PY:-python3}
WAVE=${WAVE:-final}
K=${K:-1}
export OMP_NUM_THREADS=1

CWAVE=${CWAVE:-${WAVE}_confirm}
MARK=$E/${WAVE}_confirm.json
MIN_DELTA=${MIN_DELTA:-0.05}

if [ "$PHASE" = confirm ]; then
  CAND=${2:?the winning candidate id}; CANDS=${3:?the cands.json holding its block text}
  $PY - "$CANDS" "$CAND" "$E/${CWAVE}_cands.json" <<'PY' || exit 1
import json, sys
src, cand, dst = sys.argv[1:4]
d = json.load(open(src))
assert cand in d["blocks"], "%s is not in %s" % (cand, src)
json.dump({"delim": d["delim"], "blocks": {cand: d["blocks"][cand]}}, open(dst, "w"), indent=1)
print("confirm candidate %s -> %s" % (cand, dst))
PY
  $PY $G/gepa_tree.py --wave "$CWAVE" --candidates "$E/${CWAVE}_cands.json" --split dev --verify 40 || exit 1
  NT=$(wc -l < "$E/split_dev.txt")
  echo
  echo "CONFIRM: $NT dev tasks x 2 arms ($CAND and ctl) x k=2 = $((NT * 2 * 2)) attempts ~= $((NT * 2 * 2 / 100)) node-hours"
  echo "         on the allocation already held. Fresh samples on the same tasks, to check the winner's dev margin"
  echo "         reproduces before the sealed test set is spent."
  bash $G/gepa_queue.sh add-all "$CWAVE" 2 || exit 1
  echo "enqueued at k=2; when it finishes run:  bash $G/gepa_final.sh verdict"
  exit 0
fi

if [ "$PHASE" = verdict ]; then
  $PY $G/gepa_score.py "$CWAVE" --leg dev --json "$MARK" || exit 1
  $PY - "$MARK" "$MIN_DELTA" <<'PY'
import json, sys
p, mind = sys.argv[1], float(sys.argv[2])
v = json.load(open(p))
ok = False
for c, r in v["candidates"].items():
    passed = r["delta"] >= mind and r["ci_lo"] > 0
    ok = ok or passed
    print("%s: paired dev delta %+.3f [%+.3f, %+.3f] over %d tasks at k=2, %d/%d wins -> %s"
          % (c, r["delta"], r["ci_lo"], r["ci_hi"], r["n"], r["wins"], r["losses"],
             "CONFIRMED" if passed else "NOT CONFIRMED"))
    if r.get("oodmini_delta") is not None:
        print("   oodmini delta %+.3f" % r["oodmini_delta"])
v["confirmed"] = ok
v["min_delta"] = mind
json.dump(v, open(p, "w"), indent=1)
print()
print("VERDICT: %s (needs paired delta >= %+.2f with the 95 %% interval strictly above 0)"
      % ("CONFIRMED -- the test build is unlocked" if ok else "NOT CONFIRMED -- gepa_final.sh build will refuse", mind))
PY
  exit 0
fi

if [ "$PHASE" = build ]; then
  CAND=${2:?the winning candidate id}; CANDS=${3:?the cands.json holding its block text}
  # The sealed test set is spent only on a winner that reproduced its dev margin on fresh samples.
  if [ ! -f "$MARK" ]; then
    echo "no confirm marker at $MARK -- run the confirm step first:"
    echo "  bash $G/gepa_final.sh confirm $CAND $CANDS   # then, when it finishes: bash $G/gepa_final.sh verdict"
    exit 1
  fi
  $PY -c "
import json, sys
v = json.load(open('$MARK'))
if not v.get('confirmed'):
    sys.exit('confirm step did NOT pass (see $MARK). The test set is not spent on an unconfirmed winner.')
print('confirm marker: passed')
" || exit 1
  for s in queue running done; do
    if [ "$(ls -1 "$E/$s/$WAVE."*.json 2>/dev/null | wc -l)" -gt 0 ]; then
      echo "wave $WAVE already has $s entries -- the test set is scored ONCE. Read $E/$WAVE/test_summary.md, or start"
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
  # FINAL=1 is the only thing that lets a tree be built over the test split (gepa_tree.py refuses otherwise).
  FINAL=1 $PY $G/gepa_tree.py --wave "$WAVE" --candidates "$E/${WAVE}_cands.json" --split test --verify 40 || exit 1
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
  $PY $G/gepa_score.py "$WAVE" --leg test --strata "$E/split.tsv" --ctl "$CTL" || exit 1
  echo
  echo "Read $E/$WAVE/test_summary.md. The headline is the paired pass delta vs $CTL with its bootstrap 95 % interval,"
  echo "on $(wc -l < "$E/split_test.txt") tasks scored once. The OOD-repo row is the transfer check: a block that only"
  echo "helps ID repos is a band-specific trick, not a procedure, and should be reported as one."
  exit 0
fi
echo "unknown phase $PHASE (confirm | verdict | build | readout)"; exit 1
