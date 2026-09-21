#!/usr/bin/env python3
"""gepa_judge.py <wave> <cand> --axis <axis> — sample 8 feedback traces to hand-grade one behaviour detector.

The eight axes are regexes over a command stream. They are wrong sometimes, and a wrong one is worse than a missing
one: it decides the Pareto front and the gate while looking like evidence. `in_place` already proved that -- it
cannot see a write through a variable, so every arm appeared to gain it over the block whose own style it missed.

So an axis earns its place by being read. This picks 8 of THIS WAVE's feedback trials for one candidate, balanced
between trials the detector scored 1 and trials it scored 0 (a one-sided sample only ever catches one kind of
error), prints the detector's verdict beside the dump command, and leaves a blank for the human verdict. Read all
eight, then record the agreement:

    python3 gepa_judge.py w1 c012 --axis self_check
    ... read the eight dumps ...
    python3 gepa_ledger.py judge --cand c012 --axis self_check --agree 7 --of 8

Below 6 of 8 the ledger FREEZES the axis and gepa_score.py drops it from the front until a later judge clears it.
Python 3.9 / stdlib (Jupiter login node).
"""
import argparse, csv, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gepa_feat import AXES, wave_feedback, wave_rng  # noqa: E402

E = "/e/fscratch/reformo/lee27/experiments"
ap = argparse.ArgumentParser()
ap.add_argument("wave")
ap.add_argument("cand")
ap.add_argument("--axis", required=True, choices=AXES, help="the axis this candidate was predicted to move")
ap.add_argument("--n", type=int, default=8, help="trials to grade (the ledger's default denominator is 8)")
ap.add_argument("--scores", default=None, help="default experiments/gepa/<wave>/feedback_scores.csv")
a = ap.parse_args()

sp = a.scores or "%s/gepa/%s/feedback_scores.csv" % (E, a.wave)
if not os.path.exists(sp):
    sys.exit("no feedback scores at %s -- run `gepa_score.py %s --leg feedback` first" % (sp, a.wave))
fb = set(wave_feedback(a.wave))
rows = []
for r in csv.DictReader(open(sp)):
    if r["cand"] != a.cand or (fb and r["task"] not in fb):
        continue
    try:
        v = float(r[a.axis])
    except (KeyError, ValueError, TypeError):
        continue
    rows.append((r["task"], v, r.get("pass")))
if not rows:
    sys.exit("no %s rows for candidate %s in %s" % (a.axis, a.cand, sp))

hi = sorted(t for t, v, _ in rows if v >= 0.5)
lo = sorted(t for t, v, _ in rows if v < 0.5)
rng = wave_rng(a.wave, "judge/%s/%s" % (a.cand, a.axis))
half = a.n // 2
pick = rng.sample(hi, min(half, len(hi))) + rng.sample(lo, min(a.n - half, len(lo)))
# top up from whichever side has more left, if one side was short
rest = [t for t in hi + lo if t not in pick]
pick += rng.sample(rest, min(a.n - len(pick), len(rest)))
byv = dict((t, v) for t, v, _ in rows)
bypass = dict((t, p) for t, _, p in rows)

print("# Judge %s on `%s`, wave %s -- %d trials (%d detector-yes, %d detector-no)\n"
      % (a.cand, a.axis, a.wave, len(pick), sum(1 for t in pick if byv[t] >= 0.5),
         sum(1 for t in pick if byv[t] < 0.5)))
print("Read each dump and decide for yourself whether the behaviour `%s` is present. Then compare with the" % a.axis)
print("detector column. Count the trials where you AGREE and record it:\n")
print("  python3 gepa_ledger.py judge --cand %s --axis %s --agree <n> --of %d\n" % (a.cand, a.axis, len(pick)))
print("| # | task | detector | pass | your verdict |")
print("|---|---|---|---|---|")
for i, t in enumerate(sorted(pick), 1):
    print("| %d | %s | %s | %s |   |" % (i, t, "yes" if byv[t] >= 0.5 else "no", bypass.get(t)))
print()
for t in sorted(pick):
    print("python3 gepa_dump.py %s %s %s    # detector says %s"
          % (a.wave, a.cand, t, "YES" if byv[t] >= 0.5 else "NO"))
print("\nAgreement below 6 of %d freezes `%s`: gepa_score.py will drop it from the Pareto front." % (len(pick), a.axis))
