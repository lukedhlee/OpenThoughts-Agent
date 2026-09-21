#!/usr/bin/env python3
"""gepa_worst.py <wave> <cand> [n] — the FEEDBACK tasks a candidate should be reflected on.

Reads the wave's feedback scores and ranks the candidate's tasks by how badly it did against a reference arm:
  1. tasks it LOST on pass (reference solved, candidate did not)
  2. tasks where it violated the most behaviour axes
  3. tasks where it is off the per-task Pareto front
Prints the ranked list with the axis violations named, and the gepa_dump.py command line for each, so the reflection
reads the worst traces rather than a random sample of them.

FEEDBACK ONLY. Dev and test tasks are dropped from the ranking and never get a dump command, because their
trajectories are closed to this session: dev is scores-only and test is sealed until gepa_final.sh. If the feedback
scores do not exist yet, score that leg first -- `gepa_score.py <wave> --leg feedback`.

  gepa_worst.py w1 c012                 # vs ctl
  gepa_worst.py w1 c012 8 --ref c000    # vs the parent, top 8
Python 3.9 / stdlib (Jupiter login node).
"""
import argparse, csv, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gepa_feat import AXES, read_split  # noqa: E402

E = "/e/fscratch/reformo/lee27/experiments"
ap = argparse.ArgumentParser()
ap.add_argument("wave")
ap.add_argument("cand")
ap.add_argument("n", nargs="?", type=int, default=10)
ap.add_argument("--ref", default="ctl", help="reference arm to lose against (default the control)")
ap.add_argument("--scores", default=None, help="default experiments/gepa/<wave>/feedback_scores.csv")
a = ap.parse_args()
sp = a.scores or "%s/gepa/%s/feedback_scores.csv" % (E, a.wave)
if not os.path.exists(sp):
    sys.exit("no feedback scores at %s -- run `gepa_score.py %s --leg feedback` first" % (sp, a.wave))
closed = read_split("dev") | read_split("test")
rows, dropped = {}, 0
for r in csv.DictReader(open(sp)):
    if r["task"] in closed:
        dropped += 1
        continue
    rows[(r["task"], r["cand"])] = r
if dropped:
    print("# dropped %d rows whose tasks are in dev/test: those trajectories are closed to this session\n" % dropped)
if not rows:
    sys.exit("no feedback-task rows in %s" % sp)


def num(r, k):
    v = r.get(k)
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


tasks = sorted({t for t, _ in rows})
ranked = []
for t in tasks:
    c = rows.get((t, a.cand))
    ref = rows.get((t, a.ref))
    if not c:
        continue
    pc, pr = num(c, "pass"), num(ref, "pass") if ref else None
    if pc is None:
        ranked.append((3, 0, 0, t, "dropped (no scored trial: %s)" % (c.get("exception_types") or "?")))
        continue
    lost = 1 if (pr is not None and pr > pc) else 0
    viol = [x for x in AXES if (num(c, x) or 0.0) < 0.5]
    offfront = 0 if c.get("on_front") == "1" else 1
    why = []
    if lost:
        why.append("lost to %s (%.2f vs %.2f)" % (a.ref, pr, pc))
    elif pc == 0.0:
        why.append("unsolved by both")
    if viol:
        why.append("violates " + ",".join(viol))
    if offfront and not lost:
        why.append("off the front")
    ranked.append((0 if lost else (1 if viol else 2), -len(viol), -offfront, t, "; ".join(why) or "clean"))
ranked.sort(key=lambda x: (x[0], x[1], x[2], x[3]))

print("# worst %d FEEDBACK tasks for %s in wave %s (reference %s)\n" % (min(a.n, len(ranked)), a.cand, a.wave, a.ref))
for _, nv, _, t, why in ranked[:a.n]:
    print("%-22s %s" % (t, why))
    print("    python3 gepa_dump.py %s %s %s" % (a.wave, a.cand, t))
    if rows.get((t, a.ref)):
        print("    python3 gepa_dump.py %s %s %s   # the reference, for the contrast" % (a.wave, a.ref, t))
lost_n = sum(1 for x in ranked if x[0] == 0)
print("\n%d of %d tasks lost to %s; %d have at least one axis violation." % (
    lost_n, len(ranked), a.ref, sum(1 for x in ranked if x[1] < 0)))
