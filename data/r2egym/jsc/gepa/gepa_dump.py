#!/usr/bin/env python3
"""gepa_dump.py <wave> <cand> <task> — one trial as a readable transcript, for the reflection step.

The raw trajectory.json is a few megabytes of JSON with the whole message history repeated; this prints only what a
reader needs to see WHY the run went the way it did: per turn the think flag, the analysis and plan the model wrote,
the commands it sent, and the tail of the screen it got back. Ends with the reward, the exception and the behaviour
vector so the reflection is looking at the same numbers the scorer used.

Dev traces are open -- the session may read any of them. Test traces are not read until the final run is done.

  gepa_dump.py w1 c012 r2egym-v1-00009               # the wave's run dirs
  gepa_dump.py fixture A r2egym-v1-00009 --runs /e/.../p2o6all_s0
  gepa_dump.py w1 c012 r2egym-v1-00009 --screen 40 --turns 12:20
Python 3.9 / stdlib (Jupiter login node).
"""
import argparse, glob, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gepa_feat import AXES, iter_trials, last_attempt, resolve_runs, trial_features, turns_of  # noqa: E402

E = "/e/fscratch/reformo/lee27/experiments"
ap = argparse.ArgumentParser()
ap.add_argument("wave")
ap.add_argument("cand")
ap.add_argument("task")
ap.add_argument("--runs", default=None)
ap.add_argument("--screen", type=int, default=25, help="lines of screen output per turn (0 = none)")
ap.add_argument("--turns", default=None, help="turn range, e.g. 0:10 or 12:")
ap.add_argument("--trial", type=int, default=0, help="which trial of this (task, cand), when k > 1")
ap.add_argument("--plan", action="store_true", help="also print the plan field (analysis only by default)")
a = ap.parse_args()
runs = resolve_runs(a.wave, a.runs)
if not runs:
    sys.exit("no run dirs for wave %s; pass --runs to point at them explicitly" % a.wave)

hits = [td for t, c, td in iter_trials(runs) if t == a.task and c == a.cand]
if not hits:
    sys.exit("no trial for task %s cand %s under %s" % (a.task, a.cand, ", ".join(runs)))
if a.trial >= len(hits):
    sys.exit("only %d trials for (%s, %s)" % (len(hits), a.task, a.cand))
td = hits[a.trial]
att = last_attempt(td)
if not att:
    sys.exit("no attempt with a trajectory in %s" % td)
f = trial_features(td)

lo, hi = 0, 10 ** 9
if a.turns:
    p = a.turns.split(":")
    lo = int(p[0]) if p[0] else 0
    hi = int(p[1]) if len(p) > 1 and p[1] else 10 ** 9

print("=" * 100)
print("task %s   cand %s   trial %d of %d" % (a.task, a.cand, a.trial + 1, len(hits)))
print("attempt %s" % att)
print("reward %s (%s%s)   exception %s   turns %s" % (
    f.get("reward"), f.get("reward_src"), (" / " + f["reward_error"]) if f.get("reward_error") else "",
    f.get("exception_type") or "-", f.get("turns")))
print("=" * 100)

for i, an, pl, cmds, tc, th, obs in turns_of(att):
    if i < lo or i >= hi:
        continue
    print("\n--- turn %d%s%s" % (i, "  [think]" if th else "", "  [task_complete]" if tc else ""))
    if an:
        print("  analysis: " + an.strip().replace("\n", "\n            "))
    if a.plan and pl:
        print("  plan:     " + pl.strip().replace("\n", "\n            "))
    if not cmds:
        print("  (no commands parsed from this turn)")
    for k in cmds:
        for ln in k.rstrip("\n").split("\n"):
            print("  $ " + ln)
    if a.screen and obs:
        tail = obs.rstrip("\n").split("\n")[-a.screen:]
        for ln in tail:
            print("  | " + ln[:200])

print("\n" + "=" * 100)
print("behaviour vector: " + "  ".join("%s=%s" % (x, f.get(x)) for x in AXES))
print("edit_mode=%s  first_edit_turn=%s  max_repeat_run=%s  json_reject_steps=%s  declared_done=%s" % (
    f.get("edit_mode"), f.get("first_edit_turn"), f.get("max_repeat_run"), f.get("json_reject_steps"), f.get("declared_done")))
