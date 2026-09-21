#!/usr/bin/env python3
"""gepa_score.py <wave> — score one GEPA wave: per (task, candidate) reward + behaviour vector, the per-task Pareto
front, and the paired comparisons the selection and the gate run on.

Reads the probe's trials (see gepa_feat.iter_trials for the layout) and writes
  experiments/gepa/<wave>/scores.csv   one row per (task, candidate)
  experiments/gepa/<wave>/summary.md   per-candidate pass rate, paired delta vs ctl and vs the parent, every axis,
                                       and the Pareto-front win counts
The summary is written BEFORE the next reflection -- that is the rule the loop runs on, so a reflection is never
done from memory of a run.

Scoring rules that matter:
  * one row per TRIAL DIR = one sample; the 000-003 attempt dirs inside it are retries, so only the last attempt with
    a trajectory counts (pass8_table.py's `attempts` column counts retries and is not a per-sample count);
  * a trial whose attempt died before the verifier has reward None. It is DROPPED, never scored 0. `dropped` is
    reported per candidate: if it differs much between arms the comparison is not paired and the wave is void;
  * pass = mean over scored trials of (reward >= 1.0). With the default k=1 that is the single trial's outcome.

Pareto (per task): a candidate is on that task's front when no other candidate is >= it on pass AND on all eight
behaviour axes with at least one strict >. `wins` = tasks whose front it is on; `sole` = tasks where it is the only
member. Parent sampling is proportional to `wins` (gepa_ledger.py sample).

Usage:
  gepa_score.py w1                                   # the wave's own run dirs, experiments/gepa<wave>_s*
  gepa_score.py w1 --parent c000                     # also the paired gate against the parent
  gepa_score.py fixture --runs /e/.../p2o6all_s0 --out /tmp/p2ofix   # any existing probe, as a fixture
Python 3.9 / stdlib (Jupiter login node; one process, OMP_NUM_THREADS=1).
"""
import argparse, collections, csv, glob, json, os, random, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gepa_feat import AXES, iter_trials, resolve_runs, trial_features  # noqa: E402

E = "/e/fscratch/reformo/lee27/experiments"
ap = argparse.ArgumentParser()
ap.add_argument("wave")
ap.add_argument("--runs", default=None,
                help="glob of run dirs; by default the queue's done/<wave>.*.json run_dirs, else the gepa_jobs "
                     "shards, else the fallback probe run under experiments/")
ap.add_argument("--out", default=None, help="output dir (default experiments/gepa/<wave>)")
ap.add_argument("--ctl", default="ctl", help="the control arm's candidate id")
ap.add_argument("--parent", default=None, help="also compare every candidate against this one (the cheap gate)")
ap.add_argument("--boot", type=int, default=2000)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--gate-wins", type=int, default=2, help="gate: paired (wins - losses) at or above this passes")
ap.add_argument("--gate-axis", type=float, default=0.10, help="gate: an axis gain at or above this passes")
ap.add_argument("--gate-pass-floor", type=float, default=-0.02, help="gate: an axis win still needs pass delta at or above this")
ap.add_argument("--gate-axis-name", action="append", default=[],
                help="gate: the axis this child was PREDICTED to move, as 'cand=axis' (repeatable). Without it the gate "
                     "falls back to the best of eight axes, which is a multiple-comparisons trap -- see the warning it prints")
ap.add_argument("--strata", default=None, help="split.tsv: adds the paired delta per succ bucket and the OOD-repo check")
ap.add_argument("--ood-repos", default="tornado,scrapy", help="--strata: the repos held out of the v2 training band")
ap.add_argument("--progress", type=int, default=250)
a = ap.parse_args()
runs = resolve_runs(a.wave, a.runs)
if not runs:
    sys.exit("no run dirs for wave %s (looked in the queue's done/ records, gepa_jobs/%s_*_s*, and "
             "experiments/gepa%s_s*); pass --runs to point at them explicitly" % (a.wave, a.wave, a.wave))
out = a.out or "%s/gepa/%s" % (E, a.wave)
os.makedirs(out, exist_ok=True)
rng = random.Random(a.seed)

# ---------------------------------------------------------------- read trials
cells = collections.defaultdict(list)
n = 0
for task, cand, td in iter_trials(runs):
    f = trial_features(td)
    cells[(task, cand)].append(f)
    n += 1
    if a.progress and n % a.progress == 0:
        print("  %d trials..." % n, file=sys.stderr)
if not cells:
    sys.exit("no trials found under %s" % ", ".join(runs))
cands = sorted({c for _, c in cells})
tasks = sorted({t for t, _ in cells})
print("read %d trials: %d tasks x %d arms (%s)" % (n, len(tasks), len(cands), ",".join(cands)), file=sys.stderr)

COUNTERS = ["turns", "n_commands", "no_cmd_steps", "think_turns", "declared_done", "json_reject_steps", "preamble_steps",
            "n_edit_cmds", "max_repeat_run", "rm_repo_cmds", "ran_repo_suite", "ran_repro_script",
            "wrote_check_script", "ran_check_script", "read_back_edited", "assert_or_diff", "ctx_death"]


def agg(fs):
    """(task, cand) -> one row: pass over scored trials, axes over parsed trials."""
    scored = [f for f in fs if f.get("reward") is not None]
    ok = [f for f in fs if "parse_error" not in f]
    r = {"trials": len(fs), "scored": len(scored), "dropped": len(fs) - len(scored),
         "parse_errors": len(fs) - len(ok)}
    r["pass"] = (sum(1 for f in scored if f["reward"] >= 1.0) / len(scored)) if scored else None
    r["reward"] = (sum(f["reward"] for f in scored) / len(scored)) if scored else None
    for k in AXES + COUNTERS:
        v = [f[k] for f in ok if f.get(k) is not None]
        r[k] = (sum(v) / len(v)) if v else None
    r["exception_types"] = ";".join(sorted({f.get("exception_type") or "" for f in fs} - {""}))
    return r


rows = {k: agg(v) for k, v in cells.items()}

# ---------------------------------------------------------------- per-task Pareto front
def vec(t, c):
    r = rows.get((t, c))
    if not r or r["pass"] is None:
        return None
    v = [r["pass"]] + [r[x] if r[x] is not None else 0.0 for x in AXES]
    return v


def dominates(u, v):
    return all(x >= y for x, y in zip(u, v)) and any(x > y for x, y in zip(u, v))


front = collections.defaultdict(set)
for t in tasks:
    have = [(c, vec(t, c)) for c in cands]
    have = [(c, v) for c, v in have if v is not None]
    for c, v in have:
        if not any(dominates(w, v) for d, w in have if d != c):
            front[t].add(c)
wins = {c: sum(1 for t in tasks if c in front[t]) for c in cands}
sole = {c: sum(1 for t in tasks if front[t] == {c}) for c in cands}

# ---------------------------------------------------------------- paired comparisons
def paired(c, ref, key="pass"):
    """Tasks where BOTH arms produced a scored trial. Returns (n, mean ref, mean c, deltas)."""
    pr, pc, d = [], [], []
    for t in tasks:
        x, y = rows.get((t, ref)), rows.get((t, c))
        if not x or not y or x[key] is None or y[key] is None:
            continue
        pr.append(x[key])
        pc.append(y[key])
        d.append(y[key] - x[key])
    return len(d), (sum(pr) / len(pr) if pr else None), (sum(pc) / len(pc) if pc else None), d


def ci(d):
    if len(d) < 2:
        return (0.0, 0.0)
    bs = sorted(sum(rng.choice(d) for _ in range(len(d))) / len(d) for _ in range(a.boot))
    return bs[int(.025 * a.boot)], bs[int(.975 * a.boot)]


# ---------------------------------------------------------------- write
fields = (["task", "cand", "trials", "scored", "dropped", "parse_errors", "pass", "reward"]
          + AXES + COUNTERS + ["on_front", "exception_types"])
with open("%s/scores.csv" % out, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=fields)
    w.writeheader()
    for t in tasks:
        for c in cands:
            r = rows.get((t, c))
            if not r:
                continue
            d = {"task": t, "cand": c, "on_front": int(c in front[t])}
            d.update({k: r.get(k) for k in fields if k not in d})
            w.writerow(d)

L = []


def emit(s=""):
    L.append(s)


emit("# GEPA wave %s\n" % a.wave)
emit("runs: %s" % ", ".join(runs))
emit("%d trials, %d tasks, arms %s\n" % (n, len(tasks), ",".join(cands)))
emit("| cand | tasks | scored | dropped | pass | d vs %s | 95%% CI | w/l | front wins | sole |" % a.ctl)
emit("|---|---|---|---|---|---|---|---|---|---|")
for c in cands:
    ts = [t for t in tasks if (t, c) in rows]
    sc = sum(rows[(t, c)]["scored"] for t in ts)
    dr = sum(rows[(t, c)]["dropped"] for t in ts)
    pv = [rows[(t, c)]["pass"] for t in ts if rows[(t, c)]["pass"] is not None]
    if c == a.ctl:
        emit("| %s | %d | %d | %d | %.3f | - | - | - | %d | %d |" % (
            c, len(ts), sc, dr, (sum(pv) / len(pv)) if pv else float("nan"), wins[c], sole[c]))
        continue
    m, pr, pc, d = paired(c, a.ctl)
    lo, hi = ci(d)
    emit("| %s | %d | %d | %d | %.3f | %+.3f (n=%d) | [%+.3f, %+.3f] | %d/%d | %d | %d |" % (
        c, len(ts), sc, dr, (sum(pv) / len(pv)) if pv else float("nan"),
        (sum(d) / len(d)) if d else 0.0, m, lo, hi,
        sum(1 for x in d if x > 0), sum(1 for x in d if x < 0), wins[c], sole[c]))

emit("\n## Behaviour axes (mean over parsed trials; delta vs %s on paired tasks)\n" % a.ctl)
emit("| cand | %s |" % " | ".join(AXES))
emit("|---|%s" % ("---|" * len(AXES)))
for c in cands:
    cellv = []
    for ax in AXES:
        vs = [rows[(t, c)][ax] for t in tasks if (t, c) in rows and rows[(t, c)][ax] is not None]
        mu = (sum(vs) / len(vs)) if vs else float("nan")
        if c == a.ctl:
            cellv.append("%.2f" % mu)
        else:
            _, _, _, d = paired(c, a.ctl, ax)
            cellv.append("%.2f (%+.2f)" % (mu, (sum(d) / len(d)) if d else 0.0))
    emit("| %s | %s |" % (c, " | ".join(cellv)))

if a.strata:
    meta = {r["task"]: r for r in csv.DictReader(open(a.strata), delimiter="\t")}
    ood = set(a.ood_repos.split(","))
    groups = [("bucket " + b, lambda t, b=b: (meta.get(t) or {}).get("bucket") == b) for b in ["0", "1-3", "4-5", "6-8"]]
    groups += [("OOD repo", lambda t: (meta.get(t) or {}).get("repo") in ood),
               ("ID repo", lambda t: (meta.get(t) or {}).get("repo") not in ood)]
    emit("\n## Paired pass delta vs %s by stratum\n" % a.ctl)
    emit("| cand | %s |" % " | ".join(g for g, _ in groups))
    emit("|---|%s" % ("---|" * len(groups)))
    for c in cands:
        if c == a.ctl:
            continue
        cellv = []
        for _, pred in groups:
            d = []
            for t in tasks:
                if not pred(t):
                    continue
                x, y = rows.get((t, a.ctl)), rows.get((t, c))
                if x and y and x["pass"] is not None and y["pass"] is not None:
                    d.append(y["pass"] - x["pass"])
            cellv.append("%+.3f (n=%d)" % ((sum(d) / len(d)) if d else 0.0, len(d)))
        emit("| %s | %s |" % (c, " | ".join(cellv)))

if a.parent:
    predicted = dict(x.split("=", 1) for x in a.gate_axis_name)
    bad = [v for v in predicted.values() if v not in AXES]
    if bad:
        sys.exit("--gate-axis-name: %s is not an axis (%s)" % (bad[0], ", ".join(AXES)))
    emit("\n## Cheap gate vs parent `%s`\n" % a.parent)
    emit("Rule, fixed before the run: ACCEPT if paired (wins - losses) >= %d, or if the child's PREDICTED axis gains"
         " >= %+.2f while the paired pass delta stays >= %+.2f. Pass rate alone does not decide -- at this n it cannot."
         % (a.gate_wins, a.gate_axis, a.gate_pass_floor))
    if len(predicted) < len([c for c in cands if c not in (a.parent, a.ctl)]):
        emit("\n> ⚠ Some children have no predicted axis, so the gate falls back to the BEST of eight axes for them."
             " That is a multiple-comparisons trap: with eight axes something usually clears +0.10 by chance or by a"
             " detector artifact, and those rows are marked `weak`. The p2o6all fixture shows the failure mode --"
             " every arm 'gains' ~+0.17 `in_place` over block A only because A's replace-script style opens a"
             " variable, which the literal-path detector cannot see. Name the axis the reflection predicted.")
    emit("\n| cand | n | pass d | w/l | axis tested | gain | verdict |")
    emit("|---|---|---|---|---|---|---|")
    for c in cands:
        if c in (a.parent, a.ctl):
            continue
        m, pr, pc, d = paired(c, a.parent)
        wl = (sum(1 for x in d if x > 0), sum(1 for x in d if x < 0))
        dp = (sum(d) / len(d)) if d else 0.0

        def gain(ax):
            _, _, _, da = paired(c, a.parent, ax)
            return (sum(da) / len(da)) if da else 0.0

        if c in predicted:
            gax, weak = predicted[c], False
        else:
            gax, weak = max(((gain(x), x) for x in AXES))[1], True
        g = gain(gax)
        ok_wins = wl[0] - wl[1] >= a.gate_wins
        ok_axis = g >= a.gate_axis and dp >= a.gate_pass_floor
        v = "ACCEPT" if ok_wins else ("ACCEPT" if (ok_axis and not weak) else ("weak" if ok_axis else "reject"))
        emit("| %s | %d | %+.3f | %d/%d | %s%s | %+.2f | %s |" % (
            c, m, dp, wl[0], wl[1], gax, " (unpredicted)" if weak else "", g, v))

emit("\n## Sanity\n")
emit("Dropped-sample rate per arm (an arm far off the others voids the pairing):")
for c in cands:
    ts = [t for t in tasks if (t, c) in rows]
    tot = sum(rows[(t, c)]["trials"] for t in ts)
    dr = sum(rows[(t, c)]["dropped"] for t in ts)
    pe = sum(rows[(t, c)]["parse_errors"] for t in ts)
    emit("  %-8s %d/%d dropped (%.1f %%), %d unparseable" % (c, dr, tot, 100.0 * dr / max(1, tot), pe))
open("%s/summary.md" % out, "w").write("\n".join(L) + "\n")
print("\n".join(L))
print("\nwrote %s/scores.csv and %s/summary.md" % (out, out))
