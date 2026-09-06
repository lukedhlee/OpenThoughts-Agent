#!/usr/bin/env python3
"""Stratified task sample for the workflow-prompt re-probe (2026-09-06).

Strata come from the base model's own pass@8 on TaskTrove (the tt60k tables, frozen-screen harness, header prompt):
  zero     succ == 0 of 8 scored            (never solved)
  medium   2 <= succ <= 5, in the clean band (sometimes solved; what RL trained on)
  success  succ >= 6                        (reliably solved)
Exclusions: not in the JSC allowlist, empty issue text, the band's drop list (no-op wins, truss), not fully sampled.
Writes <out>.tsv (task, repo, stratum, tt60k_succ, tt60k_scored, med_turns) and <out>.txt (task list) for tt_prompt.py.
Python 3.9 / stdlib (Jupiter login node).
"""
import argparse, csv, glob, json, os, random, collections
E = "/e/fscratch/reformo/lee27/experiments"; T = "/e/fscratch/reformo/lee27/tasks"
ap = argparse.ArgumentParser()
ap.add_argument("--per-stratum", type=int, default=50); ap.add_argument("--seed", type=int, default=20260906)
ap.add_argument("--out", default=E + "/ttwf_sample"); ap.add_argument("--tables", default=E + "/tt60k_s*/pass8_pass8_table.csv")
ap.add_argument("--tree", default=T + "/r2egym-tt-raw"); ap.add_argument("--band", default=T + "/r2egym-tt-band60k-clean")
ap.add_argument("--allow", default=T + "/allowlist_r2egym_tt_v1.txt"); ap.add_argument("--empty", default=T + "/tt_empty_issue_tasks.txt")
ap.add_argument("--drop", default=E + "/tt_ana/tt_band60k_drop.txt")
a = ap.parse_args()
rl = lambda p: {l.strip().split()[0] for l in open(p) if l.strip() and not l.startswith("#")} if os.path.exists(p) else set()
allow, empty, drop = rl(a.allow), rl(a.empty), rl(a.drop)
band = set(os.listdir(a.band))
rows = {}
for p in sorted(glob.glob(a.tables)):
    for r in csv.DictReader(open(p)):
        rows[r["task"]] = r
full = {t: r for t, r in rows.items() if r["full"] == "True" and int(r["scored"]) == 8 and t in allow and t not in empty and t not in drop}
hist = collections.Counter(int(r["succ"]) for r in full.values())
print("tables %d tasks; eligible (full, 8 scored, allowlisted, issue text, not dropped) %d" % (len(rows), len(full)))
print("succ histogram:", " ".join("%d:%d" % (k, hist[k]) for k in sorted(hist)))
strata = {
    "zero": [t for t, r in full.items() if int(r["succ"]) == 0],
    "medium": [t for t, r in full.items() if 2 <= int(r["succ"]) <= 5 and t in band],
    "success": [t for t, r in full.items() if int(r["succ"]) >= 6],
}
repo_of = {t: json.load(open(os.path.join(a.tree, t, "environment", "workspace", "metadata.json")))["repo_name"] for ts in strata.values() for t in ts}


def repo_aware_sample(ts, k, rng):
    """k tasks with per-repo counts proportional to the pool's repo mix (largest remainder), every repo present at least once."""
    by = collections.defaultdict(list)
    for t in sorted(ts): by[repo_of[t]].append(t)
    if k >= len(ts): return sorted(ts)
    quota = {r: len(v) * k / len(ts) for r, v in by.items()}
    alloc = {r: max(1, int(q)) for r, q in quota.items()}
    while sum(alloc.values()) > k:  # min-1 overshoot: take from the repo with the largest allocation
        r = max(alloc, key=lambda r: alloc[r]); alloc[r] -= 1
    for r in sorted(by, key=lambda r: quota[r] - alloc[r], reverse=True):
        if sum(alloc.values()) >= k: break
        if alloc[r] < len(by[r]): alloc[r] += 1
    out = []
    for r, v in by.items(): out += rng.sample(v, min(alloc[r], len(v)))
    return sorted(out)


rng = random.Random(a.seed); picked = []
for s, ts in strata.items():
    sel = repo_aware_sample(ts, a.per_stratum, rng)
    print("%-8s pool %4d -> %d" % (s, len(ts), len(sel))); picked += [(t, s) for t in sel]
with open(a.out + ".tsv", "w") as f, open(a.out + ".txt", "w") as g:
    f.write("task\trepo\tstratum\ttt60k_succ\ttt60k_scored\tmed_turns\n")
    for t, s in picked:
        m = json.load(open(os.path.join(a.tree, t, "environment", "workspace", "metadata.json")))
        r = rows[t]; f.write("\t".join([t, m["repo_name"], s, r["succ"], r["scored"], r["med_turns"]]) + "\n"); g.write(t + "\n")
by = collections.Counter((s, json.load(open(os.path.join(a.tree, t, "environment", "workspace", "metadata.json")))["repo_name"]) for t, s in picked)
for s in strata: print(s, dict(sorted((r, n) for (ss, r), n in by.items() if ss == s)))
print("wrote %s.tsv / .txt (%d tasks)" % (a.out, len(picked)))
