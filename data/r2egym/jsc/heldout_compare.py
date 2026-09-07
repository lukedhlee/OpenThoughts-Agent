#!/usr/bin/env python3
"""heldout_compare.py [probe...] — base / arm 3 s24 / fulldist s36 on the tt-v2 held-out set (160 never-solved + 16 always-solved,
disjoint from every arm's training set). Per-task successes of 8, paired; how many never-solved tasks each checkpoint unlocks
(>= 1 of 8), how many always-solved tasks regress, per repo; trial pass, context deaths; paired deltas with bootstrap 95 %.
Python 3.9 / stdlib."""
import csv, glob, json, os, random, sys, collections
E = "/e/fscratch/reformo/lee27/experiments"
PROBES = sys.argv[1:] or ["snowball_heldout_base_fixed", "snowball_heldout_arm3_s24_fixed", "snowball_heldout_fulldist_s36_fixed"]
SHORT = {p: p.replace("snowball_heldout_", "").replace("_fixed", "") for p in PROBES}
split = {}
for r in csv.DictReader(open(E + "/tt_v2_split.tsv"), delimiter="\t"):
    if r["set"] == "heldout": split[r["task"]] = r
tab = {}
for p in PROBES:
    f = E + "/%s/pass8_pass8_table.csv" % p
    if not os.path.exists(f): sys.exit("no table for %s" % p)
    tab[p] = {r["task"]: dict(succ=int(r["succ"]), scored=int(r["scored"]), ctx=int(r["ctx_exceeded"]), att=int(r["attempts"]), turns=float(r["med_turns"]) if r["med_turns"] else None) for r in csv.DictReader(open(f))}
tasks = sorted(t for t in split if all(t in tab[p] and tab[p][t]["scored"] for p in PROBES))
zero = [t for t in tasks if split[t]["tt60k_succ"] == "0"]; solved = [t for t in tasks if split[t]["tt60k_succ"] == "8"]
print("# Held-out (never seen by any arm): %d tasks paired (%d never-solved, %d always-solved)\n" % (len(tasks), len(zero), len(solved)))


def boot(pairs, n=2000, seed=0):
    rng = random.Random(seed); ds = [b - a for a, b in pairs]
    bs = sorted(sum(ds[rng.randrange(len(ds))] for _ in ds) / len(ds) for _ in range(n)); return sum(ds) / len(ds), bs[int(.025 * n)], bs[int(.975 * n)]


def pr(p, t): return tab[p][t]["succ"] / tab[p][t]["scored"]


print("| checkpoint | per-task pass (never-solved) | never-solved tasks unlocked (>=1/8) | >=2/8 | successes / attempts | ctx deaths | always-solved: per-task pass | regressed (<8/8) | median turns |")
print("|---|---|---|---|---|---|---|---|---|")
for p in PROBES:
    z = [pr(p, t) for t in zero]; s = [pr(p, t) for t in solved]
    print("| %s | %.3f | %d | %d | %d / %d | %d%% | %.3f | %d | %s |" % (
        SHORT[p], sum(z) / len(z), sum(1 for t in zero if tab[p][t]["succ"] >= 1), sum(1 for t in zero if tab[p][t]["succ"] >= 2),
        sum(tab[p][t]["succ"] for t in zero), sum(tab[p][t]["scored"] for t in zero),
        round(100 * sum(tab[p][t]["ctx"] for t in zero) / sum(tab[p][t]["att"] for t in zero)),
        sum(s) / len(s) if s else 0, sum(1 for t in solved if tab[p][t]["succ"] < 8),
        sorted(x for x in (tab[p][t]["turns"] for t in tasks) if x is not None)[len(tasks) // 2]))
base = PROBES[0]
print("\n| paired delta vs %s (never-solved) | mean | 95 %% |" % SHORT[base]); print("|---|---|---|")
for p in PROBES[1:]:
    m, lo, hi = boot([(pr(base, t), pr(p, t)) for t in zero]); print("| %s | **%+.3f** | [%+.3f, %+.3f] |" % (SHORT[p], m, lo, hi))
print("\n## Never-solved tasks unlocked, by repo (tasks with >= 1 success of 8)\n")
repos = sorted({split[t]["repo"] for t in zero})
print("| repo | n | " + " | ".join(SHORT[p] for p in PROBES) + " |"); print("|---|---|" + "---|" * len(PROBES))
for r in repos:
    ts = [t for t in zero if split[t]["repo"] == r]
    print("| %s | %d | %s |" % (r, len(ts), " | ".join("%d (%d succ)" % (sum(1 for t in ts if tab[p][t]["succ"] >= 1), sum(tab[p][t]["succ"] for t in ts)) for p in PROBES)))
print("\n## Tasks any checkpoint unlocked (successes of 8: %s)\n" % " / ".join(SHORT[p] for p in PROBES))
rows = [(t, split[t]["repo"], [tab[p][t]["succ"] for p in PROBES]) for t in zero if any(tab[p][t]["succ"] for p in PROBES)]
rows.sort(key=lambda x: -max(x[2]))
for t, r, s in rows: print("- %s %s: %s" % (t, r, " / ".join(map(str, s))))
print("\n## Always-solved tasks (successes of 8)\n")
for t in solved: print("- %s %s: %s" % (t, split[t]["repo"], " / ".join(str(tab[p][t]["succ"]) for p in PROBES)))
