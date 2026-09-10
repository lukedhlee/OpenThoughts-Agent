#!/usr/bin/env python3
"""v2val_compare.py [probe...] — base vs exported checkpoints on the tt-v2 val441 tree (idval 150 same-repo unseen, oodval 115
tornado+scrapy, heldout 176 = 160 never-solved + 16 always-solved), split by `experiments/tt_v2_split.tsv`. Per set: per-task pass
(mean of succ/scored), pass@8 (tasks with >= 1 success), successes/attempts, context deaths, median turns; paired deltas vs the first
probe with bootstrap 95 %; per-repo pass@8 for idval/oodval; heldout split never-solved vs always-solved. Python 3.9 / stdlib."""
import csv, os, random, sys
E = "/e/fscratch/reformo/lee27/experiments"
PROBES = sys.argv[1:] or ["snowball_v2val_base_fixed", "snowball_v2val_fulldist_s12"]
SHORT = {p: p.replace("snowball_v2val_", "").replace("_fixed", "") for p in PROBES}
split = {r["task"]: r for r in csv.DictReader(open(E + "/tt_v2_split.tsv"), delimiter="\t")}
tab = {}
for p in PROBES:
    f = E + "/%s/pass8_pass8_table.csv" % p
    if not os.path.exists(f): sys.exit("no table for %s" % p)
    tab[p] = {r["task"]: dict(succ=int(r["succ"]), scored=int(r["scored"]), ctx=int(r["ctx_exceeded"]), att=int(r["attempts"]),
                              turns=float(r["med_turns"]) if r["med_turns"] else None) for r in csv.DictReader(open(f))}
tasks = sorted(t for t in split if all(t in tab[p] and tab[p][t]["scored"] for p in PROBES))


def pr(p, t): return tab[p][t]["succ"] / tab[p][t]["scored"]


def boot(pairs, n=2000, seed=0):
    rng = random.Random(seed); ds = [b - a for a, b in pairs]
    bs = sorted(sum(ds[rng.randrange(len(ds))] for _ in ds) / len(ds) for _ in range(n)); return sum(ds) / len(ds), bs[int(.025 * n)], bs[int(.975 * n)]


def row(p, ts):
    z = [pr(p, t) for t in ts]
    return "%.3f | %d/%d | %d / %d | %d%% | %s" % (
        sum(z) / len(z), sum(1 for t in ts if tab[p][t]["succ"] >= 1), len(ts), sum(tab[p][t]["succ"] for t in ts),
        sum(tab[p][t]["scored"] for t in ts), round(100 * sum(tab[p][t]["ctx"] for t in ts) / max(1, sum(tab[p][t]["att"] for t in ts))),
        sorted(x for x in (tab[p][t]["turns"] for t in ts) if x is not None)[len(ts) // 2] if ts else "-")


base = PROBES[0]
print("# tt-v2 val441, %d tasks paired across %s\n" % (len(tasks), ", ".join(SHORT[p] for p in PROBES)))
sets = [("idval", [t for t in tasks if split[t]["set"] == "idval"]), ("oodval", [t for t in tasks if split[t]["set"] == "oodval"]),
        ("heldout never-solved", [t for t in tasks if split[t]["set"] == "heldout" and split[t]["tt60k_succ"] == "0"]),
        ("heldout always-solved", [t for t in tasks if split[t]["set"] == "heldout" and split[t]["tt60k_succ"] == "8"])]
for name, ts in sets:
    if not ts: continue
    print("## %s (n=%d)\n\n| checkpoint | per-task pass | pass@8 | successes / attempts | ctx deaths | median turns | paired Δ vs %s | 95 %% |" % (name, len(ts), SHORT[base]))
    print("|---|---|---|---|---|---|---|---|")
    for p in PROBES:
        if p == base: print("| %s | %s | – | – |" % (SHORT[p], row(p, ts)))
        else:
            m, lo, hi = boot([(pr(base, t), pr(p, t)) for t in ts]); print("| %s | %s | **%+.3f** | [%+.3f, %+.3f] |" % (SHORT[p], row(p, ts), m, lo, hi))
    print()
for name in ("idval", "oodval"):
    ts = [t for t in tasks if split[t]["set"] == name]; repos = sorted({split[t]["repo"] for t in ts})
    print("## %s by repo: pass@8 tasks (successes)\n\n| repo | n | %s |\n|---|---|%s" % (name, " | ".join(SHORT[p] for p in PROBES), "---|" * len(PROBES)))
    for r in repos:
        rs = [t for t in ts if split[t]["repo"] == r]
        print("| %s | %d | %s |" % (r, len(rs), " | ".join("%d (%d)" % (sum(1 for t in rs if tab[p][t]["succ"] >= 1), sum(tab[p][t]["succ"] for t in rs)) for p in PROBES)))
    print()
