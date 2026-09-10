#!/usr/bin/env python3
"""v2rest_compare.py [--curriculum <probe>] [probe...] — base vs exported checkpoints on the tt-v2 REST tree: the 1,840 TaskTrove
tasks outside the 993-task learnable band and outside the 176-task heldout set (1,701 never solved by the base on tt60k, 139 no-op
wins), split by `experiments/tt_v2_rest.tsv`. Same shape as v2val_compare.py: per stratum the per-task pass (mean succ/scored),
pass@8 (tasks with >= 1 success), successes/attempts, context deaths, median turns; paired deltas vs the first probe with a bootstrap
95 % interval; per-repo pass@8. `--curriculum <probe>` additionally writes the next arm's candidate set — the tasks that probe solves
1-7 of 8, i.e. the ones that have become learnable (a task solved 0 of 8 gives no gradient, 8 of 8 gives none either).
Python 3.9 / stdlib; runs on the Jupiter login node."""
import csv, os, random, sys
E = "/e/fscratch/reformo/lee27/experiments"
args = sys.argv[1:]
CURR = None
if "--curriculum" in args:
    i = args.index("--curriculum"); CURR = args[i + 1]; del args[i:i + 2]
# --min-scored N: only tasks with at least N of the 8 samples scored in EVERY probe. Default 1 (any overlap); pass 8 for an
# interim read off a running probe, where a partially sampled task would otherwise contribute a noisy per-task rate.
MINS = 1
if "--min-scored" in args:
    i = args.index("--min-scored"); MINS = int(args[i + 1]); del args[i:i + 2]
PROBES = args or ["snowball_v2rest_base_fixed", "snowball_v2rest_fulldist_s12", "snowball_v2rest_fulldist_s24"]
SHORT = {p: p.replace("snowball_v2rest_", "").replace("_fixed", "") for p in PROBES}
split = {r["task"]: r for r in csv.DictReader(open(E + "/tt_v2_rest.tsv"), delimiter="\t")}
tab = {}
for p in PROBES:
    f = E + "/%s/pass8_pass8_table.csv" % p
    if not os.path.exists(f): sys.exit("no table for %s" % p)
    tab[p] = {r["task"]: dict(succ=int(r["succ"]), scored=int(r["scored"]), ctx=int(r["ctx_exceeded"]), att=int(r["attempts"]),
                              turns=float(r["med_turns"]) if r["med_turns"] else None) for r in csv.DictReader(open(f))}
tasks = sorted(t for t in split if all(t in tab[p] and tab[p][t]["scored"] >= MINS for p in PROBES))


def pr(p, t): return tab[p][t]["succ"] / tab[p][t]["scored"]


def boot(pairs, n=2000, seed=0):
    rng = random.Random(seed); ds = [b - a for a, b in pairs]
    bs = sorted(sum(ds[rng.randrange(len(ds))] for _ in ds) / len(ds) for _ in range(n)); return sum(ds) / len(ds), bs[int(.025 * n)], bs[int(.975 * n)]


def row(p, ts):
    z = [pr(p, t) for t in ts]
    turns = sorted(x for x in (tab[p][t]["turns"] for t in ts) if x is not None)
    return "%.3f | %d/%d | %d / %d | %d%% | %s" % (
        sum(z) / len(z), sum(1 for t in ts if tab[p][t]["succ"] >= 1), len(ts), sum(tab[p][t]["succ"] for t in ts),
        sum(tab[p][t]["scored"] for t in ts), round(100 * sum(tab[p][t]["ctx"] for t in ts) / max(1, sum(tab[p][t]["att"] for t in ts))),
        "%.1f" % turns[len(turns) // 2] if turns else "-")


base = PROBES[0]
print("# tt-v2 rest (%d tasks paired across %s)\n" % (len(tasks), ", ".join(SHORT[p] for p in PROBES)))
sets = [("never solved on tt60k", [t for t in tasks if split[t]["stratum"] == "zero"]),
        ("no-op wins (1-7 of 8 on tt60k, dropped from the band)", [t for t in tasks if split[t]["stratum"] == "noop_win"]),
        ("all rest tasks", tasks)]
for name, ts in sets:
    if not ts: continue
    print("## %s (n=%d)\n\n| checkpoint | per-task pass | pass@8 | successes / attempts | ctx deaths | median turns | paired Δ vs %s | 95 %% |" % (name, len(ts), SHORT[base]))
    print("|---|---|---|---|---|---|---|---|")
    for p in PROBES:
        if p == base: print("| %s | %s | – | – |" % (SHORT[p], row(p, ts)))
        else:
            m, lo, hi = boot([(pr(base, t), pr(p, t)) for t in ts]); print("| %s | %s | **%+.3f** | [%+.3f, %+.3f] |" % (SHORT[p], row(p, ts), m, lo, hi))
    print()
ts = [t for t in tasks if split[t]["stratum"] == "zero"]; repos = sorted({split[t]["repo"] for t in ts})
print("## never-solved by repo: tasks unlocked (>= 1 of 8) with total successes\n\n| repo | n | %s |\n|---|---|%s" % (" | ".join(SHORT[p] for p in PROBES), "---|" * len(PROBES)))
for r in repos:
    rs = [t for t in ts if split[t]["repo"] == r]
    print("| %s | %d | %s |" % (r, len(rs), " | ".join("%d (%d)" % (sum(1 for t in rs if tab[p][t]["succ"] >= 1), sum(tab[p][t]["succ"] for t in rs)) for p in PROBES)))
print()
if CURR:
    if CURR not in tab: sys.exit("--curriculum %s is not among the probes read" % CURR)
    cand = [t for t in tasks if 1 <= tab[CURR][t]["succ"] <= tab[CURR][t]["scored"] - 1]
    out = E + "/tt_v2_curriculum_%s.txt" % SHORT[CURR].replace("/", "_")  # a probe passed as "<name>/interim" would otherwise name a directory
    open(out, "w").write("\n".join(cand) + "\n")
    byrepo = {}
    for t in cand: byrepo[split[t]["repo"]] = byrepo.get(split[t]["repo"], 0) + 1
    print("## curriculum candidates from %s: tasks solved 1-7 of 8 (n=%d)\n" % (SHORT[CURR], len(cand)))
    print("repos: %s" % ", ".join("%s %d" % kv for kv in sorted(byrepo.items(), key=lambda kv: -kv[1])))
    print("of which never solved on tt60k: %d" % sum(1 for t in cand if split[t]["stratum"] == "zero"))
    print("list -> %s" % out)
