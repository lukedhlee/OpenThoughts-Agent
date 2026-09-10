#!/usr/bin/env python3
"""Side-by-side of N probe tables from pass8_table.py: python3 compare_probes.py label=prefix [label=prefix ...]
Per probe: attempts, scored, nulls, context-death share, trial pass, fully-sampled, strict-mixed, all-zero,
all-solved, mean pass@8, median turns; then pairwise overlap of the strict-mixed sets on tasks fully sampled in both."""
import csv, json, sys, itertools, statistics
probes = {}
for arg in sys.argv[1:]:
    label, prefix = arg.split("=", 1)
    rows = {r["task"]: r for r in csv.DictReader(open(prefix + "_pass8_table.csv"))}
    summ = json.load(open(prefix + "_summary.json"))
    probes[label] = (rows, summ)
hdr = ["probe", "tasks", "attempts", "scored", "nulls", "ctx_death", "trial_pass", "full", "mixed", "all0", "all1", "mean_p@8", "turns_med"]
print("\t".join(hdr))
for label, (rows, s) in probes.items():
    att = sum(int(r["attempts"]) for r in rows.values()); ctx = sum(int(r["ctx_exceeded"]) for r in rows.values())
    full = [r for r in rows.values() if r["full"] == "True"]
    mp = statistics.mean(float(r["pass_at_k"]) for r in rows.values() if r["pass_at_k"] not in ("", "None"))
    print("\t".join(str(x) for x in [label, len(rows), att, s["scored"], s["attempts"] - s["scored"], f"{ctx / max(1, att):.1%}",
          f"{s['successes'] / max(1, s['scored']):.1%}", len(full), s["strict_mixed"], s["all_zero"], s["all_solved"], f"{mp:.3f}", s.get("turns_median")]))
for a, b in itertools.combinations(probes, 2):
    ra, rb = probes[a][0], probes[b][0]
    both = [t for t in ra if t in rb and ra[t]["full"] == "True" and rb[t]["full"] == "True"]
    if not both: continue
    sa = {t: int(ra[t]["succ"]) for t in both}; sb = {t: int(rb[t]["succ"]) for t in both}
    ma = {t for t in both if 0 < sa[t] < int(ra[t]["scored"])}; mb = {t for t in both if 0 < sb[t] < int(rb[t]["scored"])}
    print(f"\n{a} vs {b}: {len(both)} tasks fully sampled in both")
    print(f"  strict-mixed: {a}={len(ma)} {b}={len(mb)} both={len(ma & mb)} only_{a}={len(ma - mb)} only_{b}={len(mb - ma)}")
    print(f"  successes on shared tasks: {a}={sum(sa.values())} {b}={sum(sb.values())}; tasks solved>=1: {a}={sum(1 for t in both if sa[t] > 0)} {b}={sum(1 for t in both if sb[t] > 0)}")
    hb = sum(1 for t in both if sb[t] > sa[t]); ha = sum(1 for t in both if sa[t] > sb[t])
    print(f"  per-task successes: {b} higher on {hb}, {a} higher on {ha}, tie {len(both) - ha - hb}")
