#!/usr/bin/env python3
"""Merge per-task pass@K tables from several same-config probes: python3 merge_pass8.py --out <prefix> <prefix1> <prefix2> ...
For each task take the first source (in argument order) whose row is fully sampled; else the row with the most scored attempts.
Writes <out>_pass8_table.csv, _strict_mixed.txt, _summary.json (same schema as pass8_table.py)."""
import csv, json, sys, collections
args = sys.argv[1:]; out = args[args.index("--out") + 1]; srcs = [a for a in args if a != "--out" and a != out]
best = {}; origin = {}
for s in srcs:
    for r in csv.DictReader(open(s + "_pass8_table.csv")):
        t = r["task"]; full = r["full"] == "True"; sc = int(r["scored"])
        cur = best.get(t)
        if cur is None or (full and cur["full"] != "True") or (not full and cur["full"] != "True" and sc > int(cur["scored"])):
            best[t] = r; origin[t] = s
rows = [best[t] for t in sorted(best)]
with open(out + "_pass8_table.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) + ["source"]); w.writeheader()
    for r in rows: w.writerow({**r, "source": origin[r["task"]]})
full = [r for r in rows if r["full"] == "True"]; mixed = [r for r in full if r["strict_mixed"] == "True"]
zero = [r for r in full if int(r["succ"]) == 0]; solved = [r for r in full if int(r["succ"]) == int(r["scored"])]
att = sum(int(r["attempts"]) for r in rows); sc = sum(int(r["scored"]) for r in rows); su = sum(int(r["succ"]) for r in rows)
summ = {"tasks": len(rows), "attempts": att, "scored": sc, "successes": su, "fully_sampled": len(full), "strict_mixed": len(mixed),
        "all_zero": len(zero), "all_solved": len(solved), "tasks_ge1_success": sum(1 for r in rows if int(r["succ"]) > 0),
        "pass_at_8_any_success_over_tasks": sum(1 for r in rows if int(r["succ"]) > 0) / max(1, len(rows)),
        "sources": dict(collections.Counter(origin.values()))}
open(out + "_strict_mixed.txt", "w").write("\n".join(r["task"] for r in mixed) + "\n")
json.dump(summ, open(out + "_summary.json", "w"), indent=1); print(json.dumps(summ, indent=1))
