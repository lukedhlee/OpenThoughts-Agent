#!/usr/bin/env python3
"""compare_pass8.py — same model, same recipe, same tasks, two environments: JSC apptainer (R2E-Gym per-task images) vs
Daytona (TaskTrove images + harness-run setup). Joins the per-shard pass@8 tables written by code/snowball/pass8_table.py
(columns task,attempts,scored,nulls,succ,ctx_exceeded,pass_at_k,full,strict_mixed,med_turns) on the TaskTrove task name.

  compare_pass8.py --map overlap/tasktrove_v3_upstream_map.tsv --jsc <pass8_pass8_table.csv>... --daytona <csv>... [--out report.md]

Reports, per repo and overall: tasks on each side, tasks shared, mean pass@8 on the SHARED tasks for both sides, the paired
difference (Daytona - JSC) with its mean and the share of tasks moving by at least 1/8, and each side's all-zero / all-solved /
context-exceeded rates. Only fully sampled tasks (attempts == k scored) enter the paired comparison.
"""
import argparse, collections, csv, statistics
ap = argparse.ArgumentParser()
ap.add_argument("--map", required=True); ap.add_argument("--jsc", nargs="+", required=True); ap.add_argument("--daytona", nargs="+", required=True)
ap.add_argument("--out", default=None); ap.add_argument("--k", type=int, default=8)
a = ap.parse_args()
repo = {r["path"]: r["repo"] for r in csv.DictReader(open(a.map), delimiter="\t")}

def load(files):
    out = {}
    for f in files:
        for r in csv.DictReader(open(f)):
            t = r["task"]
            scored = int(r["scored"] or 0)
            out[t] = {"p": float(r["pass_at_k"]) if r["pass_at_k"] not in ("", None) else 0.0, "succ": int(r["succ"] or 0), "scored": scored,
                      "ctx": int(r["ctx_exceeded"] or 0), "turns": float(r["med_turns"] or 0), "full": r["full"] == "True" and scored > 0}
    return out
J, D = load(a.jsc), load(a.daytona)
shared = sorted(set(J) & set(D)); paired = [t for t in shared if J[t]["full"] and D[t]["full"]]

def side_stats(S, tasks):
    if not tasks: return (0, 0.0, 0, 0, 0.0)
    return (len(tasks), statistics.mean(S[t]["p"] for t in tasks), sum(S[t]["succ"] == 0 for t in tasks),
            sum(S[t]["succ"] == S[t]["scored"] and S[t]["scored"] > 0 for t in tasks),
            sum(S[t]["ctx"] for t in tasks) / max(1, sum(S[t]["scored"] for t in tasks)))
rows = []; groups = collections.defaultdict(list)
for t in paired: groups[repo.get(t, "?")].append(t)
groups["**all**"] = list(paired)
lines = ["repo | JSC tasks | Daytona tasks | paired | pass@8 JSC | pass@8 Daytona | Δ mean | tasks moved ≥1/8 | all-zero J/D | all-solved J/D | ctx-exceeded J/D",
         "---|---|---|---|---|---|---|---|---|---|---"]
for g in sorted(groups, key=lambda r: (r == "**all**", -len(groups[r]))):
    ts = groups[g]
    nj = len(J) if g == "**all**" else sum(repo.get(t, "?") == g for t in J)
    nd = len(D) if g == "**all**" else sum(repo.get(t, "?") == g for t in D)
    if not ts: lines.append("%s | %d | %d | 0 | | | | | | |" % (g, nj, nd)); continue
    sj, sd = side_stats(J, ts), side_stats(D, ts)
    diffs = [D[t]["p"] - J[t]["p"] for t in ts]
    moved = sum(abs(x) >= 1.0 / a.k - 1e-9 for x in diffs)  # at least one success out of k changed
    lines.append("%s | %d | %d | %d | %.3f | %.3f | %+.3f | %d (%.0f%%) | %d/%d | %d/%d | %.2f/%.2f" % (
        g, nj, nd, len(ts), sj[1], sd[1], statistics.mean(diffs), moved, 100.0 * moved / len(ts), sj[2], sd[2], sj[3], sd[3], sj[4], sd[4]))
hdr = ["JSC side: %d tasks, %d fully sampled, mean pass@8 %.3f" % (len(J), sum(v["full"] for v in J.values()), statistics.mean(v["p"] for v in J.values()) if J else 0.0),
       "Daytona side: %d tasks, %d fully sampled, mean pass@8 %.3f" % (len(D), sum(v["full"] for v in D.values()), statistics.mean(v["p"] for v in D.values()) if D else 0.0),
       "shared %d, paired (fully sampled on both) %d" % (len(shared), len(paired)), ""]
txt = "\n".join(hdr + lines); print(txt)
if a.out:
    with open(a.out, "w") as f:
        f.write(txt + "\n\n## per-task (paired)\n\ntask | repo | pass@8 JSC | pass@8 Daytona | Δ\n---|---|---|---|---\n")
        for t in sorted(paired, key=lambda t: -(abs(D[t]["p"] - J[t]["p"]))):
            f.write("%s | %s | %.3f | %.3f | %+.3f\n" % (t, repo.get(t, "?"), J[t]["p"], D[t]["p"], D[t]["p"] - J[t]["p"]))
