#!/usr/bin/env python3
"""tt_allowlist.py — allowlist + per-repo attrition for the TaskTrove-selection tree from envgate parts.

  tt_allowlist.py --map data/r2egym/overlap/tasktrove_v3_upstream_map.tsv --pairs tt_shared_pairs.txt \
      --parts <dir>... --out allowlist_r2egym_tt_v1.txt [--report report.md]

A task is allowed when (pristine reward == 0.0 AND oracle reward == 1.0) in the gate parts, OR — for tasks shared with
the raw set and not re-gated here — its raw twin passed the v3 gate (`tt_shared_pairs.txt`: path raw-task gate_v3).
Gate results win over the carried-over flag when both exist. Records without a parsable reward (timeouts, harness
errors) count as `harness` rows, not as fails, and the task is NOT allowed until re-gated.
"""
import argparse, collections, csv, glob, json, os
ap = argparse.ArgumentParser()
ap.add_argument("--map", required=True); ap.add_argument("--pairs", required=True); ap.add_argument("--parts", nargs="+", required=True)
ap.add_argument("--out", required=True); ap.add_argument("--report", default=None)
a = ap.parse_args()
repo = {r["path"]: r["repo"] for r in csv.DictReader(open(a.map), delimiter="\t")}
carry = {}
for line in open(a.pairs):
    p, raw, g = line.split(); carry[p] = (raw, g == "1")
gate = collections.defaultdict(dict)
for d in a.parts:
    for f in glob.glob(os.path.join(d, "*.json")):
        for line in open(f, errors="replace"):
            line = line.strip()
            if not line.startswith("{"): continue
            try: r = json.loads(line)
            except Exception: continue
            if "task" in r and "mode" in r: gate[r["task"]][r["mode"]] = r
rows = []
for p in sorted(repo):
    g = gate.get(p, {}); pr, orc = g.get("pristine"), g.get("oracle")
    if pr is not None or orc is not None:
        err = [m for m, r in (("pristine", pr), ("oracle", orc)) if r is None or r.get("reward") is None or r.get("error") or r.get("timed_out")]
        if err: status = "harness:" + ",".join(err)
        elif pr["reward"] == 0.0 and orc["reward"] == 1.0: status = "pass"
        else: status = "fail:p%s/o%s" % (pr["reward"], orc["reward"])
        src = "gate"
    elif p in carry:
        status = "pass" if carry[p][1] else "fail:v3"; src = "v3:" + carry[p][0]
    else:
        status = "ungated"; src = "-"
    rows.append((p, repo[p], status, src))
allowed = [p for p, _, s, _ in rows if s == "pass"]
open(a.out, "w").write("\n".join(allowed) + "\n")
tab = collections.defaultdict(collections.Counter)
for p, rp, s, src in rows: tab[rp][s.split(":")[0]] += 1
lines = ["repo | tasks | pass | fail | harness | ungated | pass %", "---|---|---|---|---|---|---"]
tot = collections.Counter()
for rp in sorted(tab, key=lambda r: -sum(tab[r].values())):
    c = tab[rp]; n = sum(c.values()); tot.update(c)
    lines.append("%s | %d | %d | %d | %d | %d | %.1f" % (rp, n, c["pass"], c["fail"], c["harness"], c["ungated"], 100.0 * c["pass"] / n))
n = sum(tot.values()); lines.append("**all** | %d | %d | %d | %d | %d | %.1f" % (n, tot["pass"], tot["fail"], tot["harness"], tot["ungated"], 100.0 * tot["pass"] / n))
print("\n".join(lines)); print("allowlist -> %s (%d tasks)" % (a.out, len(allowed)))
if a.report:
    with open(a.report, "w") as f:
        f.write("\n".join(lines) + "\n\n")
        for p, rp, s, src in rows:
            if s != "pass": f.write("%s\t%s\t%s\t%s\n" % (p, rp, s, src))
