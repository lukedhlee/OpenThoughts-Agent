#!/usr/bin/env python3
"""pass8_fast.py <trace_jobs> --out <prefix> — the columns v2rest_compare.py needs, read from result.json only.
pass8_table.py also opens every attempt's trajectory.json for the turn median, which is what makes it slow on a
1,840-task probe (~15k extra GPFS reads); this drops med_turns and uses os.scandir instead of glob. Same scoring
rules: scored = attempts with a verifier reward (nulls are not failures), succ = reward >= 1."""
import csv, json, os, sys
tj = sys.argv[1]; out = sys.argv[sys.argv.index("--out") + 1]
# --session <name>: read one eval session only. A probe whose job is not cancelled the moment its eval finishes starts a
# SECOND eval pass into eval_step1, and mixing the two would give some tasks 9-16 attempts (2026-09-09).
ONLY = sys.argv[sys.argv.index("--session") + 1] if "--session" in sys.argv else None
rows = {}
sess = os.path.join(tj, "eval_sessions")
for s in os.scandir(sess):
    if not s.is_dir() or (ONLY and s.name != ONLY): continue
    for trial in os.scandir(s.path):
        if not trial.is_dir(): continue
        ad = os.path.join(trial.path, "attempts")
        if not os.path.isdir(ad): continue
        for att in os.scandir(ad):
            f = os.path.join(att.path, "result.json")
            try:
                d = json.load(open(f))
            except Exception:
                continue
            task = d.get("task_name") or trial.name.split("__")[0]
            v = d.get("verifier_result")
            r = (v.get("rewards") or {}).get("reward") if isinstance(v, dict) else None
            e = (d.get("exception_info") or {}).get("exception_type")
            rows.setdefault(task, []).append((r, e))
table = []
for t in sorted(rows):
    at = rows[t]; scored = [x for x in at if x[0] is not None]
    table.append({"task": t, "attempts": len(at), "scored": len(scored),
                  "succ": sum(1 for x in scored if x[0] and x[0] >= 1.0),
                  "ctx_exceeded": sum(1 for x in at if x[1] == "ContextLengthExceededError"), "med_turns": ""})
with open(out + "_pass8_table.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(table[0].keys())); w.writeheader(); w.writerows(table)
print("tasks=%d attempts=%d scored=%d fully_sampled=%d" % (
    len(table), sum(r["attempts"] for r in table), sum(r["scored"] for r in table),
    sum(1 for r in table if r["scored"] >= 8)))
