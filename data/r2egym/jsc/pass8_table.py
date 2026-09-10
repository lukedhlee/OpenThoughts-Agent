#!/usr/bin/env python3
"""Per-task pass@K table + integrity report from a probe's trace_jobs (runbook pass8-probe.md).
Usage: pass8_table.py <trace_jobs dir> [--k 8] [--out prefix]
Nulls (exception before verifier) are NOT failures; strict-mixed only from fully sampled tasks."""
import argparse, collections, glob, json, os, statistics, csv
ap = argparse.ArgumentParser(); ap.add_argument("tj"); ap.add_argument("--k", type=int, default=8); ap.add_argument("--out", default=None)
a = ap.parse_args()
rows = collections.defaultdict(list); exc_all = collections.Counter(); turns = []; outtok = []
for p in glob.glob(f"{a.tj}/eval_sessions/*/*/attempts/*/result.json"):
    try: d = json.load(open(p))
    except Exception: rows["__unreadable__"].append(None); continue
    task = d.get("task_name") or os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(p))))
    v = d.get("verifier_result"); r = (v.get("rewards") or {}).get("reward") if isinstance(v, dict) else None
    e = (d.get("exception_info") or {}).get("exception_type"); ar = d.get("agent_result") or {}
    tp = os.path.join(os.path.dirname(p), "agent", "trajectory.json")
    nt = None
    if os.path.exists(tp):
        try: nt = len([s for s in json.load(open(tp)).get("steps", []) if s.get("source") == "agent"])
        except Exception: pass
    if nt: turns.append(nt)
    if ar.get("n_output_tokens"): outtok.append(ar["n_output_tokens"])
    exc_all[e] += 1
    rows[task].append({"reward": r, "exc": e, "turns": nt, "out_tok": ar.get("n_output_tokens")})
tasks = sorted(t for t in rows if t != "__unreadable__")
table = []
for t in tasks:
    at = rows[t]; scored = [x for x in at if x["reward"] is not None]
    succ = sum(1 for x in scored if x["reward"] and x["reward"] >= 1.0)
    nulls = len(at) - len(scored)
    ctx = sum(1 for x in at if x["exc"] == "ContextLengthExceededError")
    full = len(scored) >= a.k
    table.append({"task": t, "attempts": len(at), "scored": len(scored), "nulls": nulls, "succ": succ,
                  "ctx_exceeded": ctx, "pass_at_k": (succ / len(scored)) if scored else None, "full": full,
                  "strict_mixed": full and 0 < succ < len(scored),
                  "med_turns": statistics.median([x["turns"] for x in at if x["turns"]]) if any(x["turns"] for x in at) else None})
n_attempts = sum(r["attempts"] for r in table); n_scored = sum(r["scored"] for r in table); n_succ = sum(r["succ"] for r in table)
full = [r for r in table if r["full"]]; mixed = [r for r in table if r["strict_mixed"]]
zero = [r for r in full if r["succ"] == 0]; solved_all = [r for r in full if r["succ"] == r["scored"]]
print(f"tasks={len(table)} attempts={n_attempts} scored={n_scored} nulls={n_attempts-n_scored} successes={n_succ}")
print(f"trial-level pass rate (scored) = {n_succ/max(1,n_scored):.3f}")
print(f"fully-sampled tasks (>= {a.k} scored) = {len(full)}: strict-mixed={len(mixed)} all-zero={len(zero)} all-solved={len(solved_all)}")
print(f"tasks with >=1 success = {sum(1 for r in table if r['succ']>0)}; mean per-task pass@{a.k} (scored) = {statistics.mean([r['pass_at_k'] for r in table if r['pass_at_k'] is not None]):.3f}")
print("exceptions:", dict(exc_all.most_common()))
if turns: print(f"agent turns med/p90/max = {statistics.median(turns)}/{sorted(turns)[int(0.9*len(turns))]}/{max(turns)}; out_tok med = {statistics.median(outtok) if outtok else None}")
if a.out:
    with open(a.out + "_pass8_table.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0].keys())); w.writeheader(); w.writerows(table)
    open(a.out + "_strict_mixed.txt", "w").write("\n".join(r["task"] for r in mixed) + "\n")
    json.dump({"tasks": len(table), "attempts": n_attempts, "scored": n_scored, "successes": n_succ, "fully_sampled": len(full),
               "strict_mixed": len(mixed), "all_zero": len(zero), "all_solved": len(solved_all), "exceptions": dict(exc_all),
               "turns_median": statistics.median(turns) if turns else None}, open(a.out + "_summary.json", "w"), indent=1)
    print("wrote", a.out + "_{pass8_table.csv,strict_mixed.txt,summary.json}")
