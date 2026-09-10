#!/usr/bin/env python3
"""merge_scored_union.py --trace-jobs <dir>... --out <prefix> [--k 8]
Union of SCORED attempts per task across several probe/pool trace_jobs trees (e.g. pass 1 + residual waves) under the
same harness contract. Per task: the first K scored attempts by finish time count; nulls never count. Writes
<prefix>_pass8_table.csv (same columns as pass8_table.py), _strict_mixed.txt, _summary.json. Reads only the small
files (verifier/reward.txt, exception.txt mtime) — no result.json loads — so it is fast on live-sized trees."""
import argparse, glob, os, json, csv, collections
ap = argparse.ArgumentParser(); ap.add_argument("--trace-jobs", nargs="+", required=True); ap.add_argument("--out", required=True); ap.add_argument("--k", type=int, default=8)
a = ap.parse_args(); per = collections.defaultdict(list); exc = collections.Counter(); nattempts = 0
for tj in a.trace_jobs:
    for att in glob.glob(f"{tj}/eval_sessions/*/*/attempts/*"):
        task = os.path.basename(os.path.dirname(os.path.dirname(att))).split("__")[0]
        rp = os.path.join(att, "result.json")
        if not os.path.exists(rp): exc["unwritten"] += 1; continue
        nattempts += 1
        rw = os.path.join(att, "verifier", "reward.txt"); ex = os.path.join(att, "exception.txt")
        if os.path.exists(rw):
            try: r = float(open(rw).read().strip())
            except Exception: exc["bad_reward_file"] += 1; continue
            per[task].append((os.path.getmtime(rp), r))
        else:
            cls = "null"
            if os.path.exists(ex):
                lines = [l.strip() for l in open(ex, errors="replace").read().splitlines() if l.strip()]
                if lines: cls = lines[-1].split(":")[0].split("(")[0].strip().split(".")[-1] or "null"
            exc[cls] += 1
rows = []; mixed = []; hist = collections.Counter()
for task in sorted(per):
    sc = sorted(per[task])[:a.k]; succ = sum(1 for _, r in sc if r >= 1.0); full = len(sc) >= a.k
    sm = full and 0 < succ < a.k
    rows.append(dict(task=task, attempts=len(per[task]), scored=len(sc), nulls=0, succ=succ, ctx_exceeded="", pass_at_k=round(succ / max(1, len(sc)), 4), full=full, strict_mixed=sm, med_turns=""))
    if sm: mixed.append(task)
    if full: hist[succ] += 1
with open(f"{a.out}_pass8_table.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["task"]); w.writeheader(); [w.writerow(r) for r in rows]
open(f"{a.out}_strict_mixed.txt", "w").write("\n".join(mixed) + "\n")
full_n = sum(1 for r in rows if r["full"]); summ = dict(sources=a.trace_jobs, tasks=len(rows), attempts_with_result=nattempts, scored_attempts=sum(len(v) for v in per.values()),
    fully_sampled=full_n, strict_mixed=len(mixed), all_zero=hist[0], all_solved=hist[a.k], success_hist=dict(sorted(hist.items())), exceptions=dict(exc.most_common()))
json.dump(summ, open(f"{a.out}_summary.json", "w"), indent=2); print(json.dumps(summ, indent=1))
