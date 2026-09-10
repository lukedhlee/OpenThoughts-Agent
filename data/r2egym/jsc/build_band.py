#!/usr/bin/env python3
"""build_band.py --tables <pass8_table.csv>... --out <dir> [--src tasks/r2egym-raw-v3-train] [--task-dir <new task dir>]
Concatenate per-shard pass@K tables into one pool table, derive the strict-mixed band (fully sampled, 0<succ<K),
the filtered-out set (all-zero / all-solved, with baseline pass@K) and the residual (not fully sampled), and
optionally materialise the band as a task dir (copies of <src>/<task>). Writes <out>/pool_table.csv,
band_strict_mixed.txt, filtered_out.csv, residual.txt, summary.json."""
import argparse, csv, json, os, shutil, collections
ap = argparse.ArgumentParser(); ap.add_argument("--tables", nargs="+", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--src", default="/e/fscratch/reformo/lee27/tasks/r2egym-raw-v3-train"); ap.add_argument("--task-dir", default=None)
ap.add_argument("--k", type=int, default=8); a = ap.parse_args()
rows = {}; dup = 0
for t in a.tables:
    for r in csv.DictReader(open(t)):
        if r["task"] in rows: dup += 1; continue   # first table wins per task (tables are disjoint shards)
        r["source"] = t; rows[r["task"]] = r
os.makedirs(a.out, exist_ok=True)
band, allzero, allsolved, resid = [], [], [], []
for task, r in sorted(rows.items()):
    full = r["full"] == "True"; succ = int(r["succ"]); scored = int(r["scored"])
    if not full: resid.append(task); continue
    if 0 < succ < a.k: band.append(task)
    elif succ == 0: allzero.append(task)
    else: allsolved.append(task)
hdr = list(next(iter(rows.values())).keys())
with open(f"{a.out}/pool_table.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=hdr); w.writeheader(); [w.writerow(rows[t]) for t in sorted(rows)]
open(f"{a.out}/band_strict_mixed.txt", "w").write("\n".join(band) + "\n")
open(f"{a.out}/residual.txt", "w").write("\n".join(resid) + "\n")
with open(f"{a.out}/filtered_out.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["task", "class", "succ", "scored", "pass_at_k", "med_turns"])
    for t in allzero: w.writerow([t, "all_zero", rows[t]["succ"], rows[t]["scored"], rows[t]["pass_at_k"], rows[t]["med_turns"]])
    for t in allsolved: w.writerow([t, "all_solved", rows[t]["succ"], rows[t]["scored"], rows[t]["pass_at_k"], rows[t]["med_turns"]])
hist = collections.Counter(int(rows[t]["succ"]) for t in rows if rows[t]["full"] == "True")
summ = {"tables": a.tables, "tasks": len(rows), "duplicates_skipped": dup, "fully_sampled": len(rows) - len(resid),
        "strict_mixed": len(band), "all_zero": len(allzero), "all_solved": len(allsolved), "residual": len(resid),
        "success_hist": dict(sorted(hist.items())), "band_share_of_full": round(len(band) / max(1, len(rows) - len(resid)), 4)}
copied = 0
if a.task_dir:
    os.makedirs(a.task_dir, exist_ok=True)
    for t in band:
        s, d = f"{a.src}/{t}", f"{a.task_dir}/{t}"
        assert os.path.isdir(s), s
        if not os.path.isdir(d): shutil.copytree(s, d); copied += 1
    summ["task_dir"] = a.task_dir; summ["tasks_in_task_dir"] = len(os.listdir(a.task_dir)); summ["copied_now"] = copied
json.dump(summ, open(f"{a.out}/summary.json", "w"), indent=2); print(json.dumps(summ, indent=1))
