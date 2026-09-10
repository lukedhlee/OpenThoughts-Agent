#!/usr/bin/env python3
"""pool_index.py — one row per attempt across the pool trees (pass 1 + r1 partial + r1): task, attempt dir, pass, reward,
exception class, finish time; plus per-task summary with first-8-scored succ (same rule as merge_scored_union.py) and
pass@1/4/8 (unbiased estimator from n scored, c successes). Writes <out>/pool_attempts.csv, <out>/pool_tasks.csv,
<out>/passk_by_succ.json. Reads only small files."""
import glob, os, csv, json, collections, sys, math
E = "/e/fscratch/reformo/lee27/experiments"; out = sys.argv[1] if len(sys.argv) > 1 else f"{E}/snowball_pool60k_analysis"
os.makedirs(out, exist_ok=True)
trees = []
for i in range(8):
    trees += [(f"pass1", f"{E}/snowball_pool60k_s{i}/snowball_pool60k_s{i}/trace_jobs"),
              (f"r1partial", f"{E}/snowball_pool60k_r1_s{i}/run1_partial/trace_jobs"),
              (f"r1", f"{E}/snowball_pool60k_r1_s{i}/snowball_pool60k_r1_s{i}/trace_jobs")]
rows = []
for pas, tj in trees:
    for att in glob.glob(f"{tj}/eval_sessions/*/*/attempts/*"):
        task = os.path.basename(os.path.dirname(os.path.dirname(att))).split("__")[0]
        rp = os.path.join(att, "result.json")
        if not os.path.exists(rp): continue
        rw = os.path.join(att, "verifier", "reward.txt"); ex = os.path.join(att, "exception.txt"); cls = ""; r = ""
        if os.path.exists(rw):
            try: r = float(open(rw).read().strip())
            except Exception: cls = "bad_reward_file"
        else:
            cls = "null"
            if os.path.exists(ex):
                lines = [l.strip() for l in open(ex, errors="replace").read().splitlines() if l.strip()]
                if lines: cls = lines[-1].split(":")[0].split("(")[0].strip().split(".")[-1] or "null"
        rows.append(dict(task=task, pass_=pas, attempt=att, reward=r, exc=cls, mtime=os.path.getmtime(rp),
                         has_traj=os.path.exists(os.path.join(att, "agent", "trajectory.json"))))
rows.sort(key=lambda x: (x["task"], x["mtime"]))
with open(f"{out}/pool_attempts.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
def pass_at_k(n, c, k):
    if n - c < k: return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)
per = collections.defaultdict(list)
for r in rows:
    if r["reward"] != "": per[r["task"]].append(r)
tasks = []
for t, lst in sorted(per.items()):
    sc = lst[:8]; n = len(sc); c = sum(1 for r in sc if r["reward"] >= 1.0)
    tasks.append(dict(task=t, scored=n, succ=c, full=n >= 8, pass1=round(c / n, 4), pass4=round(pass_at_k(n, c, 4), 4) if n >= 4 else "",
                      pass8=round(pass_at_k(n, c, 8), 4) if n >= 8 else "", gap81=round(pass_at_k(n, c, 8) - c / n, 4) if n >= 8 else "",
                      winners=";".join(r["attempt"] for r in sc if r["reward"] >= 1.0), losers=";".join(r["attempt"] for r in sc if r["reward"] < 1.0)))
with open(f"{out}/pool_tasks.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(tasks[0].keys())); w.writeheader(); w.writerows(tasks)
full = [t for t in tasks if t["full"]]
byc = collections.Counter(t["succ"] for t in full)
curve = {c: dict(tasks=byc[c], pass1=round(c / 8, 4), pass4=round(pass_at_k(8, c, 4), 4), pass8=1.0 if c else 0.0, gap8_1=round((1.0 if c else 0.0) - c / 8, 4)) for c in range(9)}
agg = {k: round(sum(pass_at_k(8, t["succ"], k) for t in full) / len(full), 4) for k in (1, 2, 4, 8)}
summ = dict(attempts=len(rows), scored=sum(1 for r in rows if r["reward"] != ""), tasks=len(tasks), fully_sampled=len(full),
            pool_pass_at_k=agg, by_succ=curve, nulls=dict(collections.Counter(r["exc"] for r in rows if r["exc"]).most_common()))
json.dump(summ, open(f"{out}/passk_by_succ.json", "w"), indent=1); print(json.dumps(summ, indent=1))
