#!/usr/bin/env python3
"""null_series.py [--window-min 40] [--bucket-min 3] [--prefix snowball_pool60k --n 8]
Fast time series of attempt outcomes using attempts/*/exception.txt (first line) and verifier/reward.txt — no result.json loads."""
import glob, os, time, collections, sys, argparse
ap = argparse.ArgumentParser(); ap.add_argument("--window-min", type=float, default=40); ap.add_argument("--bucket-min", type=float, default=3)
ap.add_argument("--prefix", default="snowball_pool60k"); ap.add_argument("--n", type=int, default=8); a = ap.parse_args()
E = "/e/fscratch/reformo/lee27/experiments"; now = time.time(); b = collections.defaultdict(collections.Counter); bs = a.bucket_min * 60
for i in range(a.n):
    for att in glob.glob(f"{E}/{a.prefix}_s{i}/{a.prefix}_s{i}/trace_jobs/eval_sessions/*/*/attempts/*"):
        rp = os.path.join(att, "result.json")
        try: m = os.path.getmtime(rp)
        except FileNotFoundError: continue
        if now - m > a.window_min * 60: continue
        ex = os.path.join(att, "exception.txt"); k = time.strftime("%H:%M", time.localtime(m - (m % bs)))
        if os.path.exists(ex):
            try:
                lines = [l.strip() for l in open(ex, errors="replace").read().splitlines() if l.strip()]
                last = lines[-1] if lines else "?"
            except Exception: last = "?"
            cls = last.split(":")[0].split("(")[0].strip().split(".")[-1] or "?"
        else:
            cls = "ok" if os.path.exists(os.path.join(att, "verifier", "reward.txt")) else "none"
        b[k][cls] += 1
for k in sorted(b):
    c = b[k]; n = sum(c.values()); sc = c["ok"] + c["ContextLengthExceededError"]
    print(f"{k} n={n:4d} scored={sc:4d} ({sc/max(1,n):.0%}) timeout={c['BridgeOperationTimeoutError']:4d} reset={c['ConnectionResetError']:4d} tmux={c['RuntimeError']:4d} bridgeErr={c['BridgeOperationError']:3d} addtests={c['AddTestsDirError']:3d} other={n-sc-c['BridgeOperationTimeoutError']-c['ConnectionResetError']-c['RuntimeError']-c['BridgeOperationError']-c['AddTestsDirError']:3d}")
