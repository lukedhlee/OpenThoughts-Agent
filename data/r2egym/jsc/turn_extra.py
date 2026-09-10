#!/usr/bin/env python3
"""turn_extra.py <artifact_store.img|trace_root> [max_traj] — summarize the measured per-turn split written by harbor 26c307fc+
into each agent step's metrics.extra: llm_seconds, exec_seconds, sleep_seconds, turn_seconds, and the remainder
(turn - llm - exec). Works on an inactive image (mounted ro) or directly on a trace root directory."""
import json, pathlib, statistics, sys, time
sys.path.insert(0, "/e/project1/transfernetx/lee27/code/OpenThoughts-Agent")
src = pathlib.Path(sys.argv[1]); max_traj = int(sys.argv[2]) if len(sys.argv) > 2 else 400
def pct(xs, p): xs = sorted(xs); return xs[min(len(xs) - 1, int(p * len(xs)))] if xs else float("nan")
def scan(tj):
    rows = []; n = 0
    for att in sorted(tj.glob("*/attempts/*")):
        tr = att / "agent" / "trajectory.json"
        if not tr.is_file(): continue
        try: steps = json.loads(tr.read_text()).get("steps", [])
        except Exception: continue
        ex = [((s.get("metrics") or {}).get("extra") or {}) for s in steps if isinstance(s, dict) and s.get("source") == "agent"]
        ex = [e for e in ex if "llm_seconds" in e]
        if not ex: continue
        n += 1
        if n > max_traj: break
        rows += ex
    return n, rows
t0 = time.monotonic()
if src.is_dir(): n, rows = scan(src)
else:
    from hpc.artifact_store import mounted
    with mounted(src, "ro") as root: n, rows = scan(root / "trace_jobs")
if not rows: print("no timing extras found (harbor < 26c307fc?)"); sys.exit()
def col(k): return [float(r[k]) for r in rows if k in r]
llm, ex, sl, tot = col("llm_seconds"), col("exec_seconds"), col("sleep_seconds"), col("turn_seconds")
rest = [float(r["turn_seconds"]) - float(r["llm_seconds"]) - float(r.get("exec_seconds", 0)) for r in rows]
print(f"=== {src.name if src.is_dir() else src.parent.name}: {n} traj, {len(rows)} agent turns ({time.monotonic()-t0:.0f}s)")
for name, xs in (("turn", tot), ("llm", llm), ("exec", ex), ("sleep(in exec)", sl), ("rest", rest)):
    if xs: print(f"{name:15s} p50 {pct(xs,.5):6.1f}  p90 {pct(xs,.9):6.1f}  mean {statistics.mean(xs):6.1f}  share {sum(xs)/max(sum(tot),1e-9):5.2f}")
