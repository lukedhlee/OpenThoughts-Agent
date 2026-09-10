#!/usr/bin/env python3
"""turn_timing.py <artifact_store.img> [max_traj] — mount an inactive store ro and measure per-turn wall time from
agent/trajectory.json step timestamps: turns per trajectory, trajectory wall, per-turn delta percentiles, and whatever the
step 'metrics' carry (LLM latency split if present). Compares seat settings (528 vs 1536) on real traces."""
import json, pathlib, statistics, sys, time
from datetime import datetime
sys.path.insert(0, "/e/project1/transfernetx/lee27/code/OpenThoughts-Agent")
from hpc.artifact_store import mounted  # noqa: E402
image = pathlib.Path(sys.argv[1]); max_traj = int(sys.argv[2]) if len(sys.argv) > 2 else 600
def ts(s): return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()
def pct(xs, p): xs = sorted(xs); return xs[min(len(xs) - 1, int(p * len(xs)))] if xs else float("nan")
t0 = time.monotonic(); deltas = []; turns = []; walls = []; llm = []; nonllm = []; per_traj_llm_share = []; shown = False; n = 0; keys_seen = {}
with mounted(image, "ro") as root:
    tj = root / "trace_jobs"
    for att in sorted(tj.glob("*/attempts/*"))[: max_traj * 3]:
        tr = att / "agent" / "trajectory.json"; rj = att / "result.json"
        if not tr.is_file() or not rj.is_file(): continue
        try:
            r = json.loads(rj.read_text()); steps = json.loads(tr.read_text()).get("steps", [])
        except Exception: continue
        st = [s for s in steps if isinstance(s, dict) and s.get("timestamp")]
        if len(st) < 3: continue
        n += 1
        if n > max_traj: break
        times = [ts(s["timestamp"]) for s in st]
        d = [b - a for a, b in zip(times, times[1:])]
        deltas += d; turns.append(len(st))
        if r.get("started_at") and r.get("finished_at"): walls.append(ts(r["finished_at"]) - ts(r["started_at"]))
        m = st[1].get("metrics") or {}
        for k, v in m.items(): keys_seen.setdefault(k, str(v)[:60])
        if not shown:
            print("step keys:", sorted(st[1].keys()), "\nmetrics sample:", {k: str(v)[:60] for k, v in m.items()}, "\nsources:", sorted({s.get("source") for s in st}), flush=True); shown = True
        lat_key = next((k for k in m if any(t in k.lower() for t in ("latency", "duration", "elapsed", "time"))), None)
        if lat_key:
            l = [float((s.get("metrics") or {}).get(lat_key) or 0) for s in st[1:]]
            llm += l; nonllm += [max(0.0, dd - ll) for dd, ll in zip(d, l)]
            if sum(d) > 0: per_traj_llm_share.append(sum(l) / sum(d))
print(f"\n=== {image.parent.name}: {n} trajectories, {len(deltas)} turns ({time.monotonic()-t0:.0f}s)")
print(f"turns/traj  p50 {pct(turns,.5):.0f}  p90 {pct(turns,.9):.0f}  mean {statistics.mean(turns):.1f}")
if walls: print(f"traj wall s p50 {pct(walls,.5):.0f}  p90 {pct(walls,.9):.0f}  mean {statistics.mean(walls):.0f}")
print(f"turn delta s p10 {pct(deltas,.1):.1f} p50 {pct(deltas,.5):.1f}  p90 {pct(deltas,.9):.1f}  mean {statistics.mean(deltas):.1f}")
if llm: print(f"llm part s  p50 {pct(llm,.5):.1f} p90 {pct(llm,.9):.1f} mean {statistics.mean(llm):.1f} | non-llm s p50 {pct(nonllm,.5):.1f} p90 {pct(nonllm,.9):.1f} mean {statistics.mean(nonllm):.1f} | llm share p50 {pct(per_traj_llm_share,.5):.2f}")
print("metrics keys:", keys_seen)
