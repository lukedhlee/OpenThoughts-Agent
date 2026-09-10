#!/usr/bin/env python3
"""turn_split.py <artifact_store.img> [max_traj] — per-turn wall (from step timestamps) vs the agent's own declared command
durations (sum of `duration` in the turn's commands JSON) and blocking commands; the remainder is LLM decode + harness/bridge
overhead. Tells how much of the non-LLM time is the agent's chosen sleeps."""
import json, pathlib, re, statistics, sys, time
from datetime import datetime
sys.path.insert(0, "/e/project1/transfernetx/lee27/code/OpenThoughts-Agent")
from hpc.artifact_store import mounted  # noqa: E402
image = pathlib.Path(sys.argv[1]); max_traj = int(sys.argv[2]) if len(sys.argv) > 2 else 400
def ts(s): return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()
def pct(xs, p): xs = sorted(xs); return xs[min(len(xs) - 1, int(p * len(xs)))] if xs else float("nan")
def _f(v):
    try: return float(v or 0)
    except Exception: return 0.0
def cmds(msg):
    if isinstance(msg, dict): msg = json.dumps(msg)
    m = re.search(r"\{.*\}", str(msg), re.S)
    try: return json.loads(m.group(0)).get("commands", []) if m else []
    except Exception: return []
t0 = time.monotonic(); rows = []; n = 0
with mounted(image, "ro") as root:
    for att in sorted((root / "trace_jobs").glob("*/attempts/*")):
        tr = att / "agent" / "trajectory.json"
        if not tr.is_file(): continue
        try: steps = [s for s in json.loads(tr.read_text()).get("steps", []) if isinstance(s, dict) and s.get("timestamp")]
        except Exception: continue
        if len(steps) < 3: continue
        n += 1
        if n > max_traj: break
        for a, b in zip(steps, steps[1:]):
            if a.get("source") != "agent": continue
            c = [x for x in cmds(a.get("message", "")) if isinstance(x, dict)]; dur = sum(_f(x.get("duration")) for x in c); blocking = any(x.get("is_blocking") for x in c)
            out = float((b.get("metrics") or {}).get("completion_tokens") or 0) if b.get("source") == "agent" else 0
            rows.append((ts(b["timestamp"]) - ts(a["timestamp"]), dur, blocking, len(c), float((a.get("metrics") or {}).get("completion_tokens") or 0)))
delta = [r[0] for r in rows]; dur = [r[1] for r in rows]; rest = [max(0, r[0] - r[1]) for r in rows]; ctoks = [r[4] for r in rows]
print(f"=== {image.parent.name}: {n} traj, {len(rows)} agent turns ({time.monotonic()-t0:.0f}s)")
print(f"turn wall     p50 {pct(delta,.5):5.1f} p90 {pct(delta,.9):5.1f} mean {statistics.mean(delta):5.1f}")
print(f"declared dur  p50 {pct(dur,.5):5.1f} p90 {pct(dur,.9):5.1f} mean {statistics.mean(dur):5.1f}   (share of wall {sum(dur)/sum(delta):.2f}; blocking turns {sum(r[2] for r in rows)/len(rows):.2f}; cmds/turn {statistics.mean(r[3] for r in rows):.1f})")
print(f"wall - dur    p50 {pct(rest,.5):5.1f} p90 {pct(rest,.9):5.1f} mean {statistics.mean(rest):5.1f}   = LLM decode + harness/bridge")
print(f"compl tokens  p50 {pct(ctoks,.5):5.0f} p90 {pct(ctoks,.9):5.0f} mean {statistics.mean(ctoks):5.0f} per turn")
buckets = [(0, 1), (1, 3), (3, 10), (10, 60), (60, 1e9)]
for lo, hi in buckets:
    sel = [r for r in rows if lo <= r[1] < hi]
    if sel: print(f"  declared dur in [{lo:g},{hi:g}): {len(sel):5d} turns, wall mean {statistics.mean(x[0] for x in sel):5.1f}, wall-dur mean {statistics.mean(max(0,x[0]-x[1]) for x in sel):5.1f}")
