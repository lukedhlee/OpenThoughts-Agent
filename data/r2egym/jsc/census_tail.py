#!/usr/bin/env python3
"""census_tail.py <artifact_store.img> <census.jsonl> [n] — remount an inactive image ro and, for the first n tmux-lost attempts,
print exception.txt, the tail of trial.log and the final tmux pane capture (agent/terminus_2.pane), so the moment the tmux
server vanished can be read: what the agent typed, what the shell showed, whether the sandbox itself was gone."""
import json, pathlib, re, sys, time
sys.path.insert(0, "/e/project1/transfernetx/lee27/code/OpenThoughts-Agent")
from hpc.artifact_store import mounted  # noqa: E402
image, census = pathlib.Path(sys.argv[1]), sys.argv[2]; n = int(sys.argv[3]) if len(sys.argv) > 3 else 12
lost = [r for r in (json.loads(l) for l in open(census)) if not r.get("summary")]
sf = [r for r in lost if "failed to send batched keys" in r.get("message", "")][: n * 2 // 3]
nm = [r for r in lost if "no markers" in r.get("message", "")][: n - len(sf)]
t0 = time.monotonic()
def rd(p, k):
    try: return p.read_text(errors="replace")[-k:].replace("\n", "⏎")
    except Exception as e: return f"<{e.__class__.__name__}>"
with mounted(image, "ro") as root:
    tj = root / "trace_jobs"
    for r in sf + nm:
        att = tj / pathlib.Path(r["path"].split("/trace_jobs/", 1)[1]).parent
        kind = "SENDFAIL" if "failed to send batched keys" in r["message"] else "NOMARK"
        print(f"\n##### {kind} {r['trial']} steps={r.get('agent_steps')} {r['started_at'][11:19]}->{r['finished_at'][11:19]}")
        print("EXC:", rd(att / "exception.txt", 400))
        print("TRIAL.LOG tail:", rd(att / "trial.log", 1200))
        print("PANE tail:", rd(att / "agent" / "terminus_2.pane", 700))
        tr = att / "agent" / "trajectory.json"
        try:
            t = json.loads(tr.read_text()); steps = t.get("steps") or t.get("trajectory") or t
            if isinstance(t, dict): print("TRAJ keys:", sorted(t.keys())[:12], "n_steps:", len(steps) if hasattr(steps, "__len__") else "?")
            if isinstance(steps, list) and steps:
                last = steps[-1]; print("TRAJ last step keys:", sorted(last.keys()) if isinstance(last, dict) else type(last))
                print("TRAJ last step:", json.dumps(last)[-900:])
        except Exception as e: print("TRAJ:", e)
print(f"done ({time.monotonic()-t0:.0f}s)", flush=True)
