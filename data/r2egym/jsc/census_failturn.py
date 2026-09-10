#!/usr/bin/env python3
"""census_failturn.py <artifact_store.img> <census.jsonl> — remount ro; for every tmux-lost attempt read the LAST step of
agent/trajectory.json (the failing turn: its LLM message carries the commands whose send-keys hit a dead server) and print the
keystrokes, plus a kill/exit classification and the final pane tail. Settles agent-caused vs sandbox-caused per case."""
import collections, json, pathlib, re, sys, time
sys.path.insert(0, "/e/project1/transfernetx/lee27/code/OpenThoughts-Agent")
from hpc.artifact_store import mounted  # noqa: E402
image, census = pathlib.Path(sys.argv[1]), sys.argv[2]
lost = [r for r in (json.loads(l) for l in open(census)) if not r.get("summary")]
PAT = [("pkill/killall -f", r"\b(pkill|killall)\b.*-f|\bpkill\s+-9?\s*-f"), ("kill", r"\bkill\b"), ("exit", r"(^|\n|;|&&)\s*exit\b"), ("C-d", r"\"C-d\"|'C-d'"),
       ("logout", r"\blogout\b"), ("tmux", r"\btmux\b"), ("exec", r"(^|\n|;|&&)\s*exec\b"), ("reset/stty", r"\b(reset|stty)\b"), ("C-c", r"C-c")]
def keys_of(msg):
    m = re.search(r"\{.*\}", msg, re.S)
    try:
        d = json.loads(m.group(0)) if m else {}
        return [c.get("keystrokes", "") for c in d.get("commands", [])]
    except Exception:
        return re.findall(r'"keystrokes"\s*:\s*"((?:[^"\\]|\\.)*)"', msg)
def classify(keys):
    txt = "\n".join(keys)
    return "+".join(n for n, p in PAT if re.search(p, txt)) or "other"
t0 = time.monotonic(); tally = collections.Counter()
with mounted(image, "ro") as root:
    tj = root / "trace_jobs"
    for r in lost:
        att = tj / pathlib.Path(r["path"].split("/trace_jobs/", 1)[1]).parent
        kind = "SENDFAIL" if "failed to send batched keys" in r["message"] else "NOMARK"
        rc = (re.search(r"return_code=(-?\d+)", r["message"]) or [None, "?"])[1]
        try:
            steps = json.loads((att / "agent" / "trajectory.json").read_text()).get("steps", [])
            last = steps[-1] if steps else {}
            msg = last.get("message", "") if isinstance(last, dict) else ""
            if isinstance(msg, dict): msg = json.dumps(msg)
        except Exception as e:
            msg = f"<traj error {e}>"
        keys = keys_of(str(msg)); cls = classify(keys); tally[(kind, rc, cls)] += 1
        try: pane = (att / "agent" / "terminus_2.pane").read_text(errors="replace")[-260:].replace("\n", "⏎")
        except Exception: pane = "<no pane>"
        print(f"\n## {kind} rc={rc} [{cls}] {r['trial']} steps={r.get('agent_steps')} traj_steps={len(steps) if isinstance(steps, list) else '?'}")
        print("   FAILING-TURN KEYS:", [k[:90].replace("\n", "\\n") for k in keys][:5] if keys else str(msg)[-300:].replace("\n", "\\n"))
        print("   PANE:", pane)
print("\n=== TALLY (kind, rc, failing-turn class)")
for k, v in sorted(tally.items(), key=lambda x: -x[1]): print(f"{v:3d}  {k}")
print(f"done ({time.monotonic()-t0:.0f}s)", flush=True)
