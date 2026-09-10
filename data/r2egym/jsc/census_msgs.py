#!/usr/bin/env python3
"""census_msgs.py <census.jsonl> — for each TmuxSessionLostError record print the failing turn as recorded in the
exception message (the keys being sent when the server vanished), plus rc, timing, and observation tail."""
import json, re, sys
rows = [json.loads(l) for l in open(sys.argv[1])]
lost = [r for r in rows if not r.get("summary")]
print("record keys:", sorted(lost[0].keys()) if lost else None)
for i, r in enumerate(lost):
    m = r.get("message", "")
    kind = "SENDFAIL" if "failed to send batched keys" in m else ("NOMARK" if "no markers" in m else "OTHER")
    rc = (re.search(r"return_code=(-?\d+)", m) or [None, "?"])[1]
    stderr = (re.search(r"stderr=('.*?'|\".*?\")", m) or [None, ""])[1][:80]
    keys = re.search(r"(\[\{.*?\}\])", m, re.S)
    ks = ""
    if keys:
        try: ks = " | ".join((k.get("keystrokes") or "")[:60].replace("\n", "\\n") for k in json.loads(keys.group(1)))
        except Exception: ks = keys.group(1)[:200]
    extra = {k: r[k] for k in r if k not in ("message", "last_steps", "path") and not isinstance(r[k], (list, dict))}
    print(f"\n#{i} {kind} rc={rc} stderr={stderr} {extra}")
    print("   sending:", ks or m[:200].replace("\n", " "))
    steps = r.get("last_steps") or []
    if steps:
        last = steps[-1]
        obs = (last.get("observation") or last.get("obs_tail") or "")
        if isinstance(obs, list): obs = " ".join(str(o) for o in obs)
        print("   prev-turn keys:", [ (c.get("keystrokes") or "")[:50].replace("\n", "\\n") for c in (last.get("commands") or [])][-3:])
        print("   prev-turn obs tail:", str(obs)[-220:].replace("\n", " ⏎ "))
