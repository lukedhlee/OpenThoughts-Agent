#!/usr/bin/env python3
"""census_lastturn.py <artifact_store.img> <census.jsonl> — remount an inactive image ro and, for every tmux-lost attempt in the
census, print the agent's LAST LLM response (the turn whose keys were being sent when the tmux server vanished; it is not in the
trajectory because the step never completed) plus what the verifier left behind. Settles agent-caused vs sandbox-caused."""
import json, pathlib, re, sys, time
sys.path.insert(0, "/e/project1/transfernetx/lee27/code/OpenThoughts-Agent")
from hpc.artifact_store import mounted  # noqa: E402
image, census = pathlib.Path(sys.argv[1]), sys.argv[2]
lost = [r for r in (json.loads(l) for l in open(census)) if not r.get("summary")]
t0 = time.monotonic()
with mounted(image, "ro") as root:
    tj = root / "trace_jobs"
    print(f"mounted ({time.monotonic()-t0:.0f}s), {len(lost)} lost attempts", flush=True)
    for i, r in enumerate(lost):
        rel = r["path"].split("/trace_jobs/", 1)[1]
        att = tj / pathlib.Path(rel).parent
        m = r.get("message", ""); kind = "SENDFAIL" if "failed to send batched keys" in m else "NOMARK"
        rc = (re.search(r"return_code=(-?\d+)", m) or [None, "?"])[1]
        if i == 0:
            files = sorted(str(x.relative_to(att)) for x in att.rglob("*") if x.is_file())
            print("LAYOUT", len(files), files[:80], flush=True)
        eps = [p for p in att.rglob("episode-*") if p.is_dir()]
        eps.sort(key=lambda p: int(re.search(r"episode-(\d+)", p.name).group(1)))
        resp = ""
        if eps:
            cands = [p for p in eps[-1].iterdir() if "response" in p.name.lower()] or list(eps[-1].iterdir())
            for c in cands:
                try: resp = c.read_text(errors="replace"); break
                except Exception as e: resp = f"<unreadable {c.name}: {e}>"
        ver = sorted(str(x.name) for x in (att / "verifier").glob("*")) if (att / "verifier").is_dir() else "no-verifier-dir"
        tail = resp[-700:].replace("\n", "⏎")
        print(f"\n#{i} {kind} rc={rc} steps={r.get('agent_steps')} eps={len(eps)} last={eps[-1].name if eps else None} verifier={ver}\n   RESPONSE-TAIL: {tail}", flush=True)
print(f"done ({time.monotonic()-t0:.0f}s)", flush=True)
