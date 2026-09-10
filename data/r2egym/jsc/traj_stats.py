import json, statistics, glob, sys
tj = sys.argv[1]
paths = glob.glob(f"{tj}/eval_sessions/*/*/attempts/000/agent/trajectory.json")
ns, ctoks, olens, first = [], [], [], None
for p in paths:
    try: t = json.load(open(p))
    except Exception: continue
    ag = [s for s in t.get("steps", []) if s.get("source") == "agent"]
    ns.append(len(ag))
    for s in ag:
        m = s.get("metrics") or {}
        v = m.get("completion_tokens") or m.get("output_tokens")
        if v: ctoks.append(v)
    ob = [s for s in t.get("steps", []) if s.get("source") != "agent"]
    olens += [len(s.get("message", "")) for s in ob]
    if first is None and len(ag) >= 3: first = ag
print(f"trials={len(ns)} agent_steps med/min/max={statistics.median(ns)}/{min(ns)}/{max(ns)}")
print(f"completion_tokens per step: med={statistics.median(ctoks)} p90={sorted(ctoks)[int(0.9*len(ctoks))]} max={max(ctoks)}")
print(f"observation chars per step: med={statistics.median(olens)} p90={sorted(olens)[int(0.9*len(olens))]} max={max(olens)}")
for i, s in enumerate(first[:2]):
    msg = s.get("message", "")
    print(f"--- sample agent step {i}: {len(msg)} chars, completion_tokens={ (s.get('metrics') or {}).get('completion_tokens') }")
    print(msg[:900])
    print("...")
