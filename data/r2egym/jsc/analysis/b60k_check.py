import glob, json, os, re, statistics, collections
E = "/e/fscratch/reformo/lee27/experiments/snowball_probe_val_b60k"
rs = glob.glob(f"{E}/snowball_probe_val_b60k/trace_jobs/eval_sessions/*/*/attempts/*/result.json")
exc = collections.Counter(); turns = []; free = []; used = []; rew = []
for p in rs:
    try: d = json.load(open(p))
    except Exception: continue
    ei = d.get("exception_info") or {}; e = ei.get("exception_type"); exc[e] += 1
    m = re.search(r"(-?\d[\d,]*) free tokens remaining", str(ei.get("exception_message", "")))
    if m: free.append(int(m.group(1).replace(",", "")))
    ar = d.get("agent_result") or {}
    if ar.get("n_input_tokens"): used.append(ar["n_input_tokens"])
    v = d.get("verifier_result"); r = (v.get("rewards") or {}).get("reward") if isinstance(v, dict) else None
    if r is not None: rew.append(r)
    tp = os.path.join(os.path.dirname(p), "agent", "trajectory.json")
    if os.path.exists(tp):
        try: turns.append(len([s for s in json.load(open(tp)).get("steps", []) if s.get("source") == "agent"]))
        except Exception: pass
print("attempts", len(rs), "scored", len(rew), "succ", sum(1 for r in rew if r and r >= 1.0))
print("exceptions", dict(exc))
if turns: print("turns p50/p90/max", statistics.median(turns), sorted(turns)[int(0.9*len(turns))-1] if len(turns)>=10 else None, max(turns))
if free: print("ctx-death 'free tokens' p50", statistics.median(free), "n", len(free))
if used: print("agent n_input_tokens (cumulative) p50/max", statistics.median(used), max(used))
