#!/usr/bin/env python3
"""Extract replay trajectories from a probe trace tree (result.json per attempt).

Per trial: the real chat history (all_messages) plus per-turn production completion lengths,
prompt lengths and API latencies, so a benchmark can replay production-shaped multi-turn load.
Usage: prep_replay.py <trace_jobs dir> <out.jsonl> [n=256] [min_turns=6]
"""
import glob, json, os, random, sys

tj, out = sys.argv[1], sys.argv[2]
n = int(sys.argv[3]) if len(sys.argv) > 3 else 256
min_turns = int(sys.argv[4]) if len(sys.argv) > 4 else 6
paths = sorted(glob.glob(f"{tj}/eval_sessions/*/*/attempts/*/result.json"))
random.seed(0); random.shuffle(paths)
rows, skipped = [], 0
for p in paths:
    try:
        d = json.load(open(p))
    except Exception:
        skipped += 1; continue
    ar = d.get("agent_result") or {}; md = ar.get("metadata") or {}
    msgs = md.get("all_messages") or []
    rd = (ar.get("rollout_details") or [None])[0] or {}
    comp = rd.get("completion_token_ids") or []
    prm = rd.get("prompt_token_ids") or []
    api = md.get("api_request_times_msec") or []
    n_turns = min(len(comp), len(api), len(msgs) // 2)
    if n_turns < min_turns:
        skipped += 1; continue
    if any(msgs[i]["role"] != ("user" if i % 2 == 0 else "assistant") for i in range(2 * n_turns)):
        skipped += 1; continue
    rows.append({
        "task": d.get("task_name"), "trial": d.get("trial_name"), "n_turns": n_turns,
        "completion_len": [len(c) for c in comp[:n_turns]],
        "prompt_len": [len(x) for x in prm[:n_turns]],
        "api_ms": api[:n_turns],
        "messages": [{"role": m["role"], "content": m["content"]} for m in msgs[:2 * n_turns]],
    })
    if len(rows) >= n:
        break
with open(out, "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
tot = sum(r["n_turns"] for r in rows)
print(f"wrote {len(rows)} trajectories ({tot} turns, skipped {skipped}) to {out}; {os.path.getsize(out) / 1e6:.1f} MB")
