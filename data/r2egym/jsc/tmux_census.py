#!/usr/bin/env python3
"""tmux_census.py <trace_root> <out.jsonl> [time_budget_s]
Walk trace_jobs/*/attempts/*/result.json, count exception types, and for every TmuxSessionLostError attempt dump the
exception message, return code, stderr, timing, and the last 3 agent steps' commands (keystrokes) + the last observation tail,
so the loss can be attributed to the agent (C-d / exit / reset / kill of its own shell) or to the sandbox (timeout, SIGKILL).
Session 5b45b53d, 2026-09-06. Read-only; runs on the batch host where the artifact store is mounted."""
import collections, concurrent.futures, datetime, gzip, json, pathlib, re, sys, time

root = pathlib.Path(sys.argv[1]); out = open(sys.argv[2], "w"); budget = float(sys.argv[3]) if len(sys.argv) > 3 else 1500
t0 = time.monotonic()
paths = list(root.glob("*/attempts/*/result.json"))
print(f"files {len(paths)} enumerated in {time.monotonic()-t0:.0f}s", file=sys.stderr, flush=True)

def scan(p):
    try:
        d = json.loads(p.read_text()); ex = d.get("exception_info") or {}
        return {"path": str(p), "task": d.get("task_name"), "trial": d.get("trial_name"), "started_at": d.get("started_at"),
                "finished_at": d.get("finished_at"), "exception": ex.get("exception_type"), "message": (ex.get("exception_message") or "")[:400],
                "n_episodes": ((d.get("agent_result") or {}).get("metadata") or {}).get("n_episodes"),
                "reward": (d.get("verifier_result") or {}).get("rewards")}
    except Exception as e:
        return {"path": str(p), "error": str(e)}

def commands_of(step):
    msg = step.get("message") or ""; msg = msg if isinstance(msg, str) else str(msg); body = msg.split("<|end_think|>")[-1]
    for q in re.finditer(r"\{", body):
        try:
            z = json.JSONDecoder(strict=False).raw_decode(body[q.start():])[0]
            if isinstance(z, dict) and "commands" in z:
                return [{k: (str(c.get(k))[:160] if c.get(k) is not None else None) for k in ("keystrokes", "is_blocking", "timeout_sec", "duration")}
                        for c in (z.get("commands") or []) if isinstance(c, dict)], str(z.get("analysis") or "")[:300], bool(z.get("task_complete"))
        except Exception:
            pass
    return [], msg[-200:], None

def detail(r):
    p = pathlib.Path(r["path"]); t = p.parent / "agent/trajectory.json"
    try:
        if t.exists(): x = json.loads(t.read_text())
        elif t.with_suffix(".json.gz").exists(): x = json.loads(gzip.open(t.with_suffix(".json.gz"), "rt").read())
        else: r["trajectory"] = "missing"; return r
        steps = [s for s in x.get("steps", []) if s.get("source") == "agent"]
        r["agent_steps"] = len(steps); r["last_steps"] = []
        for s in steps[-3:]:
            cmds, analysis, done = commands_of(s)
            r["last_steps"].append({"ts": s.get("timestamp"), "commands": cmds, "analysis": analysis, "task_complete": done,
                                    "observation_tail": str(s.get("observation") or "")[-300:]})
    except Exception as e:
        r["trajectory_error"] = str(e)
    return r

# Streams every tmux-lost record (with its trajectory detail) as soon as it is found, so a FUSE stall late in the
# scan costs nothing already written; the summary line is written last. Stuck reads are abandoned via as_completed timeouts.
counts = collections.Counter(); lost = 0; n = 0
ex = concurrent.futures.ThreadPoolExecutor(max_workers=4)
try:
    for chunk_start in range(0, len(paths), 400):
        if time.monotonic() - t0 > budget: print(f"budget hit at {n}", file=sys.stderr, flush=True); break
        futs = [ex.submit(scan, p) for p in paths[chunk_start:chunk_start + 400]]
        try:
            for f in concurrent.futures.as_completed(futs, timeout=180):
                r = f.result(); n += 1; counts[r.get("exception") or ("error" if "error" in r else "none")] += 1
                if r.get("exception") == "TmuxSessionLostError":
                    lost += 1; print(json.dumps(detail(r)), file=out, flush=True)
        except concurrent.futures.TimeoutError:
            print(f"chunk at {chunk_start} timed out; abandoning stuck reads", file=sys.stderr, flush=True)
        print(f"scanned {n}/{len(paths)} lost={lost} {time.monotonic()-t0:.0f}s", file=sys.stderr, flush=True)
finally:
    ex.shutdown(wait=False, cancel_futures=True)
print(json.dumps({"summary": True, "root": str(root), "attempts_scanned": n, "attempts_total": len(paths), "exceptions": dict(counts.most_common())}), file=out, flush=True)
print(f"done: {n} scanned, {lost} tmux-lost, {time.monotonic()-t0:.0f}s", file=sys.stderr, flush=True)
import os; os._exit(0)
