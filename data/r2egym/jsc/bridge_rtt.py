#!/usr/bin/env python3
"""bridge_rtt.py <bridge_url> <task_dir> [n_exec] — create ONE sandbox through the apptainer bridge the same way the harbor client
does (same task_name/dockerfile hash so the worker reuses its cached sif), then time n plain execs (`true`) and n batch-shaped execs
(has-session guard + capture, ~the per-turn script size) end to end: client POST → worker poll → apptainer exec → result poll.
Isolates the harness/bridge share of a turn from LLM decode. Stops the env at the end."""
import asyncio, json, pathlib, statistics, sys, time, urllib.request
from harbor.utils.container_cache import dockerfile_hash_truncated
bridge, task_dir = sys.argv[1].rstrip("/"), pathlib.Path(sys.argv[2]); n = int(sys.argv[3]) if len(sys.argv) > 3 else 30
env_dir = task_dir / "environment"; df = env_dir / "Dockerfile"
def post(path, payload, timeout=60):
    req = urllib.request.Request(bridge + path, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r: return json.loads(r.read())
def get(path, timeout=10):
    with urllib.request.urlopen(bridge + path, timeout=timeout) as r: return json.loads(r.read())
def wait(job_id, timeout=600, poll=0.05):
    t0 = time.monotonic(); iv = poll; polls = 0
    while time.monotonic() - t0 < timeout:
        r = get(f"/job/result/{job_id}"); polls += 1
        if r.get("state") == "done": return r, time.monotonic() - t0, polls
        if r.get("state") == "error": raise RuntimeError(r.get("error"))
        time.sleep(iv); iv = min(iv * 1.3, 0.5)
    raise TimeoutError(job_id)
base_image = next((l.strip()[5:].split(" AS ")[0].strip() for l in df.read_text().splitlines() if l.strip().upper().startswith("FROM ")), "")
payload = {"task_name": task_dir.name, "dockerfile_hash": dockerfile_hash_truncated(df), "sif_path": "", "base_image": base_image,
           "environment_dir": str(env_dir), "force_build": False, "task_env_config": {"memory_mb": 4096, "cpus": 2, "allow_internet": False},
           "files_b64": {}, "setup_files_b64": {}}
t0 = time.monotonic(); resp = post("/env/create", payload); env_id = resp["env_id"]
r, t_start, _ = wait(resp["job_id"], timeout=900)
print(f"env {env_id} started in {time.monotonic()-t0:.1f}s (job wait {t_start:.1f}s) result={str(r.get('result'))[:80]}", flush=True)
def timed_exec(cmd, label):
    ts = []; polls = []; ttfr = []
    for i in range(n):
        t = time.monotonic(); j = post("/env/exec", {"env_id": env_id, "command": cmd, "cwd": "/", "user": "root", "timeout_sec": 60})
        ttfr.append(time.monotonic() - t)
        r, tw, p = wait(j["job_id"]); ts.append(time.monotonic() - t); polls.append(p)
    ts.sort(); print(f"{label:12s} n={n} rtt p50 {ts[len(ts)//2]:.2f}s p90 {ts[int(.9*len(ts))]:.2f}s min {ts[0]:.2f}s max {ts[-1]:.2f}s | submit-post p50 {statistics.median(ttfr):.3f}s | polls/job {statistics.mean(polls):.1f} | rc={r.get('return_code')}", flush=True)
try:
    timed_exec("true", "true")
    timed_exec("tmux -V >/dev/null 2>&1; printf '%s\\n' __M__; for i in $(seq 1 3); do printf 'line %s\\n' $i; done; sleep 0.5; printf '%s\\n' __E__", "batch-shape")
    timed_exec("sleep 2", "sleep2")
finally:
    j = post("/env/stop", {"env_id": env_id, "delete": True}); print("stop job", j.get("job_id"))
