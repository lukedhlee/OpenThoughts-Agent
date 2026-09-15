#!/usr/bin/env python3
"""Run the oracle / no-op gate of an artifact-shape tree on Daytona with harbor, in shards at ramped concurrency.

Each shard is a symlink sub-tree (harbor takes `datasets[].path`), run as one `harbor jobs start` with its own
n_concurrent_trials, so the sandbox-start-timeout rate can be read per concurrency level from the trial results.
Meant for the Jupiter login node (direct internet, the snowball-v2 venv with the hook-carrying harbor) or the Mac.

  python3 gate_run.py --tree <tree> --jobs <jobs_dir> --harbor <harbor binary> --key-file <daytona key file> \
      --phase oracle --shards 300:150,600:300,1000:500,rest:750 [--retry-types EnvironmentStartTimeoutError,...]
  python3 gate_run.py ... --phase nop --shards rest:750
  python3 gate_run.py --summarize --jobs <jobs_dir>        # per-shard / per-repo pass, infra rates, timings
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

YAML = """job_name: {name}
jobs_dir: {jobs}
n_attempts: 1
n_concurrent_trials: {conc}
quiet: true
timeout_multiplier: 1.0
retry:
  max_retries: 3
  include_exceptions:
  - DaytonaRateLimitError
  - EnvironmentStartTimeoutError
  - DaytonaSandboxStopError
environment:
  type: daytona
  force_build: false
  delete: true
  override_cpus: 2
  override_memory_mb: 4096
  override_storage_mb: {storage}
  kwargs:
    auto_snapshot: true
verifier:
  override_timeout_sec: 900
agents:
- name: {agent}
  override_timeout_sec: 900
datasets:
- path: {tree}
"""


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def make_shards(tree: Path, out: Path, spec: str, already: set[str], tasks_file: Path | None = None) -> list[tuple[Path, int]]:
    src = tasks_file or (tree / "TASKS.txt")
    tasks = [t for t in (l.strip() for l in open(src)) if t and not t.startswith("#") and t not in already]
    import random
    random.Random(20260915).shuffle(tasks)   # TASKS.txt is id-sorted = repo-clustered; mix repos across the ramp
    shards: list[tuple[Path, int]] = []
    pos = 0
    for i, part in enumerate(spec.split(",")):
        n, conc = part.split(":")
        chunk = tasks[pos:] if n == "rest" else tasks[pos:pos + int(n)]
        pos += len(chunk)
        if not chunk:
            continue
        d = out / f"shard{i}_c{conc}"
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)
        for t in chunk:
            os.symlink(tree / t, d / t)
        shards.append((d, int(conc)))
        log(f"shard {d.name}: {len(chunk)} tasks at conc {conc}")
    return shards


def trial_results(job_dir: Path) -> list[dict]:
    rows = []
    for rj in job_dir.glob("*/result.json"):
        try:
            r = json.load(open(rj))
        except Exception:  # noqa: BLE001
            continue
        task = r.get("task_name") or r.get("trial_name", rj.parent.name).split("__")[0]
        ver = r.get("verifier_result") or {}
        rewards = ver.get("rewards") if isinstance(ver, dict) else None
        reward = None
        if isinstance(rewards, dict) and rewards:
            reward = list(rewards.values())[0]
        exc = r.get("exception_info") or {}
        rows.append({"task": task, "reward": reward, "exc": (exc.get("exception_type") if isinstance(exc, dict) else None),
                     "started": r.get("started_at"), "finished": r.get("finished_at"),
                     "env_setup": ((r.get("environment_setup") or {}).get("started_at"), (r.get("environment_setup") or {}).get("finished_at")),
                     "dir": str(rj.parent)})
    return rows


def run_job(harbor: str, cfg: Path, env: dict) -> int:
    log("harbor jobs start", cfg)
    p = subprocess.run([harbor, "jobs", "start", "--config", str(cfg)], env=env)
    return p.returncode


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree")
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--harbor", default="harbor")
    ap.add_argument("--key-file", default=None, help="file with DAYTONA_API_KEY=... (or rely on the environment)")
    ap.add_argument("--pythonpath", default=None)
    ap.add_argument("--phase", choices=["oracle", "nop"], default="oracle")
    ap.add_argument("--shards", default="300:150,600:300,1000:500,rest:750")
    ap.add_argument("--storage-mb", type=int, default=8192)
    ap.add_argument("--tasks-file", default=None, help="task list to gate (default: <tree>/TASKS.txt)")
    ap.add_argument("--tag", default="", help="suffix for the shard/job names (e.g. a second pass)")
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--rerun", action="store_true", help="gate the listed tasks even if an earlier shard scored them (re-gate after a fix)")
    a = ap.parse_args()
    jobs = Path(a.jobs)
    jobs.mkdir(parents=True, exist_ok=True)
    if a.summarize:
        return summarize(jobs)
    tree = Path(a.tree).resolve()
    env = dict(os.environ)
    if a.key_file:
        for line in open(a.key_file):
            line = line.strip()
            if line.startswith(("DAYTONA_API_KEY=", "export DAYTONA_API_KEY=")):
                env["DAYTONA_API_KEY"] = line.split("=", 1)[1].split("#", 1)[0].strip().strip("\"'")
    if a.pythonpath:
        env["PYTHONPATH"] = a.pythonpath
    done: set[str] = set()
    for prev in ([] if a.rerun else jobs.glob(f"{a.phase}_shard*")):
        for r in trial_results(prev):
            if r["reward"] is not None:
                done.add(r["task"])
    log(f"{len(done)} tasks already scored in earlier {a.phase} shards; skipping them")
    shards = make_shards(tree, jobs / f"shards{a.tag}", a.shards, done, Path(a.tasks_file) if a.tasks_file else None)
    for d, conc in shards:
        name = f"{a.phase}_{d.name}{a.tag}"
        cfg = jobs / f"{name}.yaml"
        cfg.write_text(YAML.format(name=name, jobs=str(jobs), conc=conc, storage=a.storage_mb, agent=a.phase, tree=str(d)))
        t0 = time.time()
        rc = run_job(a.harbor, cfg, env)
        rows = trial_results(jobs / name)
        scored = sum(1 for r in rows if r["reward"] is not None)
        infra = collections.Counter(r["exc"] for r in rows if r["reward"] is None)
        log(f"{name}: rc={rc} trials={len(rows)} scored={scored} infra={dict(infra)} wall={time.time() - t0:.0f}s")
    return summarize(jobs)


def summarize(jobs: Path) -> int:
    repo_of = {}
    m = Path(__file__).resolve().parent.parent / "overlap" / "tasktrove_v3_upstream_map.tsv"
    if m.exists():
        import csv
        repo_of = {r["path"]: r["repo"] for r in csv.DictReader(open(m), delimiter="\t")}
    out = {}
    for phase in ("oracle", "nop"):
        best: dict[str, dict] = {}
        per_shard = {}
        for jd in sorted(jobs.glob(f"{phase}_shard*")):
            rows = trial_results(jd)
            conc = jd.name.split("_c")[-1]
            n = len(rows)
            scored = [r for r in rows if r["reward"] is not None]
            infra = collections.Counter(r["exc"] for r in rows if r["reward"] is None)
            per_shard[jd.name] = {"conc": conc, "trials": n, "scored": len(scored), "infra": dict(infra),
                                  "infra_rate": round(sum(infra.values()) / n, 3) if n else None}
            for r in rows:
                if r["reward"] is not None or r["task"] not in best:
                    best[r["task"]] = r
        out[phase] = {"per_shard": per_shard, "tasks": {t: r["reward"] for t, r in best.items()},
                      "unscored": {t: r["exc"] for t, r in best.items() if r["reward"] is None}}
    ok = [t for t, v in out["oracle"]["tasks"].items() if v == 1.0 and out["nop"]["tasks"].get(t) == 0.0]
    fail = {}
    for t in set(out["oracle"]["tasks"]) | set(out["nop"]["tasks"]):
        o, n = out["oracle"]["tasks"].get(t), out["nop"]["tasks"].get(t)
        if not (o == 1.0 and n == 0.0):
            fail[t] = f"oracle={o} nop={n}"
    by_repo = collections.defaultdict(lambda: [0, 0])
    for t in set(out["oracle"]["tasks"]) | set(out["nop"]["tasks"]):
        by_repo[repo_of.get(t, "?")][1] += 1
        if t in ok:
            by_repo[repo_of.get(t, "?")][0] += 1
    (jobs / "allowlist_r2egym_daytona_v3.txt").write_text("\n".join(sorted(ok)) + "\n")
    (jobs / "gate_fail.tsv").write_text("".join(f"{t}\t{repo_of.get(t, '?')}\t{why}\n" for t, why in sorted(fail.items())))
    summary = {"pass": len(ok), "fail": len(fail), "by_repo": {k: {"pass": v[0], "seen": v[1]} for k, v in sorted(by_repo.items())},
               "oracle_shards": out["oracle"]["per_shard"], "nop_shards": out["nop"]["per_shard"]}
    (jobs / "gate_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
