#!/usr/bin/env python3
"""No-op gate for the CalibForge Daytona tree: sample, run harbor's nop agent, and report.

A task passes when its trial has no exception (a failed setup.sh raises SetupScriptError before the agent), the
reward is 0, and the verifier ran to completion (a ctrf report with tests collected; verifiers without ctrf count on
reward.txt alone and are reported separately). Setup time per sandbox = harbor's environment_setup (sandbox create
and start) plus setup.sh's own wall time (from trial.log).

  python gate.py sample --tree <tree> --n 50 --out sample.txt          # stratified by base, then category
  python gate.py run --tree <tree> --tasks sample.txt --jobs <dir> [--conc 25]
  python gate.py report --jobs <dir>/calibforge_nop
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import os
import random
import re
import statistics
import subprocess
import sys
import tomllib
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from build_snapshots import DEFAULT_KEY_FILE, load_secret  # noqa: E402

HARBOR = "/Users/lukedhlee/harbor/.venv/bin/harbor"
HARBOR_SRC = "/Users/lukedhlee/harbor-wt/snowball-r2egym/src"


def sample(a) -> int:
    tree = Path(a.tree)
    tasks = (tree / "TASKS.txt").read_text().split()
    meta = {}
    for t in tasks:
        m = tomllib.loads((tree / t / "task.toml").read_text())["metadata"]
        meta[t] = (m["calibforge_daytona_base"], m.get("category", "?"))
    by_base = collections.defaultdict(list)
    for t, (b, _) in meta.items():
        by_base[b].append(t)
    # proportional to base size, at least 6 per base
    quota = {b: max(6, round(a.n * len(v) / len(tasks))) for b, v in by_base.items()}
    while sum(quota.values()) > a.n:
        quota[max(quota, key=quota.get)] -= 1
    rng = random.Random(a.seed)
    picked = []
    for b, want in sorted(quota.items()):
        by_cat = collections.defaultdict(list)
        for t in sorted(by_base[b]):
            by_cat[meta[t][1]].append(t)
        for v in by_cat.values():
            rng.shuffle(v)
        cats = sorted(by_cat, key=lambda c: -len(by_cat[c]))
        i = 0
        while want and any(by_cat.values()):
            c = cats[i % len(cats)]
            if by_cat[c]:
                picked.append(by_cat[c].pop())
                want -= 1
            i += 1
    Path(a.out).write_text("\n".join(sorted(picked)) + "\n")
    print(json.dumps({"n": len(picked), "by_base": collections.Counter(meta[t][0] for t in picked),
                      "by_category": collections.Counter(meta[t][1] for t in picked)}, indent=1))
    return 0


def run(a) -> int:
    import yaml
    tree, jobs = Path(a.tree).resolve(), Path(a.jobs).resolve()
    sub = jobs / "tree"
    sub.mkdir(parents=True, exist_ok=True)
    for t in Path(a.tasks).read_text().split():
        if not (sub / t).exists():
            (sub / t).symlink_to(tree / t, target_is_directory=True)
    cfg = yaml.safe_load((HERE / "gate_nop.yaml").read_text())
    cfg.update(job_name=a.name, jobs_dir=str(jobs), n_concurrent_trials=a.conc, quiet=True)
    cfg["datasets"] = [dict(path=str(sub))]
    path = jobs / f"{a.name}.yaml"
    path.write_text(json.dumps(cfg, indent=1))
    env = dict(os.environ, DAYTONA_API_KEY=load_secret("DAYTONA_API_KEY", a.key_file), PYTHONPATH=HARBOR_SRC)
    rc = subprocess.run([a.harbor, "jobs", "start", "--config", str(path)], env=env).returncode
    print("harbor rc", rc, flush=True)
    return report_dir(jobs / a.name)


def ts(s):
    return dt.datetime.fromisoformat(s.replace("Z", "+00:00")) if s else None


def trial_row(tdir: Path) -> dict:
    r = json.loads((tdir / "result.json").read_text())
    attempts = sorted(tdir.glob("attempts/*"))
    last = attempts[-1] if attempts else tdir
    exc = r.get("exception_info") or {}
    es = r.get("environment_setup") or {}
    env_s = (ts(es.get("finished_at")) - ts(es.get("started_at"))).total_seconds() if es.get("finished_at") else None
    setup_s = cf = None
    log = last / "trial.log"
    if log.exists():
        text = log.read_text(errors="replace")
        m = re.search(r"setup_files/setup\.sh exited (\d+) after ([\d.]+)s", text)
        if m:
            setup_s = float(m.group(2))
        m = re.search(r"CFDELTA (\{[^}]*\})", text)
        if m:
            cf = json.loads(m.group(1))
    ctrf = None
    for p in sorted(last.glob("verifier/ctrf.json")):
        try:
            ctrf = json.loads(p.read_text())["results"]["summary"]
        except Exception:  # noqa: BLE001
            ctrf = {"unreadable": True}
    rewards = (r.get("verifier_result") or {}).get("rewards") or {}
    return {"task": r.get("task_name"), "trial": tdir.name, "reward": rewards.get("reward"),
            "exception": exc.get("exception_type"), "exception_msg": (exc.get("exception_message") or "")[:300],
            "env_setup_s": env_s, "setup_sh_s": setup_s, "cfdelta": cf,
            "ctrf_tests": (ctrf or {}).get("tests"), "ctrf_failed": (ctrf or {}).get("failed"),
            "ctrf_passed": (ctrf or {}).get("passed"), "attempts": len(attempts)}


def pct(v, q):
    v = sorted(x for x in v if x is not None)
    return round(v[min(len(v) - 1, int(q * len(v)))], 1) if v else None


def report_dir(job: Path, tree: Path | None = None) -> int:
    rows = [trial_row(p.parent) for p in sorted(job.glob("*/result.json"))]
    base_of = {}
    for r in rows:
        link = job.parent / "tree" / (r["task"] or "")
        try:
            base_of[r["task"]] = tomllib.loads((link / "task.toml").read_text())["metadata"]["calibforge_daytona_base"]
        except OSError:
            base_of[r["task"]] = "?"
    for r in rows:
        r["base"] = base_of.get(r["task"])
        r["verifier_completed"] = (r["ctrf_tests"] or 0) > 0 if r["ctrf_tests"] is not None else r["reward"] is not None
        r["pass"] = r["exception"] is None and r["reward"] == 0.0 and r["verifier_completed"]
    total = [(r["env_setup_s"] or 0) + (r["setup_sh_s"] or 0) for r in rows if r["setup_sh_s"] is not None]
    summary = {
        "trials": len(rows), "pass": sum(r["pass"] for r in rows),
        "by_base": {b: f"{sum(r['pass'] for r in rows if r['base'] == b)}/{sum(r['base'] == b for r in rows)}"
                    for b in sorted({r["base"] for r in rows})},
        "exceptions": collections.Counter(r["exception"] for r in rows if r["exception"]),
        "reward_nonzero": [r["task"] for r in rows if r["reward"] not in (0.0, None)],
        "no_ctrf": [r["task"] for r in rows if r["ctrf_tests"] is None],
        "ctrf_zero_tests": [r["task"] for r in rows if r["ctrf_tests"] == 0],
        "env_setup_s": {"median": pct([r["env_setup_s"] for r in rows], .5), "p90": pct([r["env_setup_s"] for r in rows], .9)},
        "setup_sh_s": {"median": pct([r["setup_sh_s"] for r in rows], .5), "p90": pct([r["setup_sh_s"] for r in rows], .9),
                       "max": pct([r["setup_sh_s"] for r in rows], 1.0)},
        "sandbox_ready_s": {"median": pct(total, .5), "p90": pct(total, .9)},
        "delta_mb": {"median": pct([(r["cfdelta"] or {}).get("bytes", 0) / 1e6 for r in rows if r["cfdelta"]], .5),
                     "p90": pct([(r["cfdelta"] or {}).get("bytes", 0) / 1e6 for r in rows if r["cfdelta"]], .9)},
        "failed": [{k: r[k] for k in ("task", "exception", "exception_msg", "reward", "ctrf_tests")}
                   for r in rows if not r["pass"]],
    }
    (job / "gate_rows.json").write_text(json.dumps(rows, indent=1))
    (job / "gate_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    return 0 if summary["pass"] == len(rows) and rows else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("sample"); p.add_argument("--tree", required=True); p.add_argument("--n", type=int, default=50)
    p.add_argument("--seed", type=int, default=20260924); p.add_argument("--out", required=True)
    p = sub.add_parser("run"); p.add_argument("--tree", required=True); p.add_argument("--tasks", required=True)
    p.add_argument("--jobs", required=True); p.add_argument("--conc", type=int, default=25)
    p.add_argument("--name", default="calibforge_nop"); p.add_argument("--harbor", default=HARBOR)
    p.add_argument("--key-file", default=DEFAULT_KEY_FILE)
    p = sub.add_parser("report"); p.add_argument("--jobs", required=True)
    a = ap.parse_args()
    if a.cmd == "sample":
        return sample(a)
    if a.cmd == "run":
        return run(a)
    return report_dir(Path(a.jobs))


if __name__ == "__main__":
    raise SystemExit(main())
