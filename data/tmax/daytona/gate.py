#!/usr/bin/env python3
"""No-op gate for the TMax Daytona tree: one nop sandbox per task, from a tree built with build_tree.py --gate.

A task passes when
  - setup ran: no trial exception (a failed setup.sh raises SetupScriptError before the agent);
  - the environment is the one the task describes: test_initial_state.py collects tests and all of them pass;
  - the untouched sandbox fails the verifier for a real reason: reward 0, and test_final_state.py collected tests of
    which at least one failed (not only collection errors: that is a broken verifier, not a failing one).
Anything else is dropped from the relay list, with its class: setup_failed, initial_failed, final_passes_untouched,
final_broken, no_verifier_output, or the exception type. Setup time per sandbox = harbor's environment_setup (sandbox
create and start) plus setup.sh's own wall time (trial.log).

  python gate.py sample --tree <gate tree> --n 20 --out sample.txt        # stratified by Dockerfile shape, then domain
  python gate.py run --tree <gate tree> --tasks sample.txt --jobs <dir> [--conc 20] [--name tmax_nop]
  python gate.py report --jobs <dir>/tmax_nop                             # writes gate_pass.txt for the relay
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import os
import random
import re
import subprocess
import sys
import tomllib
import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "calibforge" / "daytona"))
from build_snapshots import DEFAULT_KEY_FILE, load_secret  # noqa: E402

HARBOR = os.environ.get("TMAX_HARBOR", "/Users/lukedhlee/harbor/.venv/bin/harbor")   # Horizon: ~/snowball/envs/snowball/bin/harbor
HARBOR_SRC = os.environ.get("TMAX_HARBOR_SRC", "/Users/lukedhlee/harbor-wt/snowball-r2egym/src")   # Horizon: ~/snowball/harbor-relay/src


def ts(s):  # as in data/calibforge/daytona/gate.py
    return dt.datetime.fromisoformat(s.replace("Z", "+00:00")) if s else None


def pct(v, q):
    v = sorted(x for x in v if x is not None)
    return round(v[min(len(v) - 1, int(q * len(v)))], 1) if v else None


def sample(a) -> int:
    tree = Path(a.tree)
    tasks = (tree / "TASKS.txt").read_text().split()
    meta = {}
    for t in tasks:
        m = tomllib.loads((tree / t / "task.toml").read_text())["metadata"]
        meta[t] = (m["tmax_dockerfile_shape"], m.get("domain", "?"))
    by_shape = collections.defaultdict(list)
    for t, (s, _) in meta.items():
        by_shape[s].append(t)
    quota = {s: max(2, round(a.n * len(v) / len(tasks))) for s, v in by_shape.items()}  # every shape at least 2
    while sum(quota.values()) > a.n:
        quota[max(quota, key=quota.get)] -= 1
    rng = random.Random(a.seed)
    picked = []
    for s, want in sorted(quota.items()):
        by_dom = collections.defaultdict(list)
        for t in sorted(by_shape[s]):
            by_dom[meta[t][1]].append(t)
        for v in by_dom.values():
            rng.shuffle(v)
        doms = sorted(by_dom, key=lambda d: -len(by_dom[d]))
        i = 0
        while want and any(by_dom.values()):
            d = doms[i % len(doms)]
            if by_dom[d]:
                picked.append(by_dom[d].pop())
                want -= 1
            i += 1
    Path(a.out).write_text("\n".join(sorted(picked)) + "\n")
    print(json.dumps({"n": len(picked), "by_shape": collections.Counter(meta[t][0] for t in picked),
                      "by_domain": collections.Counter(meta[t][1] for t in picked)}, indent=1))
    return 0


def snapshot_active(name: str, key: str) -> str:
    """Read-only: the snapshot's state, or 'missing'. With auto_snapshot, harbor BUILDS a missing snapshot on the first
    trial, and the eval org is at its 40-snapshot cap, so the gate refuses to start unless the snapshot is ACTIVE."""
    import asyncio
    from daytona import AsyncDaytona, DaytonaConfig

    async def get():
        async with AsyncDaytona(DaytonaConfig(api_key=key)) as d:
            try:
                s = await d.snapshot.get(name)
            except Exception as exc:  # noqa: BLE001
                if "404" in str(exc) or "not found" in str(exc).lower():
                    return "missing"
                raise
            return str(getattr(s.state, "value", s.state)).lower()
    return asyncio.run(get())


def run(a) -> int:
    import yaml
    tree, jobs = Path(a.tree).resolve(), Path(a.jobs).resolve()
    pool = json.loads((tree / "pool.json").read_text())["jammy"]
    if not pool.get("gate"):
        raise SystemExit(f"{tree} was not built with --gate (no test_initial_state.py)")
    key = load_secret("DAYTONA_API_KEY", a.key_file)
    state = snapshot_active(pool["name"], key)
    if state != "active":
        raise SystemExit(f"{pool['name']} is {state}, not active: build it first (build_snapshots.py --tree {tree}, "
                         "after a slot is free); the gate never lets harbor build it")
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
    env = dict(os.environ, DAYTONA_API_KEY=key, PYTHONPATH=HARBOR_SRC)
    rc = subprocess.run([a.harbor, "jobs", "start", "--config", str(path)], env=env).returncode
    print("harbor rc", rc, flush=True)
    return report_dir(jobs / a.name)


def junit(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError:
        return {"unreadable": True}
    suites = [root] if root.tag == "testsuite" else root.findall("testsuite")
    tot = collections.Counter()
    for s in suites:
        for k in ("tests", "failures", "errors", "skipped"):
            tot[k] += int(s.get(k, 0))
    return dict(tot)


def trial_row(tdir: Path) -> dict:
    r = json.loads((tdir / "result.json").read_text())
    attempts = sorted(tdir.glob("attempts/*"))
    last = attempts[-1] if attempts else tdir
    exc = r.get("exception_info") or {}
    es = r.get("environment_setup") or {}
    env_s = (ts(es.get("finished_at")) - ts(es.get("started_at"))).total_seconds() if es.get("finished_at") else None
    setup_s = tm = None
    log = last / "trial.log"
    if log.exists():
        text = log.read_text(errors="replace")
        m = re.search(r"setup_files/setup\.sh exited (\d+) after ([\d.]+)s", text)
        if m:
            setup_s = float(m.group(2))
        m = re.search(r"TMAXSETUP (\{[^}]*\})", text)
        if m:
            tm = json.loads(m.group(1))
    vdir = last / "verifier"
    rc = (vdir / "initial_rc.txt").read_text().strip() if (vdir / "initial_rc.txt").exists() else None
    rewards = (r.get("verifier_result") or {}).get("rewards") or {}
    return {"task": r.get("task_name"), "trial": tdir.name, "reward": rewards.get("reward"),
            "exception": exc.get("exception_type"), "exception_msg": (exc.get("exception_message") or "")[:300],
            "env_setup_s": env_s, "setup_sh_s": setup_s, "tmaxsetup": tm, "initial_rc": rc,
            "initial": junit(vdir / "initial.xml"), "final": junit(vdir / "final.xml"), "attempts": len(attempts)}


def classify(r: dict) -> str:
    if r["exception"]:
        return "setup_failed" if r["exception"] == "SetupScriptError" else r["exception"]
    ini, fin = r["initial"] or {}, r["final"] or {}
    if r["reward"] is None or not fin:
        return "no_verifier_output"
    if r["initial_rc"] != "0" or not ini.get("tests") or ini.get("failures") or ini.get("errors"):
        return "initial_failed"
    if r["reward"] != 0.0:
        return "final_passes_untouched"
    if not fin.get("tests") or not fin.get("failures"):
        return "final_broken"
    return "pass"


def report_dir(job: Path) -> int:
    rows = [trial_row(p.parent) for p in sorted(job.glob("*/result.json"))]
    shape_of = {}
    for r in rows:
        try:
            md = tomllib.loads((job.parent / "tree" / (r["task"] or "") / "task.toml").read_text())["metadata"]
            shape_of[r["task"]] = "v2" if md.get("source") == "v2" else "legacy"
        except OSError:
            shape_of[r["task"]] = "?"
    for r in rows:
        r["source"] = shape_of.get(r["task"])
        r["class"] = classify(r)
    total = [(r["env_setup_s"] or 0) + (r["setup_sh_s"] or 0) for r in rows if r["setup_sh_s"] is not None]
    by = collections.defaultdict(list)
    for r in rows:
        by[r["source"]].append(r)
    summary = {
        "trials": len(rows), "pass": sum(r["class"] == "pass" for r in rows),
        "classes": collections.Counter(r["class"] for r in rows),
        "by_source": {s: f"{sum(r['class'] == 'pass' for r in v)}/{len(v)}" for s, v in sorted(by.items())},
        "env_setup_s": {"median": pct([r["env_setup_s"] for r in rows], .5), "p90": pct([r["env_setup_s"] for r in rows], .9)},
        "setup_sh_s": {s: {"median": pct([r["setup_sh_s"] for r in v], .5), "p90": pct([r["setup_sh_s"] for r in v], .9),
                           "max": pct([r["setup_sh_s"] for r in v], 1.0)} for s, v in sorted(by.items())},
        "sandbox_ready_s": {"median": pct(total, .5), "p90": pct(total, .9)},
        "setup_killed_processes": sum(bool((r["tmaxsetup"] or {}).get("killed")) for r in rows),
        "retried": sum(r["attempts"] > 1 for r in rows),
        "failed": [{k: r[k] for k in ("task", "class", "exception_msg", "reward", "initial", "final")}
                   for r in rows if r["class"] != "pass"][:200],
    }
    (job / "gate_rows.json").write_text(json.dumps(rows, indent=1))
    (job / "gate_summary.json").write_text(json.dumps(summary, indent=1))
    (job / "gate_pass.txt").write_text("".join(f"{r['task']}\n" for r in sorted(rows, key=lambda r: r["task"] or "")
                                               if r["class"] == "pass"))
    print(json.dumps({k: v for k, v in summary.items() if k != "failed"}, indent=1))
    return 0 if rows and summary["pass"] == len(rows) else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("sample"); p.add_argument("--tree", required=True); p.add_argument("--n", type=int, default=20)
    p.add_argument("--seed", type=int, default=20260929); p.add_argument("--out", required=True)
    p = sub.add_parser("run"); p.add_argument("--tree", required=True); p.add_argument("--tasks", required=True)
    p.add_argument("--jobs", required=True); p.add_argument("--conc", type=int, default=20)
    p.add_argument("--name", default="tmax_nop"); p.add_argument("--harbor", default=HARBOR)
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
