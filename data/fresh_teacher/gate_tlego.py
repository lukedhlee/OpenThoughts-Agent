#!/usr/bin/env python3
"""No-op and gold gates for a Terminal-Lego tree on the shared jammy snapshot.

  python gate_tlego.py run --tree <tree> --tasks <list> --jobs <dir> --agent nop|oracle [--conc 16]
  python gate_tlego.py report --jobs <dir>      # needs both runs; writes gate_pass.txt

A task passes only when the untouched sandbox scores 0 and solution/solve.sh scores 1, with the setup replay clean in
both. The config names the snapshot and leaves auto_snapshot off, and the run refuses unless the snapshot is ACTIVE, so
a gate can never build one. Harbor comes from GATE_HARBOR / GATE_HARBOR_SRC; the key file is never printed.
"""
from __future__ import annotations

import argparse
import collections
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "tmax" / "daytona"))
_g = importlib.util.spec_from_file_location("tmax_gate", HERE.parent / "tmax" / "daytona" / "gate.py")
tmax_gate = importlib.util.module_from_spec(_g)
_g.loader.exec_module(tmax_gate)

HARBOR = os.environ.get("GATE_HARBOR", os.path.expanduser("~/snowball/envs/snowball/bin/harbor"))
HARBOR_SRC = os.environ.get("GATE_HARBOR_SRC", os.path.expanduser("~/snowball/harbor-p0924/src"))
RETRY = ["DaytonaRateLimitError", "EnvironmentStartTimeoutError", "DaytonaSandboxStopError", "SetupScriptError"]


def run(a) -> int:
    tree, jobs = Path(a.tree).resolve(), Path(a.jobs).resolve()
    snapshot = json.loads((tree / "pool.json").read_text())["jammy"]["name"]
    key = tmax_gate.load_secret("DAYTONA_API_KEY", a.key_file)
    state = tmax_gate.snapshot_active(snapshot, key)
    if state != "active":
        raise SystemExit(f"{snapshot} is {state}, not active; a gate never builds or activates it")
    sub = jobs / "tree"
    sub.mkdir(parents=True, exist_ok=True)
    for t in Path(a.tasks).read_text().split():
        if not (sub / t).exists():
            (sub / t).symlink_to(tree / t, target_is_directory=True)
    cfg = dict(job_name=a.agent, jobs_dir=str(jobs), n_attempts=1, n_concurrent_trials=a.conc, quiet=True,
               timeout_multiplier=1.0, environment_build_timeout_multiplier=4.0,
               retry=dict(max_retries=2, include_exceptions=RETRY),
               environment=dict(type="daytona", force_build=False, delete=True, override_cpus=2, override_memory_mb=4096,
                                override_storage_mb=10240, kwargs=dict(snapshot_template_name=snapshot)),
               verifier=dict(override_timeout_sec=600),
               agents=[dict(name=a.agent, override_timeout_sec=600 if a.agent == "oracle" else 60)],
               datasets=[dict(path=str(sub))])
    path = jobs / f"{a.agent}.json"
    path.write_text(json.dumps(cfg, indent=1))
    env = dict(os.environ, DAYTONA_API_KEY=key, PYTHONPATH=HARBOR_SRC)
    return subprocess.run([HARBOR, "jobs", "start", "--config", str(path)], env=env).returncode


def rows(job: Path) -> dict:
    out = {}
    for p in sorted(job.glob("*/result.json")):
        r = tmax_gate.trial_row(p.parent)
        out[r["task"]] = r
    return out


def report(a) -> int:
    jobs = Path(a.jobs)
    nop, oracle = rows(jobs / "nop"), rows(jobs / "oracle")
    table, classes = [], collections.Counter()
    for t in sorted(set(nop) | set(oracle)):
        n, o = nop.get(t), oracle.get(t)
        if not n or not o:
            c = "missing_run"
        elif n["exception"] or o["exception"]:
            c = "setup_failed" if "SetupScriptError" in (n["exception"], o["exception"]) else "infrastructure"
        elif n["reward"] != 0.0:
            c = "passes_untouched"
        elif o["reward"] != 1.0:
            c = "gold_fails"
        else:
            c = "pass"
        classes[c] += 1
        table.append(dict(task=t, cls=c, nop=n and n["reward"], gold=o and o["reward"], nop_exc=n and n["exception"],
                          gold_exc=o and o["exception"], setup_s=n and n["setup_sh_s"], killed=(n and n["tmaxsetup"] or {}).get("killed")))
    (jobs / "gate_rows.json").write_text(json.dumps(table, indent=1))
    (jobs / "gate_pass.txt").write_text("".join(r["task"] + "\n" for r in table if r["cls"] == "pass"))
    print(json.dumps(dict(classes), indent=1))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("run")
    p.add_argument("--tree", required=True); p.add_argument("--tasks", required=True); p.add_argument("--jobs", required=True)
    p.add_argument("--agent", choices=["nop", "oracle"], required=True); p.add_argument("--conc", type=int, default=16)
    p.add_argument("--key-file", default=tmax_gate.DEFAULT_KEY_FILE)
    p = sp.add_parser("report")
    p.add_argument("--jobs", required=True)
    a = ap.parse_args()
    return run(a) if a.cmd == "run" else report(a)


if __name__ == "__main__":
    raise SystemExit(main())
