#!/usr/bin/env python3
"""Build the TMax task tree for Daytona: one shared ubuntu 22.04 snapshot, each task's own Dockerfile replayed at start.

The CalibForge pattern (data/calibforge/daytona), with a replay instead of layer stacking: TMax ships a recipe per task
(laion/TMax-15K-Harbor: Dockerfile + post_install.sh [+ base_install.sh] + fixtures), all `FROM ubuntu:22.04`, so
each task becomes:
  environment/Dockerfile   the shared base recipe (tmax_base.dockerfile), byte-identical for every task, so harbor's
                           environment-dir hash gives one `harbor__<hash>__snapshot` for the whole pool.
  setup_files/setup.sh     setup_template.sh with the task's Dockerfile after FROM translated step by step
                           (ENV -> setenv, COPY -> copy with the file's mode, RUN -> step); see the template's header.
  setup_files/context/     the task's original environment/ dir (install scripts, fixtures, Dockerfile.orig).
  task.toml                TMax's, plus [environment] workdir "/" (the image has no WORKDIR) and env (the image's
                           ENV, which the agent's shell must see), and the relay budgets (--agent-timeout /
                           --verifier-timeout; TMax's own 600 s / 120 s are kept under [metadata]).
  instruction.md, tests/   verbatim. With --gate <allenai/TMax-15K parquet>: tests/ also gets the task's
                           test_initial_state.py and a gate test.sh (initial state first, then the final-state tests,
                           junit XML for both), for gate.py's no-op run.

  python build_tree.py --src <TMax harbor dir> --out <tree> [--only <task list>] [--gate <TMax-15K parquet>]
      [--agent-timeout 1800] [--verifier-timeout 600]

Writes TASKS.txt, Dockerfile.jammy (build_snapshots.py --tree reads it), pool.json (snapshot name, counts per
Dockerfile shape) and coverage.tsv (every selected task: shape, steps, context bytes, status). A task is covered when
its Dockerfile is FROM ubuntu:22.04 and uses only ENV, COPY (single file) and RUN.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import importlib.util
import json
import re
import shlex
import shutil
import sys
import tomllib
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import tmax_base  # noqa: E402

_cf = importlib.util.spec_from_file_location("cf_build_tree", HERE.parent.parent / "calibforge/daytona/build_tree.py")
cf = importlib.util.module_from_spec(_cf)
_cf.loader.exec_module(cf)  # toml_value / dump_toml: the CalibForge tree's task.toml writer

TEMPLATE = (HERE / "setup_template.sh").read_text()
GATE_TEST = """#!/bin/bash
# TMax no-op gate verifier (data/tmax/daytona/build_tree.py --gate): the task's own initial-state tests, then its
# final-state tests exactly as tests/test.sh runs them. Junit XML for both goes to /logs/verifier for gate.py.
mkdir -p /logs/verifier
cd /tests
python3 -m pytest test_initial_state.py -rA -p no:cacheprovider --junitxml=/logs/verifier/initial.xml \\
  > /logs/verifier/initial-stdout.txt 2>&1
echo $? > /logs/verifier/initial_rc.txt
python3 -m pytest test_final_state.py -v -p no:cacheprovider --junitxml=/logs/verifier/final.xml 2>&1 \\
  | tee /logs/verifier/test-stdout.txt
TEST_EXIT=${PIPESTATUS[0]}
if [ $TEST_EXIT -eq 0 ]; then echo 1 > /logs/verifier/reward.txt; else echo 0 > /logs/verifier/reward.txt; fi
exit 0
"""


def expand(value: str, env: dict[str, str]) -> str:
    """Docker ENV substitution: $VAR and ${VAR} from the environment built so far (unset -> empty)."""
    return re.sub(r"\$(\w+)|\$\{(\w+)\}", lambda m: env.get(m.group(1) or m.group(2), ""), value)


def translate(dockerfile: str, envdir: Path):
    """Dockerfile -> (setup.sh step lines, final ENV added by the Dockerfile, shape, copied files) or a reason."""
    lines = [ln for ln in dockerfile.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
    if not lines or lines[0].split() != ["FROM", "ubuntu:22.04"]:
        return None, f"FROM is {lines[0] if lines else 'missing'!r}, not ubuntu:22.04"
    env = dict(tmax_base.BASE_ENV)
    added, out, shape, copied = {}, [], [], []
    for ln in lines[1:]:
        if ln.endswith("\\"):
            return None, "line continuation"
        op, _, rest = ln.partition(" ")
        if op == "ENV":
            m = re.fullmatch(r"(\w+)=(.*)", rest.strip())
            if not m or m.group(2).startswith(("'", '"')):
                return None, f"ENV form {rest!r}"
            k, v = m.group(1), expand(m.group(2), env)
            env[k] = added[k] = v
            out.append(f"setenv {shlex.quote(f'{k}={v}')}")
            shape.append("ENV")
        elif op == "COPY":
            parts = rest.split()
            if len(parts) != 2 or parts[0].startswith("--") or parts[1].endswith("/"):
                return None, f"COPY form {rest!r}"
            src = envdir / parts[0]
            if not src.is_file():
                return None, f"COPY source {parts[0]} missing"
            mode = oct(src.stat().st_mode & 0o777)[2:]
            out.append(f"copy {shlex.quote(parts[0])} {shlex.quote(parts[1])} {mode}")
            copied.append(parts[0])
            shape.append("COPY fixture" if parts[0].startswith("_fixtures/") else f"COPY {parts[0]}")
        elif op == "RUN":
            if rest.lstrip().startswith("["):
                return None, "RUN exec form"
            out.append(f"step {shlex.quote(rest)}")
            shape.append("RUN " + (rest if len(rest) < 60 else rest[:57] + "..."))
        else:
            return None, f"instruction {op}"
    collapsed = []  # a run of fixture COPYs reads as one entry
    for s in shape:
        if s == "COPY fixture" and collapsed and collapsed[-1] == "COPY fixtures":
            continue
        collapsed.append("COPY fixtures" if s == "COPY fixture" else s)
    base = [f"{k}={v}" for k, v in {"HOME": "/root", **tmax_base.BASE_ENV}.items()]
    return (out, added, " | ".join(collapsed), copied, base), ""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="laion/TMax-15K-Harbor harbor/ dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", help="file of task ids (one per line): build only these")
    ap.add_argument("--gate", help="allenai/TMax-15K parquet: add test_initial_state.py and the gate test.sh")
    ap.add_argument("--agent-timeout", type=float, default=1800.0)
    ap.add_argument("--verifier-timeout", type=float, default=600.0)
    a = ap.parse_args()

    src, out = Path(a.src), Path(a.out)
    if out.exists():
        raise SystemExit(f"{out} exists; build into a new directory")
    tasks = sorted(p.name for p in src.iterdir() if (p / "task.toml").is_file())
    if a.only:
        want = [ln.strip() for ln in open(a.only) if ln.strip()]
        if set(want) - set(tasks):
            raise SystemExit(f"{len(set(want) - set(tasks))} --only tasks are not in {src}")
        tasks = sorted(set(want))
    initial = {}
    if a.gate:
        import pandas as pd
        d = pd.read_parquet(a.gate, columns=["task_id", "test_initial_state"])
        initial = {t: s for t, s in zip(d.task_id, d.test_initial_state) if isinstance(s, str) and s.strip()}

    recipe = tmax_base.dockerfile()
    out.mkdir(parents=True)
    (out / f"Dockerfile.{tmax_base.LABEL}").write_text(recipe)
    cover, shapes, built = [], collections.Counter(), []
    for t in tasks:
        tdir, envdir = src / t, src / t / "environment"
        res, reason = translate((envdir / "Dockerfile").read_text(), envdir)
        if res is None:
            cover.append((t, "", 0, 0, reason))
            continue
        steps, added, shape, copied, base_env = res
        if a.gate and t not in initial:
            cover.append((t, shape, 0, 0, "no test_initial_state upstream"))
            continue
        dst = out / t
        shutil.copytree(tdir / "tests", dst / "tests")
        shutil.copy2(tdir / "instruction.md", dst / "instruction.md")
        ctx = dst / "setup_files" / "context"
        shutil.copytree(envdir, ctx)
        (ctx / "Dockerfile").rename(ctx / "Dockerfile.orig")
        sh = dst / "setup_files" / "setup.sh"
        sh.write_text(TEMPLATE.replace("@@TASK@@", t)
                      .replace("@@BASE_ENV@@", " ".join(shlex.quote(x) for x in base_env))
                      .replace("@@STEPS@@", "\n".join(steps)))
        sh.chmod(0o755)
        (dst / "environment").mkdir()
        (dst / "environment" / "Dockerfile").write_text(recipe)
        cfg = tomllib.loads((tdir / "task.toml").read_text())
        md = cfg.setdefault("metadata", {})
        md["tmax_agent_timeout_sec"] = cfg.get("agent", {}).get("timeout_sec")
        md["tmax_verifier_timeout_sec"] = cfg.get("verifier", {}).get("timeout_sec")
        md["tmax_daytona_base"] = tmax_base.LABEL
        md["tmax_dockerfile_shape"] = shape
        cfg.setdefault("agent", {})["timeout_sec"] = a.agent_timeout
        cfg.setdefault("verifier", {})["timeout_sec"] = a.verifier_timeout
        env = cfg.setdefault("environment", {})
        env["workdir"] = "/"
        env["env"] = added
        # dump_toml writes top-level scalars then one [section] per dict; keep TMax's section order
        (dst / "task.toml").write_text(cf.dump_toml(cfg))
        if a.gate:
            (dst / "tests" / "test_initial_state.py").write_text(initial[t])
            (dst / "tests" / "test.sh").write_text(GATE_TEST)
            (dst / "tests" / "test.sh").chmod(0o755)
        nbytes = sum(p.stat().st_size for p in ctx.rglob("*") if p.is_file())
        cover.append((t, shape, len(steps), nbytes, "covered"))
        shapes[shape] += 1
        built.append(t)

    try:
        from harbor.utils.container_cache import environment_dir_hash_truncated as h12
    except Exception:  # noqa: BLE001
        h12 = None
    names = {h12(out / t / "environment", truncate=12) for t in built} if h12 else set()
    if h12 and len(names) > 1:
        raise SystemExit(f"{len(names)} environment hashes; every task must share one snapshot")
    ctx_sizes = sorted(c[3] for c in cover if c[4] == "covered")
    pool = {tmax_base.LABEL: dict(
        base_image=tmax_base.IMAGE, name=f"harbor__{names.pop()}__snapshot" if names else None,
        dockerfile=f"Dockerfile.{tmax_base.LABEL}", dockerfile_sha256=hashlib.sha256(recipe.encode()).hexdigest(),
        task=built[0] if built else None, task_count=len(built), gate=bool(a.gate),
        agent_timeout_sec=a.agent_timeout, verifier_timeout_sec=a.verifier_timeout,
        shapes=dict(shapes.most_common()),
        context_kb_median=round(ctx_sizes[len(ctx_sizes) // 2] / 1e3, 1) if ctx_sizes else None,
        context_kb_max=round(ctx_sizes[-1] / 1e3, 1) if ctx_sizes else None)}
    (out / "pool.json").write_text(json.dumps(pool, indent=1) + "\n")
    (out / "TASKS.txt").write_text("\n".join(built) + "\n")
    with open(out / "coverage.tsv", "w") as f:
        f.write("task_id\tshape\trun_copy_env_steps\tcontext_bytes\tstatus\n")
        for c in cover:
            f.write("\t".join(map(str, c)) + "\n")
    print(json.dumps({"selected": len(cover), "covered": len(built),
                      "not covered": collections.Counter(c[4] for c in cover if c[4] != "covered"),
                      "pool": pool}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
