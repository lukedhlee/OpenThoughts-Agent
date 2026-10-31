#!/usr/bin/env python3
"""Build a Terminal-Lego task tree on the shared TMax jammy snapshot, replaying each task's Dockerfile at start.

Same pattern as data/tmax/daytona/build_tree.py, whose setup template and base recipe are reused. Terminal-Lego adds
three Dockerfile forms TMax never has: WORKDIR, relative COPY destinations and directory COPY, plus line continuations.
Directory COPY expands to one `copy` per file so every file keeps its mode through harbor's upload.

Only `FROM ubuntu:22.04` tasks are covered. The environment/Dockerfile is byte-identical to TMax's, so harbor's hash
gives the existing harbor__05b5bff9013d__snapshot and nothing is built. solution/ stays for the oracle gate; harbor only
uploads it for the oracle agent.

  python build_tlego_tree.py --src <dir of task_NNNNN> --only <list> --out <tree> [--agent-timeout 1800] [--verifier-timeout 600]

Writes pool.json, TASKS.txt, coverage.tsv.
"""
from __future__ import annotations

import argparse
import collections
import json
import posixpath
import re
import shlex
import shutil
import tomllib
from pathlib import Path

HERE = Path(__file__).resolve().parent
TMAX = HERE.parent / "tmax" / "daytona"

import importlib.util  # noqa: E402
import sys  # noqa: E402

sys.path.insert(0, str(TMAX))
_t = importlib.util.spec_from_file_location("tmax_build_tree", TMAX / "build_tree.py")
tmax = importlib.util.module_from_spec(_t)
_t.loader.exec_module(tmax)

SNAPSHOT = "harbor__05b5bff9013d__snapshot"
TEMPLATE = (TMAX / "setup_template.sh").read_text().replace(
    '( cd / && exec env -i', '( cd "$WD" && exec env -i'
).replace("NSTEP=0; NKILL=0\n", 'NSTEP=0; NKILL=0\nWD=/\nworkdir() { mkdir -p "$1" && chown 0:0 "$1" && WD=$1; }\n')
IGNORED = {"CMD", "ENTRYPOINT", "EXPOSE", "LABEL"}


def logical_lines(dockerfile: str) -> list[str]:
    out, cur = [], ""
    for ln in dockerfile.splitlines():
        if ln.lstrip().startswith("#") or (not ln.strip() and not cur):
            continue
        if ln.rstrip().endswith("\\"):
            cur += ln.rstrip()[:-1] + " "
            continue
        out.append(cur + ln)
        cur = ""
    if cur:
        out.append(cur)
    return out


def parse_env(rest: str) -> list[tuple[str, str]] | None:
    m = re.fullmatch(r"(\w+)\s+(?!\S*=)(.*)", rest.strip())
    if m:
        return [(m.group(1), m.group(2))]
    try:
        toks = shlex.split(rest)
    except ValueError:
        return None
    pairs = [t.split("=", 1) for t in toks]
    return [(k, v) for k, v in pairs] if pairs and all(len(p) == 2 for p in pairs) else None


def translate(dockerfile: str, envdir: Path):
    """Dockerfile -> (setup steps, added env, final workdir, ignored instructions) or (None, reason)."""
    lines = logical_lines(dockerfile)
    if not lines or lines[0].split() != ["FROM", "ubuntu:22.04"]:
        return None, f"FROM is {lines[0] if lines else 'missing'!r}, not ubuntu:22.04"
    env = dict(tmax.tmax_base.BASE_ENV)
    added, out, ignored = {}, [], []
    wd = "/"
    for ln in lines[1:]:
        op, _, rest = ln.strip().partition(" ")
        rest = rest.strip()
        if op == "ENV":
            pairs = parse_env(rest)
            if not pairs:
                return None, f"ENV form {rest!r}"
            for k, v in pairs:
                v = tmax.expand(v, env)
                env[k] = added[k] = v
                out.append(f"setenv {shlex.quote(f'{k}={v}')}")
        elif op == "WORKDIR":
            wd = posixpath.normpath(posixpath.join(wd, tmax.expand(rest, env)))
            out.append(f"workdir {shlex.quote(wd)}")
        elif op == "COPY":
            parts = rest.split()
            if len(parts) != 2 or parts[0].startswith("--") or "*" in parts[0]:
                return None, f"COPY form {rest!r}"
            src = (envdir / parts[0]).resolve()
            if envdir.resolve() not in (src, *src.parents):
                return None, "COPY source outside the context"
            dst = parts[1]
            if not dst.startswith("/"):
                dst = posixpath.join(wd, dst)
            if src.is_file():
                if dst.endswith("/"):
                    dst += src.name
                rel = src.relative_to(envdir.resolve()).as_posix()
                out.append(f"copy {shlex.quote(rel)} {shlex.quote(posixpath.normpath(dst))} {oct(src.stat().st_mode & 0o777)[2:]}")
            elif src.is_dir():
                # harbor's upload drops empty directories, so they are recreated by the replay
                for d in sorted(p for p in (src, *src.rglob("*")) if p.is_dir() and not any(p.iterdir())):
                    target = posixpath.normpath(posixpath.join(dst, d.relative_to(src).as_posix()))
                    out.append(f"step {shlex.quote(f'mkdir -p {shlex.quote(target)}')}")
                for f in sorted(p for p in src.rglob("*") if p.is_file()):
                    rel = f.relative_to(envdir.resolve()).as_posix()
                    target = posixpath.normpath(posixpath.join(dst, f.relative_to(src).as_posix()))
                    out.append(f"copy {shlex.quote(rel)} {shlex.quote(target)} {oct(f.stat().st_mode & 0o777)[2:]}")
            else:
                return None, f"COPY source {parts[0]} missing"
        elif op == "RUN":
            if rest.startswith("[") or "<<" in rest:
                return None, "RUN exec or heredoc form"
            out.append(f"step {shlex.quote(rest)}")
        elif op in IGNORED:
            ignored.append(op)
        else:
            return None, f"instruction {op}"
    return (out, added, wd, ignored), ""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", required=True)
    ap.add_argument("--agent-timeout", type=float, default=1800.0)
    ap.add_argument("--verifier-timeout", type=float, default=600.0)
    a = ap.parse_args()
    src, out = Path(a.src), Path(a.out)
    if out.exists():
        raise SystemExit(f"{out} exists; build into a new directory")
    recipe = tmax.tmax_base.dockerfile()
    base_env = [f"{k}={v}" for k, v in {"HOME": "/root", **tmax.tmax_base.BASE_ENV}.items()]
    out.mkdir(parents=True)
    cover, built = [], []
    for t in [x for x in Path(a.only).read_text().split() if x]:
        tdir, envdir = src / t, src / t / "environment"
        res, reason = translate((envdir / "Dockerfile").read_text(), envdir)
        if res is None:
            cover.append((t, reason))
            continue
        if not (tdir / "solution" / "solve.sh").is_file():
            cover.append((t, "no solution/solve.sh"))
            continue
        steps, added, wd, ignored = res
        dst = out / t
        shutil.copytree(tdir / "tests", dst / "tests")
        shutil.copytree(tdir / "solution", dst / "solution")
        shutil.copy2(tdir / "instruction.md", dst / "instruction.md")
        ctx = dst / "setup_files" / "context"
        shutil.copytree(envdir, ctx)
        (ctx / "Dockerfile").rename(ctx / "Dockerfile.orig")
        sh = dst / "setup_files" / "setup.sh"
        sh.write_text(TEMPLATE.replace("@@TASK@@", t).replace("@@BASE_ENV@@", " ".join(shlex.quote(x) for x in base_env))
                      .replace("@@STEPS@@", "\n".join(steps)))
        sh.chmod(0o755)
        (dst / "environment").mkdir()
        (dst / "environment" / "Dockerfile").write_text(recipe)
        cfg = tomllib.loads((tdir / "task.toml").read_text())
        md = cfg.setdefault("metadata", {})
        md["tlego_agent_timeout_sec"] = cfg.get("agent", {}).get("timeout_sec")
        md["tlego_verifier_timeout_sec"] = cfg.get("verifier", {}).get("timeout_sec")
        md["tlego_ignored_instructions"] = sorted(set(ignored))
        cfg.setdefault("agent", {})["timeout_sec"] = a.agent_timeout
        cfg.setdefault("verifier", {})["timeout_sec"] = a.verifier_timeout
        env = cfg.setdefault("environment", {})
        env["workdir"] = wd
        env["env"] = added
        (dst / "task.toml").write_text(tmax.cf.dump_toml(cfg))
        cover.append((t, "covered"))
        built.append(t)

    from harbor.utils.container_cache import environment_dir_hash_truncated as h12
    names = {f"harbor__{h12(out / t / 'environment', truncate=12)}__snapshot" for t in built}
    if names - {SNAPSHOT}:
        raise SystemExit(f"environment hashes {names}: must be exactly {SNAPSHOT}")
    (out / "pool.json").write_text(json.dumps({"jammy": dict(name=SNAPSHOT, base_image=tmax.tmax_base.IMAGE, task_count=len(built),
                                                              gate=True, agent_timeout_sec=a.agent_timeout,
                                                              verifier_timeout_sec=a.verifier_timeout)}, indent=1) + "\n")
    (out / "TASKS.txt").write_text("\n".join(built) + "\n")
    (out / "coverage.tsv").write_text("task_id\tstatus\n" + "".join(f"{t}\t{s}\n" for t, s in cover))
    print(json.dumps({"selected": len(cover), "covered": len(built),
                      "not covered": collections.Counter(s.split("'")[0][:40] for _, s in cover if s != "covered")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
