#!/usr/bin/env python3
"""Build a SWE-rebench-V2 tree from TaskTrove's shared-image tasks, one tree per base image.

TaskTrove puts clone, checkout and install in the agent's instruction. Here the harness runs them from setup_files before
the agent starts, so the agent gets a ready /testbed, as in SWE-bench. The setup also removes everything the agent could
read the fix from: origin, every other ref and all objects not reachable from the base commit. The verifier only needs
/testbed to be a worktree containing the base commit.

  python build_rebench_tree.py --src <dir of tasks> --only <list> --out <tree> [--label go]
Writes pool.json, TASKS.txt, coverage.tsv, Dockerfile.<label>.
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import shlex
import shutil
import tomllib
from pathlib import Path

HERE = Path(__file__).resolve().parent
import importlib.util  # noqa: E402

_cf = importlib.util.spec_from_file_location("cf_build_tree", HERE.parent / "calibforge/daytona/build_tree.py")
cf = importlib.util.module_from_spec(_cf)
_cf.loader.exec_module(cf)

PREAMBLE = re.compile(r"\A## Environment Setup \(complete these steps first\)\n+```bash\n(.*?)\n```\n+---\n+", re.S)
TEST_RUN = re.compile(r"(^|&&|;|\|\|)\s*(pytest|tox|nox|python3? -m (pytest|unittest))(\s|$)")
SETUP = """#!/bin/bash
# Harness-run setup: the task's own install steps, then a git scrub, then /setup_files is removed.
set -uo pipefail
cd /testbed || exit 3
run() {{
{body}
}}
run > /tmp/.rebench_setup.log 2>&1
rc=$?
# Some install steps fail harmlessly; the no-op and gold gates judge the environment, and a bad checkout stops below.
echo "rebenchsetup: install exited $rc"
rm -f /tmp/.rebench_setup.log
cd /testbed
base={sha}
[ "$(git rev-parse HEAD)" = "$base" ] || {{ echo "rebenchsetup: HEAD is not the base commit" >&2; exit 12; }}
git checkout -q --detach "$base"
git remote | xargs -r -n1 git remote remove
git for-each-ref --format='%(refname)' | xargs -r -n1 git update-ref -d
rm -f .git/FETCH_HEAD .git/ORIG_HEAD .git/packed-refs
git reflog expire --expire=now --all
git gc -q --prune=now
[ "$(git log --all --format=%H -1)" = "$base" ] || {{ echo "rebenchsetup: history past the base commit remains" >&2; exit 13; }}
{persist}
cd / && rm -rf /setup_files
echo "TMAXSETUP {{\\"task\\": \\"{task}\\", \\"run_steps\\": 1, \\"killed\\": 0}}"
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True); ap.add_argument("--only", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--agent-timeout", type=float, default=1800.0); ap.add_argument("--verifier-timeout", type=float, default=900.0)
    a = ap.parse_args()
    src, out = Path(a.src), Path(a.out)
    if out.exists():
        raise SystemExit(f"{out} exists; build into a new directory")
    out.mkdir(parents=True)
    built, cover, recipes = [], [], set()
    for t in Path(a.only).read_text().split():
        tdir = src / t
        text = (tdir / "instruction.md").read_text()
        m = PREAMBLE.match(text)
        sha = re.search(r"git checkout ([0-9a-f]{40})", m.group(1)) if m else None
        if not sha:
            cover.append((t, "setup preamble not recognised"))
            continue
        # The upstream steps can end by running the tests; a setup must not, since the target tests fail by design.
        lines = [ln for ln in m.group(1).splitlines() if not TEST_RUN.search(ln)]
        exports = [ln.strip() for ln in lines if ln.strip().startswith("export ") and "DEBIAN_FRONTEND" not in ln]
        body = "\n".join("  " + ln for ln in lines)
        persist = "\n".join(f"echo {shlex.quote(ln)} >> /root/.bashrc" for ln in exports)
        dst = out / t
        shutil.copytree(tdir / "tests", dst / "tests"); shutil.copytree(tdir / "solution", dst / "solution")
        shutil.copytree(tdir / "environment", dst / "environment")
        recipes.add((dst / "environment" / "Dockerfile").read_text())
        (dst / "instruction.md").write_text(text[m.end():])
        (dst / "setup_files").mkdir()
        sh = dst / "setup_files" / "setup.sh"
        sh.write_text(SETUP.format(body=body, sha=sha.group(1), persist=persist or ":", task=t)); sh.chmod(0o755)
        cfg = tomllib.loads((tdir / "task.toml").read_text())
        cfg.setdefault("agent", {})["timeout_sec"] = a.agent_timeout
        cfg.setdefault("verifier", {})["timeout_sec"] = a.verifier_timeout
        cfg.setdefault("environment", {})["workdir"] = "/testbed"
        (dst / "task.toml").write_text(cf.dump_toml(cfg))
        built.append(t); cover.append((t, "covered"))
    from harbor.utils.container_cache import environment_dir_hash_truncated as h12
    names = {f"harbor__{h12(out / t / 'environment', truncate=12)}__snapshot" for t in built}
    if len(names) != 1 or len(recipes) != 1:
        raise SystemExit(f"{len(names)} environment hashes; every task must share one snapshot")
    (out / f"Dockerfile.{a.label}").write_text(recipes.pop())
    (out / "pool.json").write_text(json.dumps({a.label: dict(name=names.pop(), task_count=len(built), gate=True)}, indent=1) + "\n")
    (out / "TASKS.txt").write_text("\n".join(built) + "\n")
    (out / "coverage.tsv").write_text("task_id\tstatus\n" + "".join(f"{t}\t{s}\n" for t, s in cover))
    print(json.dumps({"selected": len(cover), "covered": len(built), "not covered": collections.Counter(s for _, s in cover if s != "covered"), "pool": json.loads((out / "pool.json").read_text())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
