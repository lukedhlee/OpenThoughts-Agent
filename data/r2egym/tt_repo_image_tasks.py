#!/usr/bin/env python3
"""ONE image per repo for TaskTrove r2egym (Daytona-friendly middle path, SWE-smith's shape).

TaskTrove's flattened tasks use 3 generic images and make the AGENT clone + install the repo per rollout (or, with the
harbor setup hook, the harness does it per rollout — still a clone and a pip install per sandbox, still network at
solve time). This builder writes, for one repo, tasks whose Dockerfile is TaskTrove's own bucket Dockerfile PLUS the
repo cloned and installed at build time — byte-identical across the repo's tasks, so Daytona needs exactly ONE snapshot
per repo — and whose `setup_files/setup.sh` only does `git checkout <base_commit>` (seconds, no network) before the agent
starts. Instruction = issue only. tests/, solution/ (oracle = overwrite with fix-commit files) and task.toml are kept as
TaskTrove ships them, so the verifier and the oracle are unchanged.

Fits repos whose install is cheap and commit-independent (pure Python: sympy, tornado, scrapy, pyramid, datalad,
coveragepy, aiohttp, moto). Compiled repos (pandas, numpy, pillow) need a rebuild per commit and are out of scope here.

  tt_repo_image_tasks.py --repo sympy --out <dir> [--n 100 --seed 20260904] [--allow <file>] [--exclude <file>]
"""
import argparse, glob, io, os, random, re, tarfile
import pyarrow.parquet as pq

HUB = os.path.expanduser("~/.cache/huggingface/hub")
GITHUB = {"sympy": "sympy/sympy", "tornado": "tornadoweb/tornado", "scrapy": "scrapy/scrapy", "pyramid": "Pylons/pyramid",
          "datalad": "datalad/datalad", "coveragepy": "nedbat/coveragepy", "aiohttp": "aio-libs/aiohttp", "moto": "getmoto/moto",
          "orange3": "biolab/orange3"}
ap = argparse.ArgumentParser()
ap.add_argument("--repo", required=True, choices=sorted(GITHUB)); ap.add_argument("--out", required=True)
ap.add_argument("--n", type=int, default=100); ap.add_argument("--seed", type=int, default=20260904)
ap.add_argument("--allow", default=None, help="only sample from these task names (e.g. the gate allowlist)")
ap.add_argument("--exclude", default=None); ap.add_argument("--tasktrove", default=None)
ap.add_argument("--map", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "overlap", "tasktrove_v3_upstream_map.tsv"))
a = ap.parse_args()
if a.tasktrove is None:
    c = glob.glob(f"{HUB}/datasets--laion--r2egym-patched-full-oracle-v3/snapshots/*/tasks.parquet"); assert len(c) == 1; a.tasktrove = c[0]
import csv
repo_paths = [r["path"] for r in csv.DictReader(open(a.map), delimiter="\t") if r["repo"] == a.repo]
allow = {l.strip() for l in open(a.allow)} if a.allow else None
excl = {l.strip() for l in open(a.exclude)} if a.exclude else set()
pool = sorted(p for p in repo_paths if (allow is None or p in allow) and p not in excl)
want = sorted(random.Random(a.seed).sample(pool, min(a.n, len(pool)))); want_set = set(want)
print(f"{a.repo}: {len(repo_paths)} tasks, eligible {len(pool)}, sampled {len(want)}")

FENCE = re.compile(r"^## Environment Setup[^\n]*\n\n```bash\n(.*?)\n```\n\n---\n\n", re.S)
pf = pq.ParquetFile(a.tasktrove); made = 0; dockerfiles = set()
for rg in range(pf.num_row_groups):
    d = pf.read_row_group(rg, columns=["path", "task_binary"]).to_pydict()
    for path, blob in zip(d["path"], d["task_binary"]):
        if path not in want_set:
            continue
        dst = os.path.join(a.out, path)
        tf = tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz"); tf.extractall(dst, filter="data")
        ins_p = os.path.join(dst, "instruction.md"); ins = open(ins_p).read(); m = FENCE.match(ins); assert m, path
        commit = re.search(r"git checkout ([0-9a-f]{40})", m.group(1)).group(1)
        df_p = os.path.join(dst, "environment", "Dockerfile"); df = open(df_p).read().rstrip("\n")
        # the repo baked into the image: cloned once at build time, installed editable so a later checkout is picked up
        df += (f"\n\n# --- one image per repo: the repo is part of the environment, not the agent's job ---\n"
               f"RUN rm -rf /testbed && git clone https://github.com/{GITHUB[a.repo]}.git /testbed\n"
               f"WORKDIR /testbed\n"
               f"RUN pip install --no-build-isolation -e . 2>/dev/null || pip install -e . 2>/dev/null || pip install --no-build-isolation . 2>/dev/null || pip install . || true\n")
        open(df_p, "w").write(df + "\n"); dockerfiles.add(df)
        os.makedirs(os.path.join(dst, "setup_files"), exist_ok=True)
        with open(os.path.join(dst, "setup_files", "setup.sh"), "w") as f:
            f.write("#!/bin/bash\n# Per-task state on top of the per-repo image: check out the buggy commit. No clone, no pip, no network.\n"
                    "set -euo pipefail\ncd /testbed\n"
                    f"git checkout -q {commit}\n"
                    "git status --porcelain | head -3\n"
                    f"python -c \"import {a.repo}\"\n"
                    "git rev-parse --verify HEAD >/dev/null\n")
        os.chmod(os.path.join(dst, "setup_files", "setup.sh"), 0o755)
        open(ins_p, "w").write(ins[m.end():]); made += 1
assert len(dockerfiles) == 1, f"expected one Dockerfile for the repo, got {len(dockerfiles)}"
open(os.path.join(a.out, "TASKS.txt"), "w").write("\n".join(want) + "\n")
print(f"tasks: {made} -> {a.out}; distinct Dockerfiles: {len(dockerfiles)} (one Daytona snapshot)")
