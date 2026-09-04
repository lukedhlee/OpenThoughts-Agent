#!/usr/bin/env python3
"""Whole-tree builder for TaskTrove r2egym on Daytona: every task becomes "harness-prepared" (issue-only instruction,
`setup_files/setup.sh` run by harbor before the agent — branch lukedhlee/setup-files-hook), in one of two shapes per repo:

  repo-image  (pure-Python repos; default sympy,pyramid,tornado,scrapy,datalad): TaskTrove's bucket Dockerfile + the repo
              cloned and `pip install -e .` at BUILD time → one Daytona snapshot per repo; setup.sh = `git checkout <commit>`
              + import check (2 s, no network). Validated on sympy 2026-09-03: 100/100 oracle 1.0, 100/100 pristine 0.0.
  per-rollout (everything else: pillow/pandas/numpy need a compiled build per commit; aiohttp/coveragepy/orange3 have in-tree
              C extensions that an editable master install would leave stale after checkout): Dockerfile untouched (TaskTrove's
              3 generic snapshots), setup.sh = TaskTrove's own `## Environment Setup` preamble verbatim (clone + pip per
              sandbox, ~20 s, network). Validated on 30 mixed tasks 2026-09-03 (30/30 setups ok).

  tt_daytona_tree.py --allow <allowlist> --out <dir> [--repo-image sympy,pyramid,...] [--exclude <file>] [--n-per-repo N --seed S]

Writes TASKS.txt and SNAPSHOTS.md (per repo: shape, task count, Dockerfile sha256[:12]). Same-repo repo-image Dockerfiles are
byte-identical by construction (asserted), so harbor's environment-dir hash — and thus the snapshot — is shared.
"""
import argparse, collections, csv, glob, hashlib, io, os, random, re, tarfile
import pyarrow.parquet as pq

HUB = os.path.expanduser("~/.cache/huggingface/hub")
GITHUB = {"sympy": "sympy/sympy", "tornado": "tornadoweb/tornado", "scrapy": "scrapy/scrapy", "pyramid": "Pylons/pyramid",
          "datalad": "datalad/datalad", "coveragepy": "nedbat/coveragepy", "aiohttp": "aio-libs/aiohttp", "moto": "getmoto/moto",
          "orange3": "biolab/orange3"}
MODULE = {"coveragepy": "coverage", "orange3": "Orange"}
FENCE = re.compile(r"^## Environment Setup[^\n]*\n\n```bash\n(.*?)\n```\n\n---\n\n", re.S)

ap = argparse.ArgumentParser()
ap.add_argument("--allow", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--repo-image", default="sympy,pyramid,tornado,scrapy,datalad", help="comma list of repos built as one image per repo")
ap.add_argument("--exclude", default=None); ap.add_argument("--n-per-repo", type=int, default=None); ap.add_argument("--seed", type=int, default=20260904)
ap.add_argument("--tasktrove", default=None)
ap.add_argument("--map", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "overlap", "tasktrove_v3_upstream_map.tsv"))
a = ap.parse_args()
if a.tasktrove is None:
    c = glob.glob(f"{HUB}/datasets--laion--r2egym-patched-full-oracle-v3/snapshots/*/tasks.parquet"); assert len(c) == 1; a.tasktrove = c[0]
repo_of = {r["path"]: r["repo"] for r in csv.DictReader(open(a.map), delimiter="\t")}
repo_image = {r for r in a.repo_image.split(",") if r}
for r in repo_image: assert r in GITHUB, f"no GitHub mapping for {r}"
excl = {l.strip() for l in open(a.exclude)} if a.exclude else set()
allow = [l.strip() for l in open(a.allow) if l.strip() and not l.startswith("#") and l.strip() not in excl]
by_repo = collections.defaultdict(list)
for p in allow: by_repo[repo_of[p]].append(p)
if a.n_per_repo:
    rng = random.Random(a.seed)
    by_repo = {r: sorted(rng.sample(sorted(ps), min(a.n_per_repo, len(ps)))) for r, ps in by_repo.items()}
want = sorted(p for ps in by_repo.values() for p in ps); want_set = set(want)
print(f"allowlisted {len(allow)} → building {len(want)} tasks; repo-image: {sorted(repo_image)}")

dockerfiles = collections.defaultdict(set); made = collections.Counter()
pf = pq.ParquetFile(a.tasktrove)
for rg in range(pf.num_row_groups):
    d = pf.read_row_group(rg, columns=["path", "task_binary"]).to_pydict()
    for path, blob in zip(d["path"], d["task_binary"]):
        if path not in want_set: continue
        repo = repo_of[path]; dst = os.path.join(a.out, path)
        tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz").extractall(dst, filter="data")
        ins_p = os.path.join(dst, "instruction.md"); ins = open(ins_p).read(); m = FENCE.match(ins); assert m, f"{path}: no preamble"
        os.makedirs(os.path.join(dst, "setup_files"), exist_ok=True); sh = os.path.join(dst, "setup_files", "setup.sh")
        df_p = os.path.join(dst, "environment", "Dockerfile")
        if repo in repo_image:
            cm = re.search(r"git checkout ([0-9a-f]{40})", m.group(1)); assert cm, f"{path}: no checkout commit in preamble"; commit = cm.group(1)
            df = open(df_p).read().rstrip("\n")
            # `cd /` first: the bucket Dockerfile ends in WORKDIR /testbed; deleting the cwd makes git die
            df += (f"\n\n# --- one image per repo: the repo is part of the environment, not the agent's job ---\n"
                   f"RUN cd / && rm -rf /testbed && git clone -q https://github.com/{GITHUB[repo]}.git /testbed\n"
                   f"WORKDIR /testbed\n"
                   f"RUN pip install --no-build-isolation -e . 2>/dev/null || pip install -e . 2>/dev/null || pip install --no-build-isolation . 2>/dev/null || pip install . || true\n")
            open(df_p, "w").write(df + "\n")
            open(sh, "w").write("#!/bin/bash\n# Per-task state on top of the per-repo image: check out the buggy commit. No clone, no pip, no network.\n"
                                "set -euo pipefail\ncd /testbed\n"
                                f"git checkout -q {commit}\n"
                                "git status --porcelain | head -3\n"
                                f"python -c \"import {MODULE.get(repo, repo)}\"\n"
                                "git rev-parse --verify HEAD >/dev/null\n")
        else:
            open(sh, "w").write("#!/bin/bash\n# Environment preparation moved out of the agent's instruction (TaskTrove r2egym preamble, verbatim).\n"
                                "# Run by harbor before the agent starts (setup_files/setup.sh hook). Fails the trial on a non-zero exit.\n"
                                "set -o pipefail\n" + m.group(1) + "\n"
                                "# readiness: the repo must be checked out at the requested commit\n"
                                "cd /testbed && git rev-parse --verify HEAD >/dev/null\n")
        os.chmod(sh, 0o755); open(ins_p, "w").write(ins[m.end():])
        dockerfiles[repo].add(hashlib.sha256(open(df_p, "rb").read()).hexdigest()[:12]); made[repo] += 1
for r in repo_image:
    assert len(dockerfiles.get(r, ())) <= 1, f"{r}: {len(dockerfiles[r])} distinct Dockerfiles (expected 1)"
open(os.path.join(a.out, "TASKS.txt"), "w").write("\n".join(want) + "\n")
lines = ["repo | shape | tasks | Dockerfile sha256[:12]", "---|---|---|---"]
for r in sorted(made, key=lambda r: -made[r]):
    lines.append(f"{r} | {'repo-image' if r in repo_image else 'per-rollout'} | {made[r]} | {' '.join(sorted(dockerfiles[r]))}")
lines.append(f"**all** | | {sum(made.values())} | {len(set().union(*dockerfiles.values()))} distinct Dockerfiles")
open(os.path.join(a.out, "SNAPSHOTS.md"), "w").write("\n".join(lines) + "\n"); print("\n".join(lines))
