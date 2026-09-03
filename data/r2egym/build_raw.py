#!/usr/bin/env python3
# Verbatim copy of /e/fscratch/reformo/lee27/build_raw.py (Jupiter, 2026-08-05), committed 2026-09-03 so the
# raw R2E-Gym rebuild no longer lives only on a 30-day purge tier. Companion: data/swesmith/build_raw.py.
"""Build a usable r2egym task tree from Marianna's UNPATCHED dataset.

WHY. Our DCAgent/r2egym-patched-full-oracle was flattened (per data/r2egym/PATCHING.md)
to collapse 8,101 Daytona snapshots into 3: the per-task prebuilt image
`FROM namanjain12/<repo>_final:<sha>` -- whose /testbed already holds the repo at the
buggy commit -- was swapped for a generic `python:X.Y` base plus an instruction preamble
telling the AGENT to git-clone the repo. Sandboxes have no network, so the clone failed,
/testbed stayed empty, and the verifier scored a stock pip-installed wheel instead of the
agent's work. Reward became a per-task constant => 0% learnable band by construction
(measured on 1248713: 61% pass, 0 of 75 groups with any within-group variance).

Her dataset at $SRC is the pre-patch original: 4,578 tasks (exactly the pool behind her
"~1.6k of 4.5k" band), Dockerfiles still FROM the prebuilt image, and 4,568 SIFs already
built in her cache. We reuse both. No Docker pulls, no SIF builds.

TWO INVARIANTS, or her SIFs stop resolving:
  * task directory names must stay identical (`r2egym-0000`, ...)
  * environment/Dockerfile must be copied BYTE-IDENTICAL
because a SIF is keyed `build_${task_name}-${sha256(Dockerfile)[:12]}.sif`.

The one thing we must add: her tasks carry no instruction.md -- the prompt lives in
environment/workspace/metadata.json:problem_statement, which her harbor fork reads and
ours does not. We generate instruction.md from it, stripping the [ISSUE] wrapper, and
state plainly that the repo is ALREADY at /testbed so no agent should try to clone.
"""
import json
import os
import re
import shutil
import sys

SRC = "/p/scratch/transfernetx/nezhurina1/r2egym_apptainer_dataset"
DST = sys.argv[1] if len(sys.argv) > 1 else "/e/scratch/reformo/lee27/tasks/r2egym-raw"
LIMIT = int(os.environ.get("LIMIT", "0"))
APPLY = "--apply" in sys.argv

HEADER = """You are working in an existing Python repository checked out at `/testbed`.
The repository is ALREADY present at the correct commit -- do NOT clone anything, and do
not expect network access. Fix the issue described below by editing the code in
`/testbed`.

"""

made = skipped = nometa = 0
tasks = sorted(os.listdir(SRC))
if LIMIT:
    tasks = tasks[:LIMIT]

for t in tasks:
    s = os.path.join(SRC, t)
    df = os.path.join(s, "environment", "Dockerfile")
    md = os.path.join(s, "environment", "workspace", "metadata.json")
    if not (os.path.isfile(df) and os.path.isfile(md)):
        nometa += 1
        continue
    try:
        meta = json.load(open(md))
    except Exception:
        nometa += 1
        continue
    ps = (meta.get("problem_statement") or "").strip()
    if not ps:
        nometa += 1
        continue
    # strip the [ISSUE]...[/ISSUE] wrapper the raw r2egym prompt carries
    body = re.sub(r"^\[ISSUE\]\s*", "", ps)
    body = re.sub(r"\s*\[/ISSUE\]\s*$", "", body).strip()

    d = os.path.join(DST, t)
    if not APPLY:
        made += 1
        continue

    os.makedirs(os.path.join(d, "environment", "workspace"), exist_ok=True)
    os.makedirs(os.path.join(d, "tests"), exist_ok=True)
    # BYTE-IDENTICAL, or the SIF hash changes and her cache misses
    shutil.copyfile(df, os.path.join(d, "environment", "Dockerfile"))
    shutil.copyfile(md, os.path.join(d, "environment", "workspace", "metadata.json"))
    for name in ("task.toml",):
        p = os.path.join(s, name)
        if os.path.isfile(p):
            shutil.copyfile(p, os.path.join(d, name))
    for name in os.listdir(os.path.join(s, "tests")):
        shutil.copyfile(os.path.join(s, "tests", name),
                        os.path.join(d, "tests", name))
    with open(os.path.join(d, "instruction.md"), "w") as f:
        f.write(HEADER + body + "\n")
    made += 1

print(f"source tasks         : {len(tasks)}")
print(f"built                : {made}")
print(f"missing metadata     : {nometa}")
print(f"dest                 : {DST}")
print("DRY RUN -- pass --apply to write" if not APPLY else "APPLIED")
