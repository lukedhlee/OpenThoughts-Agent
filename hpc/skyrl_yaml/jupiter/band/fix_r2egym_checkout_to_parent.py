#!/usr/bin/env python3
"""Rewrite each r2egym task's instruction.md to check out the PARENT of base_commit.

WHY. Every task's instruction.md opens with an "Environment Setup" block telling the
agent to populate the empty /testbed itself:

    cd /testbed
    git clone https://github.com/<owner>/<repo>.git . && git checkout <base_commit>

But `base_commit` is the FIX commit, not the buggy parent. Proven per-file: every file
under solution/patched_files is byte-identical to the repo content AT base_commit, and
solve.sh "solves" a task by copying those files into /testbed -- which would be a no-op
if /testbed were already at base_commit. Confirmed on numpy/aiohttp/pandas tasks.

So an agent that follows the instructions verbatim checks out the already-fixed source,
the graded r2e_tests pass with zero work, and the reward becomes a per-task constant
that is independent of the model. Measured on job 1248713: 61% trial pass rate with
0 of 75 fully-sampled pass@4 groups showing any within-group variance.

Rewriting `git checkout <sha>` -> `git checkout <sha>~1` puts /testbed in the buggy
state the task intends. The graded tests come from /setup_files/r2e_tests (copied in by
a later line of the same instruction block), so using the parent's older in-repo test
files is harmless -- they are not what is scored.

This is not a divergence from the reference setup: r2egym proper ships a prebuilt image
whose /testbed is ALREADY at the buggy state, so the clone-it-yourself preamble is an
artifact of how DCAgent/r2egym-patched-full-oracle was generated.

Idempotent: a checkout already ending in ~1 is left alone.
"""
import json
import os
import re
import sys

TASKS = sys.argv[1] if len(sys.argv) > 1 else "/e/scratch/reformo/lee27/tasks/r2egym-patched-full-oracle"
APPLY = "--apply" in sys.argv

changed = already = nosha = nomatch = 0
mismatch = []

for name in sorted(os.listdir(TASKS)):
    d = os.path.join(TASKS, name)
    inst = os.path.join(d, "instruction.md")
    info = os.path.join(d, "setup_files", "test_info.json")
    if not (os.path.isfile(inst) and os.path.isfile(info)):
        continue
    try:
        sha = json.load(open(info)).get("base_commit")
    except Exception:
        nosha += 1
        continue
    if not sha:
        nosha += 1
        continue

    text = open(inst).read()
    if re.search(r"git checkout\s+" + re.escape(sha) + r"~1\b", text):
        already += 1
        continue
    pat = re.compile(r"(git checkout\s+)" + re.escape(sha) + r"(?![~^0-9a-f])")
    new, n = pat.subn(r"\g<1>" + sha + "~1", text)
    if n == 0:
        # instruction references a different sha than test_info.json -- do not guess
        if "git checkout" in text:
            mismatch.append(name)
        nomatch += 1
        continue
    changed += 1
    if APPLY:
        with open(inst, "w") as f:
            f.write(new)

print(f"tasks rewritten        : {changed}")
print(f"already had ~1         : {already}")
print(f"no base_commit         : {nosha}")
print(f"no matching checkout   : {nomatch}")
if mismatch:
    print(f"MISMATCH (instruction sha != test_info sha), first 10: {mismatch[:10]}")
print("DRY RUN -- pass --apply to write" if not APPLY else "APPLIED")
