#!/usr/bin/env python3
"""The r2egym APPTAINER task prompt: our 4-line header + TaskTrove's workflow body, minus TaskTrove's setup block.

Spec (Luke, 2026-09-06). INCLUDE our header (repo is at /testbed, ALREADY at the correct commit, do NOT clone, no
network, fix by editing code in /testbed), the GitHub issue, and TaskTrove's body from <uploaded_files> down -- "you
DON'T have to modify the testing logic or any of the tests", "the environment is already set up", the five steps
(explore -> write /testbed/reproduce_issue.py and confirm the bug -> implement -> rerun the repro + run the repo's
existing tests -> final review). EXCLUDE TaskTrove's `## Environment Setup` block (git clone + pip install -e .): the
SIF is prebuilt with /testbed at the buggy commit and the sandbox has no network; that preamble is the
data/r2egym/PATCHING.md failure (clone fails, /testbed stays empty, the verifier scores a stock wheel). TaskTrove's own
file contradicts itself there (setup block says install, body says already installed); only the body is true for us.

BODY is byte-for-byte TaskTrove's text from <uploaded_files> down (verified against laion/r2egym-patched-full-oracle-v3
task tarballs, 2026-09-06: TaskTrove = data/r2egym/utils.py:create_r2egym_instruction with the [ISSUE] wrapper stripped
from the problem statement; `--check` re-verifies). The issue sits in <issue_description>, once.

Library:  workflow_instruction(problem_statement, base_commit) -> str      (header + body, wrapper stripped)
CLI:      tt_prompt.py rewrite --src <tree> --dst <tree> [--allow <list>] [--parents <tsv>] [--test-sh <file>]
              copy a task tree and replace every instruction.md with the workflow prompt; base commit = metadata's
              tasktrove_base_commit, else <tsv> (task \t docker_image \t parent_commit ...); --test-sh also replaces
              tests/test.sh (the hardened verifier). Python 3.9 / stdlib only (runs on the Jupiter login node).
          tt_prompt.py check [--n 20]   (Mac, HF cache) compare against TaskTrove's shipped instruction.md.
"""
import argparse, json, os, re, shutil, sys

HEADER = """You are working in an existing Python repository checked out at `/testbed`.
The repository is ALREADY present at the correct commit -- do NOT clone anything, and do
not expect network access. Fix the issue described below by editing the code in
`/testbed`.

"""

# data/r2egym/utils.py:create_r2egym_instruction, verbatim (kept as a plain string: problem statements contain braces)
BODY = """<uploaded_files>
/testbed
</uploaded_files>

I've uploaded a python code repository in the directory /testbed. Consider the following issue description:

<issue_description>
{problem_statement}
</issue_description>

Can you help me implement the necessary changes to the repository so that the requirements specified in the <issue_description> are met?
I've already taken care of all changes to any of the test files described in the <issue_description>. This means you DON'T have to modify the testing logic or any of the tests in any way!
Also the development Python environment is already set up for you (i.e., all dependencies already installed), so you don't need to install other packages.
Your task is to make the minimal changes to non-test files in the /testbed directory to ensure the <issue_description> is satisfied.

Follow these steps to resolve the issue:

1. EXPLORATION: First, thoroughly explore the repository structure using tools like `find` and `grep`.
- Identify all files mentioned in the problem statement
- Locate where the issue occurs in the codebase
- Understand the surrounding context and dependencies
- Use `grep` to search for relevant functions, classes, or error messages

2. Assess whether you can reproduce the issue:
    - Create a script at /testbed/reproduce_issue.py that demonstrates the error.
    - Execute this script to confirm the error behavior.
    - You should reproduce the issue before fixing it.
    - Your reproduction script should also assert the expected behavior for the fixed code.

3. IMPLEMENTATION: Edit the source code to implement your chosen solution.
- Make minimal, focused changes to fix the issue

4. VERIFICATION: Test your implementation thoroughly.
- Run your reproduction script to verify the fix works
- Add edge cases to your test script to ensure comprehensive coverage
- Run existing tests related to the modified code to ensure you haven't broken anything

5. FINAL REVIEW: Carefully re-read the problem description and compare your changes with the base commit {base_commit}.
- Ensure you've fully addressed all requirements

Be thorough in your exploration, testing, and reasoning. It's fine if your thinking process is lengthy - quality and completeness are more important than brevity.
"""


def strip_issue_wrapper(problem_statement):
    ps = (problem_statement or "").strip()
    ps = re.sub(r"^\[ISSUE\]\s*", "", ps)
    ps = re.sub(r"\s*\[/ISSUE\]\s*$", "", ps)
    return ps.strip()


def workflow_body(problem_statement, base_commit):
    """TaskTrove's instruction from <uploaded_files> down (wrapper stripped), no header."""
    return BODY.replace("{problem_statement}", strip_issue_wrapper(problem_statement)).replace("{base_commit}", base_commit).strip() + "\n"


def workflow_instruction(problem_statement, base_commit):
    """The apptainer prompt: header + TaskTrove body. base_commit = the buggy parent (what /testbed is checked out at)."""
    assert re.fullmatch(r"[0-9a-f]{40}", base_commit or ""), "base_commit must be the 40-hex buggy parent commit: %r" % (base_commit,)
    return HEADER + workflow_body(problem_statement, base_commit)


def cmd_rewrite(a):
    parents = {}
    if a.parents:
        with open(a.parents) as f:
            hdr = f.readline().rstrip("\n").split("\t")
            for line in f:
                r = dict(zip(hdr, line.rstrip("\n").split("\t")))
                parents[r["task"]] = r["parent_commit"]
    keep = None
    if a.allow:
        keep = {l.strip() for l in open(a.allow) if l.strip() and not l.startswith("#")}
    tasks = sorted(d for d in os.listdir(a.src) if os.path.isdir(os.path.join(a.src, d)) and (keep is None or d in keep))
    if keep is not None and len(tasks) != len(keep):
        sys.exit("allowlist has %d tasks, %d found in %s" % (len(keep), len(tasks), a.src))
    if os.path.exists(a.dst):
        if not a.force: sys.exit("%s exists (pass --force to replace)" % a.dst)
        shutil.rmtree(a.dst)
    os.makedirs(a.dst)
    test_sh = open(a.test_sh, "rb").read() if a.test_sh else None
    src_base = {"tasktrove_base_commit": 0, "parents": 0}
    for t in tasks:
        shutil.copytree(os.path.join(a.src, t), os.path.join(a.dst, t), symlinks=True)
        meta = json.load(open(os.path.join(a.dst, t, "environment", "workspace", "metadata.json")))
        base = meta.get("tasktrove_base_commit")
        if base: src_base["tasktrove_base_commit"] += 1
        else:
            base = parents.get(t); src_base["parents"] += 1
        if not base: sys.exit("%s: no tasktrove_base_commit in metadata.json and no --parents row" % t)
        if not a.keep_instruction:
            with open(os.path.join(a.dst, t, "instruction.md"), "w") as f:
                f.write(workflow_instruction(meta["problem_statement"], base))
        if test_sh is not None:
            open(os.path.join(a.dst, t, "tests", "test.sh"), "wb").write(test_sh)
    print("%s %d tasks -> %s (base commit from %s; test.sh %s)" % (
        "copied (instruction kept)" if a.keep_instruction else "rewrote", len(tasks), a.dst, src_base,
        "replaced from " + a.test_sh if test_sh else "unchanged"))


def cmd_check(a):
    """Mac only: TaskTrove tarball instruction.md (fence removed) == workflow_body(upstream problem_statement, tt_base_commit)."""
    import glob, io, tarfile
    import pyarrow.parquet as pq
    HUB = os.path.expanduser("~/.cache/huggingface/hub")
    SP = os.path.dirname(os.path.abspath(__file__))
    rows = {}
    with open(os.path.join(SP, "..", "overlap", "tasktrove_v3_upstream_map.tsv")) as f:
        hdr = f.readline().rstrip("\n").split("\t")
        for line in f:
            r = dict(zip(hdr, line.rstrip("\n").split("\t"))); rows[r["path"]] = r
    want = sorted(rows)[:: max(1, len(rows) // a.n)][: a.n]
    imgs = {rows[p]["docker_image"] for p in want}
    up = {}
    for p in sorted(glob.glob(HUB + "/datasets--R2E-Gym--R2E-Gym-V1/snapshots/*/data/*.parquet")):
        for r in pq.read_table(p, columns=["docker_image", "problem_statement"]).to_pylist():
            if r["docker_image"] in imgs: up[r["docker_image"]] = r["problem_statement"]
    tt = glob.glob(HUB + "/datasets--laion--r2egym-patched-full-oracle-v3/snapshots/*/tasks.parquet")[0]
    pf = pq.ParquetFile(tt); ok = bad = 0
    for rg in range(pf.num_row_groups):
        d = pf.read_row_group(rg, columns=["path", "task_binary"]).to_pydict()
        for path, blob in zip(d["path"], d["task_binary"]):
            if path not in want: continue
            tf = tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz")
            m = [n for n in tf.getnames() if n.endswith("instruction.md")][0]
            fh = tf.extractfile(m); assert fh is not None, (path, m)
            shipped = fh.read().decode()
            shipped = shipped[shipped.index("<uploaded_files>"):]
            ours = workflow_body(up[rows[path]["docker_image"]], rows[path]["tt_base_commit"])
            norm = lambda s: "\n".join(l.rstrip() for l in s.strip().splitlines() if l.strip())
            if norm(shipped) == norm(ours): ok += 1
            else:
                bad += 1; print("MISMATCH", path)
                import difflib
                for l in list(difflib.unified_diff(ours.splitlines(), shipped.splitlines(), lineterm="", n=0))[:12]: print("   ", l)
    print("check: %d identical (modulo blank lines), %d mismatches" % (ok, bad))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("rewrite"); r.add_argument("--src", required=True); r.add_argument("--dst", required=True)
    r.add_argument("--allow", default=None); r.add_argument("--parents", default=None); r.add_argument("--test-sh", default=None)
    r.add_argument("--force", action="store_true")
    r.add_argument("--keep-instruction", action="store_true", help="control tree: copy + --test-sh only, leave instruction.md as is")
    c = sub.add_parser("check"); c.add_argument("--n", type=int, default=20)
    a = ap.parse_args()
    (cmd_rewrite if a.cmd == "rewrite" else cmd_check)(a)
