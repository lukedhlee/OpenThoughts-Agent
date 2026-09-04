#!/usr/bin/env python3
"""Turn TaskTrove r2egym tasks (flattened shape) into "harness-prepared" tasks: the `## Environment Setup` preamble that
TaskTrove puts in instruction.md (git clone, checkout, pip install) moves into `setup_files/setup.sh`, which harbor runs
before the agent starts (harbor branch lukedhlee/setup-files-hook). The Dockerfile is untouched, so Daytona still sees the
same 3 generic snapshots; the agent sees an issue-only instruction and a ready /testbed.

  tt_setup_hook_tasks.py --out <dir> [--n 30 --seed 20260904] [--exclude <file>] [--paths <file>]

Reads laion/r2egym-patched-full-oracle-v3 from the local HF cache. Fails loudly on a task whose instruction has no
recognisable preamble (the format is `## Environment Setup ...` + a ```bash fence, then a `---` rule).
"""
import argparse, glob, io, os, random, re, tarfile
import pyarrow.parquet as pq

HUB = os.path.expanduser("~/.cache/huggingface/hub")
ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True); ap.add_argument("--n", type=int, default=30); ap.add_argument("--seed", type=int, default=20260904)
ap.add_argument("--exclude", default=None, help="task names never to sample (e.g. the empty-issue list)")
ap.add_argument("--paths", default=None, help="explicit list of TaskTrove paths instead of a random sample")
ap.add_argument("--tasktrove", default=None)
a = ap.parse_args()
if a.tasktrove is None:
    c = glob.glob(f"{HUB}/datasets--laion--r2egym-patched-full-oracle-v3/snapshots/*/tasks.parquet"); assert len(c) == 1, c; a.tasktrove = c[0]
excl = {l.strip() for l in open(a.exclude)} if a.exclude else set()
pf = pq.ParquetFile(a.tasktrove)
all_paths = [p for rg in range(pf.num_row_groups) for p in pf.read_row_group(rg, columns=["path"]).to_pydict()["path"]]
if a.paths:
    want = [l.strip() for l in open(a.paths) if l.strip()]
else:
    want = sorted(random.Random(a.seed).sample(sorted(p for p in all_paths if p not in excl), a.n))
want_set = set(want)

FENCE = re.compile(r"^## Environment Setup[^\n]*\n\n```bash\n(.*?)\n```\n\n---\n\n", re.S)
made = 0
for rg in range(pf.num_row_groups):
    d = pf.read_row_group(rg, columns=["path", "task_binary"]).to_pydict()
    for path, blob in zip(d["path"], d["task_binary"]):
        if path not in want_set:
            continue
        dst = os.path.join(a.out, path)
        tf = tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz")
        tf.extractall(dst, filter="data")
        ins_p = os.path.join(dst, "instruction.md"); ins = open(ins_p).read()
        m = FENCE.match(ins)
        assert m, f"{path}: no Environment Setup preamble in instruction.md"
        os.makedirs(os.path.join(dst, "setup_files"), exist_ok=True)
        with open(os.path.join(dst, "setup_files", "setup.sh"), "w") as f:
            f.write("#!/bin/bash\n# Environment preparation moved out of the agent's instruction (TaskTrove r2egym preamble, verbatim).\n"
                    "# Run by harbor before the agent starts (setup_files/setup.sh hook). Fails the trial on a non-zero exit.\n"
                    "set -o pipefail\n" + m.group(1) + "\n"
                    "# readiness: the repo must be checked out at the requested commit\n"
                    "cd /testbed && git rev-parse --verify HEAD >/dev/null\n")
        os.chmod(os.path.join(dst, "setup_files", "setup.sh"), 0o755)
        open(ins_p, "w").write(ins[m.end():])
        made += 1
print(f"tasks: {made} -> {a.out}")
open(os.path.join(a.out, "TASKS.txt"), "w").write("\n".join(want) + "\n")
