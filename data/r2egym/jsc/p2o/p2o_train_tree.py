#!/usr/bin/env python3
"""p2o_train_tree.py: a prompted copy of a training task tree for context distillation (P2O Stage 3).

Every task dir of --src is copied to --dst under the SAME name (so uids pair with the control arm, and the trainer's
task-level bookkeeping is unchanged) with one block appended to instruction.md as a strict suffix behind the delimiter,
the wave-0 method (p2o_trees.py). Per task it verifies that the two dirs are byte-identical except instruction.md, that
the new instruction.md starts with the source bytes and ends with DELIM + block + "\\n", and that the delimiter occurs
exactly once in the file (the trainer's strip, trainer.algorithm.context_distillation, requires exactly one start marker
and an end marker after it; terminus-2 supplies the end marker "\\n\\nCurrent terminal state:"). Python 3.9 / stdlib.

  python3 p2o_train_tree.py --src $T/r2egym-tt-v2-train-basecurr-x16 --block A      # -> $T/r2egym-tt-v2-train-basecurr-x16-pA
"""
import argparse, hashlib, json, os, shutil, sys

E = "/e/fscratch/reformo/lee27/experiments"
T = "/e/fscratch/reformo/lee27/tasks"
ap = argparse.ArgumentParser()
ap.add_argument("--src", default=T + "/r2egym-tt-v2-train-basecurr-x16")
ap.add_argument("--dst", default=None, help="default: <src>-p<block>")
ap.add_argument("--blocks", default=E + "/p2o/blocks.json")
ap.add_argument("--block", default="A")
ap.add_argument("--force", action="store_true", help="rebuild an existing --dst")
a = ap.parse_args()
B = json.load(open(a.blocks))
DELIM, block = B["delim"], B["blocks"][a.block]
suffix = (DELIM + block + ("" if block.endswith("\n") else "\n")).encode()
dst = a.dst or (a.src.rstrip("/") + "-p" + a.block)
if os.path.isdir(dst):
    if not a.force:
        sys.exit("%s exists (use --force to rebuild)" % dst)
    shutil.rmtree(dst)
os.makedirs(dst)


def tree_hash(d, skip="instruction.md"):
    h = hashlib.md5()
    for root, dirs, files in os.walk(d):
        dirs.sort()
        for f in sorted(files):
            rel = os.path.relpath(os.path.join(root, f), d)
            if rel == skip:
                continue
            h.update(rel.encode())
            h.update(open(os.path.join(root, f), "rb").read())
    return h.hexdigest()


tasks = sorted(t for t in os.listdir(a.src) if os.path.isdir(os.path.join(a.src, t)))
bad = 0
for t in tasks:
    s, d = os.path.join(a.src, t), os.path.join(dst, t)
    src_bytes = open(os.path.join(s, "instruction.md"), "rb").read()
    if DELIM.encode() in src_bytes:
        print("SOURCE ALREADY CARRIES THE DELIMITER:", t)
        bad += 1
        continue
    shutil.copytree(s, d)
    with open(os.path.join(d, "instruction.md"), "ab") as f:
        f.write(suffix)
    got = open(os.path.join(d, "instruction.md"), "rb").read()
    ok = got.startswith(src_bytes) and got.endswith(suffix) and got.count(DELIM.encode()) == 1
    ok = ok and tree_hash(s) == tree_hash(d)
    if not ok:
        print("VERIFY FAILED:", t)
        bad += 1
print("built %s: %d tasks, block %s (%d bytes appended), %d failures" % (dst, len(tasks), a.block, len(suffix), bad))
sys.exit(1 if bad else 0)
