#!/usr/bin/env python3
"""p2o_trees.py: the six-variant task tree for P2O wave 0.

For every task in dev120.tsv, six copies of the 09-06 header tree's task dir (r2egym-tt-hd300, hardened tests/test.sh):
  <task>-pctl   instruction.md byte-identical to the source
  <task>-p{A..E} instruction.md = source + DELIM + block + "\n"   (strict suffix; the strip in Stage 3 is exact)
Variant suffix uses "-p<arm>" (no "__": pass8_table.py / wf_feat.py split trial dirs on "__").
Then verifies: for each task, all six dirs are byte-identical except instruction.md, and each variant's instruction.md
starts with the control's bytes and ends with exactly DELIM + block + "\n". Also writes the three shard allowlists
(tasks round-robin into a/b/c, all six variants of a task in the same shard, so every arm runs under the same load).
Python 3.9 / stdlib (Jupiter login node).
"""
import argparse, csv, hashlib, json, os, shutil, sys
E = "/e/fscratch/reformo/lee27/experiments"; T = "/e/fscratch/reformo/lee27/tasks"
ap = argparse.ArgumentParser()
ap.add_argument("--dev", default=E + "/p2o/dev120.tsv")
ap.add_argument("--src", default=T + "/r2egym-tt-hd300")
ap.add_argument("--src2", default=T + "/r2egym-tt-v2-rest", help="fallback source tree (rest-pool zero tasks)")
ap.add_argument("--dst", default=T + "/p2o6")
ap.add_argument("--blocks", default=E + "/p2o/blocks.json")
ap.add_argument("--shards", type=int, default=3)
a = ap.parse_args()
B = json.load(open(a.blocks)); DELIM = B["delim"]; blocks = B["blocks"]
ARMS = ["ctl"] + sorted(blocks)
tasks = [r["task"] for r in csv.DictReader(open(a.dev), delimiter="\t")]
if os.path.isdir(a.dst):
    shutil.rmtree(a.dst)
os.makedirs(a.dst)


def tree_hash(d, skip="instruction.md"):
    h = hashlib.md5()
    for root, dirs, files in os.walk(d):
        dirs.sort()
        for f in sorted(files):
            rel = os.path.relpath(os.path.join(root, f), d)
            if rel == skip:
                continue
            h.update(rel.encode()); h.update(open(os.path.join(root, f), "rb").read())
    return h.hexdigest()


bad = 0
for t in tasks:
    src = os.path.join(a.src, t)
    if not os.path.isdir(src):
        src = os.path.join(a.src2, t)
    if not os.path.isdir(src):
        sys.exit("missing source task %s" % t)
    ctl_bytes = open(os.path.join(src, "instruction.md"), "rb").read()
    hashes = set()
    for arm in ARMS:
        d = os.path.join(a.dst, "%s-p%s" % (t, arm))
        shutil.copytree(src, d)
        if arm != "ctl":
            with open(os.path.join(d, "instruction.md"), "ab") as f:
                f.write((DELIM + blocks[arm] + ("" if blocks[arm].endswith("\n") else "\n")).encode())
        got = open(os.path.join(d, "instruction.md"), "rb").read()
        if arm == "ctl":
            ok = got == ctl_bytes
        else:
            tail = (DELIM + blocks[arm] + ("" if blocks[arm].endswith("\n") else "\n")).encode()
            ok = got.startswith(ctl_bytes) and got[len(ctl_bytes):] == tail
        if not ok:
            bad += 1; print("BYTE CHECK FAILED", t, arm)
        hashes.add(tree_hash(d))
    if len(hashes) != 1:
        bad += 1; print("NON-INSTRUCTION FILES DIFFER", t)
print("tasks %d x arms %d = %d dirs under %s; byte-identity failures: %d" % (len(tasks), len(ARMS), len(tasks) * len(ARMS), a.dst, bad))
for i in range(a.shards):
    names = ["%s-p%s" % (t, arm) for t in tasks[i::a.shards] for arm in ARMS]
    p = E + "/p2o/p2o6_shard%s.txt" % "abc"[i]
    open(p, "w").write("\n".join(names) + "\n")
    print("shard %s: %d tasks x %d arms = %d dirs -> %s" % ("abc"[i], len(tasks[i::a.shards]), len(ARMS), len(names), p))
for arm in sorted(blocks):
    print("block %s: %d words, %d bytes" % (arm, len(blocks[arm].split()), len(blocks[arm].encode())))
sys.exit(1 if bad else 0)
