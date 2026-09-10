#!/usr/bin/env python3
"""curriculum_build.py <rest probe> [--reps 16] [--dst-suffix plus] — build the next arm's training tree from a rest-pool probe.

The rule (decisions.md, 2026-09-09): a task teaches only where the policy we resume from solves it 1-7 times of 8, so the
added set is selected by THAT checkpoint's probe, not the base's. Three filters, in order:

  1. learnable at the probe's checkpoint  — 1 <= succ <= scored-1, and scored must be the full 8
  2. not a no-edit-win task               — stratum "zero" in tt_v2_rest.tsv; the "noop_win" class has a winning trace that
                                            never edited /testbed, so its reward does not mean what it should
  3. not tornado or scrapy                — those two repos are the held-out OOD validation; training on them ends oodval

Writes the task list, then builds `tasks/r2egym-tt-v2-<suffix>` (fixed prompt + hardened verifier, from the raw tree) and
the combined replicated tree `tasks/r2egym-tt-v2-train-<suffix>-x<reps>` = the 728 train tasks plus the added ones, each
symlinked <reps> times in the `<task>`, `<task>__r1` … convention the existing x16 tree uses.

Python 3.9 / stdlib; runs on the Jupiter login node. Prints the composition so the launch can be checked before submitting.
"""
import argparse, collections, csv, os, subprocess, sys

E = "/e/fscratch/reformo/lee27/experiments"
T = "/e/fscratch/reformo/lee27/tasks"
C = "/e/project1/transfernetx/lee27/code/snowball"
OOD_REPOS = {"tornado", "scrapy"}

ap = argparse.ArgumentParser()
ap.add_argument("probe", help="rest-pool probe name, e.g. snowball_v2rest_fulldist_s36")
ap.add_argument("--table", default=None, help="pass@8 table (default <probe>/pass8_pass8_table.csv, else <probe>/interim/…)")
ap.add_argument("--reps", type=int, default=16)
ap.add_argument("--dst-suffix", default="plus")
ap.add_argument("--build", action="store_true", help="actually build the trees; without it only the list and the counts")
a = ap.parse_args()

tbl = a.table
if tbl is None:
    for cand in ("%s/%s/pass8_pass8_table.csv" % (E, a.probe), "%s/%s/interim/pass8_pass8_table.csv" % (E, a.probe)):
        if os.path.exists(cand):
            tbl = cand
            break
if not tbl or not os.path.exists(tbl):
    sys.exit("no pass@8 table for %s" % a.probe)

split = {r["task"]: r for r in csv.DictReader(open(E + "/tt_v2_rest.tsv"), delimiter="\t")}
rows = {r["task"]: (int(r["succ"]), int(r["scored"])) for r in csv.DictReader(open(tbl))}

full = [t for t, (_succ, n) in rows.items() if n >= 8]
learnable = [t for t in full if 1 <= rows[t][0] <= rows[t][1] - 1]
no_noop = [t for t in learnable if split.get(t, {}).get("stratum") == "zero"]
added = sorted(t for t in no_noop if split[t]["repo"] not in OOD_REPOS)

print("table            %s" % tbl)
print("fully sampled    %d" % len(full))
print("learnable (1-7)  %d" % len(learnable))
print("minus no-op wins %d" % len(no_noop))
print("minus OOD repos  %d   <- the added set" % len(added))
print("by repo          %s" % dict(collections.Counter(split[t]["repo"] for t in added).most_common()))

train = sorted(os.listdir(T + "/r2egym-tt-v2-train"))
overlap = set(train) & set(added)
assert not overlap, "added tasks overlap the train split: %s" % sorted(overlap)[:5]
print("combined         %d = %d train + %d added (+%.0f %%)" % (len(train) + len(added), len(train), len(added),
                                                                100 * len(added) / len(train)))
lst = "%s/tt_v2_curriculum_%s_final.txt" % (E, a.probe)
open(lst, "w").write("\n".join(added) + "\n")
print("list             %s" % lst)

if not a.build:
    print("\n(dry run — pass --build to create the trees)")
    sys.exit(0)

dst = "%s/r2egym-tt-v2-%s" % (T, a.dst_suffix)
print(subprocess.check_output([sys.executable, C + "/tt_prompt.py", "rewrite", "--src", T + "/r2egym-tt-raw",
                               "--dst", dst, "--allow", lst, "--test-sh", C + "/tt_test_hardened.sh", "--force"],
                              text=True).strip())

xdst = "%s/r2egym-tt-v2-train-%s-x%d" % (T, a.dst_suffix, a.reps)
if os.path.isdir(xdst):
    sys.exit("%s already exists — remove it first" % xdst)
os.makedirs(xdst)
n = 0
for src_dir, tasks in ((T + "/r2egym-tt-v2-train", train), (dst, added)):
    for t in tasks:
        for i in range(a.reps):
            name = t if i == 0 else "%s__r%d" % (t, i)
            os.symlink("%s/%s" % (src_dir, t), "%s/%s" % (xdst, name))
            n += 1
print("tree             %s (%d entries = %d tasks x %d)" % (xdst, n, len(train) + len(added), a.reps))
