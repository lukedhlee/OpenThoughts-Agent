#!/usr/bin/env python3
"""tt_split.py — the TaskTrove r2egym train / ID-val / OOD-val / held-out split (2026-09-06, Luke's design).

Facts the split rests on (verified 2026-09-06): the 993-task clean band holds EVERY fully-sampled task the base model solved 1-7
times of 8 in tt60k (the 102 others are the dropped no-op wins), so a held-out "sometimes solved" set does not exist outside the
band; no two TaskTrove tasks share a docker image or an issue text.

  oodval   = band tasks of the held-out repos (default tornado + scrapy)           -> new arm only; OOD-repo transfer
  idval    = 150 band tasks of the remaining repos, stratified by repo x tt60k succ -> new arm only; ID transfer
  train    = the rest of the band                                                  -> the new arm's training set
  heldout  = today's 100 zero-stratum probe tasks + 60 more never-solved tasks (band repo mix) + the 16 always-solved
             -> held out from EVERY arm so far (existing checkpoints included); ID val for arm 3 / fulldist, regression guard
All four are disjoint; every tree carries the fixed-clauses prompt and the hardened verifier (tt_prompt.py rewrite).
Writes experiments/tt_v2_{train,idval,oodval,heldout}.txt + tt_v2_split.tsv and builds tasks/r2egym-tt-v2-{train,idval,oodval,heldout}.
Python 3.9 / stdlib (Jupiter login node).
"""
import argparse, collections, csv, glob, json, os, random, subprocess, sys
E = "/e/fscratch/reformo/lee27/experiments"; T = "/e/fscratch/reformo/lee27/tasks"; C = "/e/project1/transfernetx/lee27/code/snowball"
ap = argparse.ArgumentParser()
ap.add_argument("--band", default=T + "/r2egym-tt-band60k-clean"); ap.add_argument("--raw", default=T + "/r2egym-tt-raw")
ap.add_argument("--tables", default=E + "/tt60k_s*/pass8_pass8_table.csv"); ap.add_argument("--sample", default=E + "/ttwf_sample.tsv")
ap.add_argument("--zero-pool", default=E + "/tt_pool_zero.txt"); ap.add_argument("--solved-pool", default=E + "/tt_pool_solved.txt")
ap.add_argument("--ood-repos", default="tornado,scrapy"); ap.add_argument("--idval-n", type=int, default=150); ap.add_argument("--heldout-extra-zero", type=int, default=60)
ap.add_argument("--seed", type=int, default=20260906); ap.add_argument("--test-sh", default=C + "/tt_test_hardened.sh"); ap.add_argument("--build", action="store_true")
a = ap.parse_args()
rng = random.Random(a.seed)
succ = {}
for p in sorted(glob.glob(a.tables)):
    for r in csv.DictReader(open(p)): succ[r["task"]] = int(r["succ"])
meta = {}
def repo(t):
    if t not in meta: meta[t] = json.load(open(f"{a.raw}/{t}/environment/workspace/metadata.json"))["repo_name"]
    return meta[t]
band = sorted(os.listdir(a.band)); ood = set(a.ood_repos.split(","))
oodval = [t for t in band if repo(t) in ood]
rest = [t for t in band if repo(t) not in ood]


def bucket(s): return "lo" if s <= 2 else ("mid" if s <= 5 else "hi")


def strat_sample(tasks, k, key):
    """proportional allocation over key(task) groups, largest remainder, min 1 per group; random within group"""
    by = collections.defaultdict(list)
    for t in sorted(tasks): by[key(t)].append(t)
    quota = {g: len(v) * k / len(tasks) for g, v in by.items()}
    alloc = {g: max(1, int(q)) for g, q in quota.items()}
    while sum(alloc.values()) > k: g = max(alloc, key=lambda g: alloc[g]); alloc[g] -= 1
    for g in sorted(by, key=lambda g: quota[g] - alloc[g], reverse=True):
        if sum(alloc.values()) >= k: break
        if alloc[g] < len(by[g]): alloc[g] += 1
    out = []
    for g, v in by.items(): out += rng.sample(v, min(alloc[g], len(v)))
    return sorted(out)


idval = strat_sample(rest, a.idval_n, lambda t: (repo(t), bucket(succ[t])))
train = sorted(set(rest) - set(idval))
# held-out: today's 100 zero tasks + extra zeros with the band's repo mix + all always-solved
sample_zero = []
with open(a.sample) as f:
    h = f.readline().rstrip("\n").split("\t")
    for l in f:
        r = dict(zip(h, l.rstrip("\n").split("\t")))
        if r["stratum"] == "zero": sample_zero.append(r["task"])
zero_pool = [l.strip() for l in open(a.zero_pool) if l.strip()]; solved_pool = [l.strip() for l in open(a.solved_pool) if l.strip()]
band_mix = collections.Counter(repo(t) for t in band)
extra_pool = [t for t in zero_pool if t not in set(sample_zero)]
# allocate the extra zeros by the BAND's repo mix (not the zero pool's, which is sympy-heavier)
by = collections.defaultdict(list)
for t in sorted(extra_pool): by[repo(t)].append(t)
quota = {r: band_mix[r] * a.heldout_extra_zero / len(band) for r in band_mix}
alloc = {r: max(1, int(q)) for r, q in quota.items()}
while sum(alloc.values()) > a.heldout_extra_zero: r = max(alloc, key=lambda r: alloc[r]); alloc[r] -= 1
for r in sorted(quota, key=lambda r: quota[r] - alloc[r], reverse=True):
    if sum(alloc.values()) >= a.heldout_extra_zero: break
    alloc[r] += 1
extra = sorted(t for r, v in by.items() for t in rng.sample(v, min(alloc.get(r, 0), len(v))))
heldout = sorted(set(sample_zero) | set(extra) | set(solved_pool))
sets = {"train": train, "idval": idval, "oodval": oodval, "heldout": heldout}
allt = [t for v in sets.values() for t in v]
assert len(allt) == len(set(allt)), "sets overlap"
assert not (set(heldout) & set(band)), "held-out overlaps the band"
for n, v in sets.items():
    mix = collections.Counter(repo(t) for t in v); sb = collections.Counter(bucket(succ.get(t, 0)) for t in v) if n != "heldout" else collections.Counter(("zero" if succ.get(t, 0) == 0 else "solved") for t in v)
    print("%-8s %4d  repos %s  succ %s" % (n, len(v), dict(mix.most_common()), dict(sb)))
    open(f"{E}/tt_v2_{n}.txt", "w").write("\n".join(v) + "\n")
with open(f"{E}/tt_v2_split.tsv", "w") as f:
    f.write("task\tset\trepo\ttt60k_succ\n")
    for n, v in sets.items():
        for t in v: f.write(f"{t}\t{n}\t{repo(t)}\t{succ.get(t, '')}\n")
print("lists -> %s/tt_v2_*.txt, tt_v2_split.tsv" % E)
if a.build:
    for n in sets:
        cmd = [sys.executable, C + "/tt_prompt.py", "rewrite", "--src", a.raw, "--dst", f"{T}/r2egym-tt-v2-{n}", "--allow", f"{E}/tt_v2_{n}.txt", "--test-sh", a.test_sh, "--force"]
        print(subprocess.check_output(cmd, text=True).strip())
