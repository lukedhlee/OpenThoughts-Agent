#!/usr/bin/env python3
"""p2o_dev120.py: frozen dev subset for P2O wave 0.

Source = the 09-06 header-prompt control (tthd_s0..2: 300 tasks x 8 on the current harness + hardened verifier, base model),
strata by the CONTROL's own succ/8: zero 0, hard 1-3, medium 4-5, success 6-8. Excludes every task in the RL gate splits
(idval / oodval / heldout). Repo-proportional within stratum, seeded. Python 3.9 / stdlib (Jupiter login node).
Writes <out> with: task, repo, stratum, base_succ8 (tthd), base_ctx8 (tthd context deaths of 8), tt60k_succ, med_turns.
"""
import argparse, collections, csv, glob, os, random
E = "/e/fscratch/reformo/lee27/experiments"
ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=20260912)
ap.add_argument("--out", default=E + "/p2o/dev120.tsv")
ap.add_argument("--n", default="zero:45,hard:35,medium:20,success:20")
ap.add_argument("--zero-from-rest", default=E + "/snowball_v2rest_base_fixed/pass8_pass8_table.csv",
                help="fill the zero stratum from this base probe of the rest pool (succ 0, full) when the 09-06 sample has too few")
ap.add_argument("--rest-tsv", default=E + "/tt_v2_rest.tsv")
ap.add_argument("--ood-repos", default="tornado,scrapy")
a = ap.parse_args()
want = {k: int(v) for k, v in (x.split(":") for x in a.n.split(","))}
sample = {r["task"]: r for r in csv.DictReader(open(E + "/ttwf_sample.tsv"), delimiter="\t")}
rows = {}
for p in sorted(glob.glob(E + "/tthd_s*/pass8_pass8_table.csv")):
    for r in csv.DictReader(open(p)):
        rows[r["task"]] = r
excl = set()
for s in ("idval", "oodval", "heldout"):
    excl |= {l.strip() for l in open(E + "/tt_v2_%s.txt" % s) if l.strip()}


def stratum(succ):
    return "zero" if succ == 0 else "hard" if succ <= 3 else "medium" if succ <= 5 else "success"


elig = collections.defaultdict(list)
n_excl = 0
for t, r in rows.items():
    if t not in sample or r["full"] != "True" or int(r["scored"]) < 8:
        continue
    if t in excl:
        n_excl += 1
        continue
    elig[stratum(int(r["succ"]))].append(t)
print("control tables %d tasks; excluded by gate splits %d; eligible %s" % (len(rows), n_excl, {k: len(v) for k, v in elig.items()}))
rng = random.Random(a.seed)


def repo_prop(ts, k):
    by = collections.defaultdict(list)
    for t in sorted(ts):
        by[sample[t]["repo"]].append(t)
    for v in by.values():
        rng.shuffle(v)
    tot = len(ts)
    alloc = {r: len(v) * k / tot for r, v in by.items()}
    base = {r: int(x) for r, x in alloc.items()}
    rem = k - sum(base.values())
    for r, _ in sorted(alloc.items(), key=lambda kv: -(kv[1] - int(kv[1])))[:rem]:
        base[r] += 1
    out = []
    for r, v in by.items():
        out += v[:base[r]]
    return sorted(out)


chosen = []
src_of = {}
for s, k in want.items():
    pool = elig[s]
    k2 = min(k, len(pool))
    print("%-8s eligible %3d take %3d" % (s, len(pool), k2))
    chosen += [(t, s) for t in repo_prop(pool, k2)]
    for t in pool:
        src_of[t] = "tthd"
n_zero = sum(1 for _, s in chosen if s == "zero")
if n_zero < want.get("zero", 0) and a.zero_from_rest and os.path.exists(a.zero_from_rest):
    need = want["zero"] - n_zero
    ood = set(a.ood_repos.split(","))
    rest_repo = {r["task"]: r["repo"] for r in csv.DictReader(open(a.rest_tsv), delimiter="\t")}
    zrows = {r["task"]: r for r in csv.DictReader(open(a.zero_from_rest))
             if r["full"] == "True" and int(r["scored"]) >= 8 and int(r["succ"]) == 0 and r["task"] not in excl
             and r["task"] not in rows and rest_repo.get(r["task"]) not in ood and r["task"] in rest_repo}
    # repo-proportional to the sample's hard+medium mix (the band's mix), not to the rest pool's own sympy-heavy mix
    target = collections.Counter(sample[t]["repo"] for t, s in chosen if s in ("hard", "medium"))
    for t in zrows:
        sample.setdefault(t, {"repo": rest_repo[t], "tt60k_succ": "0"})
    by = collections.defaultdict(list)
    for t in sorted(zrows):
        by[rest_repo[t]].append(t)
    for v in by.values():
        rng.shuffle(v)
    tot = sum(target[r] for r in by if r in target) or 1
    alloc = {r: target[r] * need / tot for r in by if r in target}
    base = {r: int(x) for r, x in alloc.items()}; rem = need - sum(base.values())
    for r, _ in sorted(alloc.items(), key=lambda kv: -(kv[1] - int(kv[1])))[:rem]:
        base[r] += 1
    extra = []
    for r, k in base.items():
        extra += by[r][:k]
    print("zero     from rest pool: eligible %d, take %d (band repo mix)" % (len(zrows), len(extra)))
    for t in extra:
        z = zrows[t]; rows[t] = z; src_of[t] = "rest"
    chosen += [(t, "zero") for t in sorted(extra)]
os.makedirs(os.path.dirname(a.out), exist_ok=True)
with open(a.out, "w") as f:
    f.write("task\trepo\tstratum\tbase_succ8\tbase_ctx8\ttt60k_succ\tmed_turns\tsource\n")
    for t, s in chosen:
        r = rows[t]
        f.write("\t".join([t, sample[t]["repo"], s, r["succ"], r["ctx_exceeded"], sample[t].get("tt60k_succ", "0"), r["med_turns"], src_of.get(t, "tthd")]) + "\n")
print("wrote %s: %d tasks" % (a.out, len(chosen)))
print("strata:", dict(collections.Counter(s for _, s in chosen)))
print("repos:", dict(collections.Counter(sample[t]["repo"] for t, _ in chosen)))
