#!/usr/bin/env python3
"""gepa_split.py — the GEPA train / dev / test split over the Daytona-shape R2E-Gym tree (seed 20260921).

The prompt-evolution loop needs three disjoint task sets over the SAME tree the probes run on
(/e/fscratch/reformo/lee27/tasks/r2egym-tt-daytona, 2,476 task dirs):
  dev   500  open to the reflection LLM: it may read any dev trace, and every candidate is scored here
  test  500  touched exactly once, by the final paired run of the winning block vs the control
  train rest the pool left for RL / future data work; the loop never evaluates on it

Facts this split rests on (verified 2026-09-20 on Jupiter):
  - the daytona tree has NO environment/workspace/metadata.json (that is the r2egym-tt-raw shape). The repo name is
    `repo_name` in <task>/tests/test_info.json, present and parseable in all 2,476 dirs.
  - base pass@8 comes from two probe families: ttd60k_s{0,1,2} (Daytona backend, 2,013 tasks) and tt60k_s{0..5}
    (apptainer bridge, 2,467 of the tree). ttd60k is the same backend the GEPA probes use, so it WINS on overlap and
    tt60k is the fallback; `succ_src` records which. Exactly one tree task has neither and lands in bucket "na".
  - the v2 split lists only partially survive into the daytona tree, and every missing task is sympy:
    idval 127/150, oodval 113/115, heldout 145/176. Those survivors are forced (idval -> dev, oodval+heldout -> test).

Stratification: cell = (succ bucket, repo), buckets 0 / 1-3 / 4-5 / 6-8 / na. The forced sets are skewed (heldout is
never-solved + always-solved by construction, oodval is tornado+scrapy only), so the free picks are allocated to
CORRECT that skew: each cell's target is the tree's proportion of the 500, the forced members are subtracted, and the
remainder is drawn by largest remainder clipped to availability. Where a forced set already overshoots a cell the cell
gets no free picks and the overshoot is absorbed elsewhere -- report that residual, do not pretend it is gone.

Writes experiments/gepa/split_{train,dev,test,dev_mini}.txt, split.tsv and split_strata.md.
Python 3.9 / stdlib (Jupiter login node). Read-only on every input; writes only under --out.
"""
import argparse, collections, csv, glob, json, os, random, sys

E = "/e/fscratch/reformo/lee27/experiments"
T = "/e/fscratch/reformo/lee27/tasks"
ap = argparse.ArgumentParser()
ap.add_argument("--tree", default=T + "/r2egym-tt-daytona", help="the Daytona-shape task tree (dir per task)")
ap.add_argument("--tables", default=E + "/ttd60k_s*/pass8_pass8_table.csv", help="primary base pass@8 tables (same backend)")
ap.add_argument("--tables2", default=E + "/tt60k_s*/pass8_pass8_table.csv", help="fallback base pass@8 tables (apptainer)")
ap.add_argument("--force-dev", default=E + "/tt_v2_idval.txt", help="comma-separated lists whose tasks must land in dev")
ap.add_argument("--force-test", default=E + "/tt_v2_oodval.txt," + E + "/tt_v2_heldout.txt", help="comma-separated lists whose tasks must land in test")
ap.add_argument("--dev", type=int, default=500)
ap.add_argument("--test", type=int, default=500)
ap.add_argument("--dev-mini", type=int, default=32, help="cheap-gate subset of dev, stratified the same way")
ap.add_argument("--seed", type=int, default=20260921)
ap.add_argument("--out", default=E + "/gepa")
ap.add_argument("--dry-run", action="store_true", help="print the strata and write nothing")
a = ap.parse_args()
rng = random.Random(a.seed)

# ---------------------------------------------------------------- inputs
tasks = sorted(d for d in os.listdir(a.tree) if os.path.isdir(os.path.join(a.tree, d)))
if not tasks:
    sys.exit("no task dirs under %s" % a.tree)

repo = {}
for t in tasks:
    p = os.path.join(a.tree, t, "tests", "test_info.json")
    try:
        repo[t] = json.load(open(p))["repo_name"]
    except Exception as e:
        sys.exit("no repo_name for %s (%s): %s" % (t, p, e))


def read_tables(pattern):
    """task -> (succ, scored) from pass8_table.py CSVs; later shards win ties (they are disjoint in practice)."""
    out = {}
    for f in sorted(glob.glob(pattern)):
        for r in csv.DictReader(open(f)):
            try:
                out[r["task"]] = (int(r["succ"]), int(r["scored"]))
            except (KeyError, ValueError):
                continue
    return out


prim, fall = read_tables(a.tables), read_tables(a.tables2)
succ, src = {}, {}
for t in tasks:
    if t in prim:
        succ[t], src[t] = prim[t][0], "ttd60k"
    elif t in fall:
        succ[t], src[t] = fall[t][0], "tt60k"
    else:
        succ[t], src[t] = None, "none"


def bucket(t):
    s = succ[t]
    if s is None:
        return "na"
    return "0" if s == 0 else ("1-3" if s <= 3 else ("4-5" if s <= 5 else "6-8"))


BUCKETS = ["0", "1-3", "4-5", "6-8", "na"]


def cell(t):
    return (bucket(t), repo[t])


def read_lists(spec):
    out = set()
    for p in [x for x in spec.split(",") if x.strip()]:
        if not os.path.exists(p):
            sys.exit("missing list %s" % p)
        out |= {l.strip() for l in open(p) if l.strip() and not l.startswith("#")}
    return out


in_tree = set(tasks)
force_dev = sorted(read_lists(a.force_dev) & in_tree)
force_test = sorted(read_lists(a.force_test) & in_tree)
overlap = set(force_dev) & set(force_test)
if overlap:
    sys.exit("forced dev and forced test overlap on %d tasks, e.g. %s" % (len(overlap), sorted(overlap)[:3]))
if len(force_dev) > a.dev:
    sys.exit("forced dev %d exceeds --dev %d" % (len(force_dev), a.dev))
if len(force_test) > a.test:
    sys.exit("forced test %d exceeds --test %d" % (len(force_test), a.test))

# ---------------------------------------------------------------- allocation
tree_cells = collections.Counter(cell(t) for t in tasks)


def strat_alloc(pool, want, k):
    """Pick k tasks from pool (cell -> [tasks]) so the per-cell counts approach `want` (cell -> float desire).

    Largest remainder, clipped to availability; `want` need not sum to k and may be 0 for a cell the forced set
    already overshot. Deterministic given the module rng."""
    cells = sorted(pool)
    avail = sum(len(pool[c]) for c in cells)
    if k <= 0 or avail == 0:
        return []
    if k > avail:
        sys.exit("asked for %d tasks but only %d are left unassigned -- check --dev / --test" % (k, avail))
    tot = sum(max(0.0, want.get(c, 0.0)) for c in cells)
    if tot <= 0:  # no signal left (every cell overshot): fall back to the pool's own shape
        w = {c: len(pool[c]) * k / float(avail) for c in cells}
    else:
        w = {c: max(0.0, want.get(c, 0.0)) * k / tot for c in cells}
    alloc = {c: min(len(pool[c]), int(w[c])) for c in cells}
    while sum(alloc.values()) < k:
        free = [c for c in cells if alloc[c] < len(pool[c])]
        if not free:
            break
        c = max(free, key=lambda c: (w[c] - alloc[c], c))
        alloc[c] += 1
    while sum(alloc.values()) > k:
        c = max([c for c in cells if alloc[c] > 0], key=lambda c: (alloc[c] - w[c], c))
        alloc[c] -= 1
    out = []
    for c in cells:
        out += rng.sample(sorted(pool[c]), alloc[c])
    return sorted(out)


def pick(free_tasks, forced, size):
    """The free half of one split: target each cell at the tree's share of `size`, minus what the forced set already put there."""
    pool = collections.defaultdict(list)
    for t in free_tasks:
        pool[cell(t)].append(t)
    have = collections.Counter(cell(t) for t in forced)
    want = {}
    for c, n in tree_cells.items():
        want[c] = max(0.0, n * size / float(len(tasks)) - have[c])
    return strat_alloc(pool, want, size - len(forced))


free = [t for t in tasks if t not in set(force_dev) | set(force_test)]
dev = sorted(set(force_dev) | set(pick(free, force_dev, a.dev)))
free = [t for t in free if t not in set(dev)]
test = sorted(set(force_test) | set(pick(free, force_test, a.test)))
train = sorted(set(tasks) - set(dev) - set(test))

dev_pool = collections.defaultdict(list)
for t in dev:
    dev_pool[cell(t)].append(t)
dev_mini = strat_alloc(dev_pool, {c: float(len(v)) for c, v in dev_pool.items()}, a.dev_mini)

sets = [("train", train), ("dev", dev), ("test", test)]
allt = [t for _, v in sets for t in v]
assert len(allt) == len(set(allt)), "splits overlap"
assert set(allt) == set(tasks), "splits do not cover the tree (%d vs %d)" % (len(set(allt)), len(tasks))
assert set(force_dev) <= set(dev) and set(force_test) <= set(test), "a forced task did not land in its split"
assert set(dev_mini) <= set(dev), "dev_mini escaped dev"
if len(dev) != a.dev or len(test) != a.test:
    print("WARNING: dev %d / test %d (asked %d / %d) -- a cell ran out of tasks" % (len(dev), len(test), a.dev, a.test))

# ---------------------------------------------------------------- report
lines = []


def emit(s=""):
    lines.append(s)
    print(s)


emit("# GEPA split (seed %d, tree %s)\n" % (a.seed, a.tree))
emit("| set | tasks | forced | %s |" % " | ".join(BUCKETS))
emit("|---|---|---|%s" % ("---|" * len(BUCKETS)))
forced_of = {"train": set(), "dev": set(force_dev), "test": set(force_test)}
for n, v in sets + [("dev_mini", dev_mini)]:
    b = collections.Counter(bucket(t) for t in v)
    emit("| %s | %d | %d | %s |" % (n, len(v), len(forced_of.get(n, set()) & set(v)), " | ".join(str(b[x]) for x in BUCKETS)))
repos = [r for r, _ in collections.Counter(repo.values()).most_common()]
emit("\n| set | %s |" % " | ".join(repos))
emit("|---|%s" % ("---|" * len(repos)))
for n, v in sets + [("dev_mini", dev_mini)] + [("tree", tasks)]:
    c = collections.Counter(repo[t] for t in v)
    emit("| %s | %s |" % (n, " | ".join(str(c[r]) for r in repos)))
emit("\n| set | ttd60k | tt60k | none |")
emit("|---|---|---|---|")
for n, v in sets:
    c = collections.Counter(src[t] for t in v)
    emit("| %s | %d | %d | %d |" % (n, c["ttd60k"], c["tt60k"], c["none"]))
emit("\nBase pass@8 mean (scored tasks only): %s" % ", ".join(
    "%s %.3f" % (n, sum(succ[t] for t in v if succ[t] is not None) / max(1, sum(1 for t in v if succ[t] is not None)) / 8.0)
    for n, v in sets))
skew = []
for c, n in sorted(tree_cells.items()):
    want_dev, want_test = n * len(dev) / float(len(tasks)), n * len(test) / float(len(tasks))
    got_dev = sum(1 for t in dev if cell(t) == c)
    got_test = sum(1 for t in test if cell(t) == c)
    if abs(got_dev - want_dev) >= 5 or abs(got_test - want_test) >= 5:
        skew.append("%s/%s: dev %d (want %.0f), test %d (want %.0f)" % (c[0], c[1], got_dev, want_dev, got_test, want_test))
emit("\nResidual skew vs the tree's shape (cells off by >= 5), caused by the forced lists:")
for s in skew or ["(none)"]:
    emit("  " + s)

if a.dry_run:
    print("\n--dry-run: wrote nothing")
    sys.exit(0)
os.makedirs(a.out, exist_ok=True)
for n, v in sets + [("dev_mini", dev_mini)]:
    open("%s/split_%s.txt" % (a.out, n), "w").write("\n".join(v) + "\n")
with open("%s/split.tsv" % a.out, "w") as f:
    f.write("task\tset\trepo\tsucc\tsucc_src\tbucket\tforced\tdev_mini\n")
    for n, v in sets:
        for t in v:
            f.write("%s\t%s\t%s\t%s\t%s\t%s\t%d\t%d\n" % (
                t, n, repo[t], "" if succ[t] is None else succ[t], src[t], bucket(t),
                int(t in forced_of[n]), int(t in set(dev_mini))))
open("%s/split_strata.md" % a.out, "w").write("\n".join(lines) + "\n")
print("\nwrote %s/split_{train,dev,test,dev_mini}.txt, split.tsv, split_strata.md" % a.out)
