#!/usr/bin/env python3
"""p2o_compare.py [--prefixes p2o6a,p2o6b,p2o6c] [--dev dev120.tsv] > report.md
Paired comparison of the five guidance-block arms (A..E) against the control (ctl) on the same dev tasks, same shard,
same harness, same base model. Task dirs are <task>-p<arm>; the pass@k tables come from pass8_table.py on each shard
(run with --k 4 for wave 0). Per stratum and per arm: per-task pass rate, paired delta with a bootstrap 95 % interval,
tasks newly solved / lost, context-death rate, median turns. Python 3.9 / stdlib.
"""
import argparse, collections, csv, glob, random, statistics
E = "/e/fscratch/reformo/lee27/experiments"
ap = argparse.ArgumentParser()
ap.add_argument("--prefixes", default="p2o6a,p2o6b,p2o6c")
ap.add_argument("--dev", default=E + "/p2o/dev120.tsv")
ap.add_argument("--table", default="pass8_pass8_table.csv")
ap.add_argument("--boot", type=int, default=2000); ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
dev = {r["task"]: r for r in csv.DictReader(open(a.dev), delimiter="\t")}
rows = {}
for p in a.prefixes.split(","):
    for f in glob.glob("%s/%s_s*/%s" % (E, p, a.table)):
        for r in csv.DictReader(open(f)):
            name = r["task"]; task, arm = name.rsplit("-p", 1)
            rows[(task, arm)] = r
arms = sorted({k[1] for k in rows} - {"ctl"})
strata = ["zero", "hard", "medium", "success", "all"]
rng = random.Random(a.seed)


def scored(r):
    return int(r["scored"]) > 0


def rate(r):
    return int(r["succ"]) / int(r["scored"])


def ci(deltas):
    n = len(deltas)
    if n < 2:
        return (0.0, 0.0)
    bs = sorted(sum(rng.choice(deltas) for _ in range(n)) / n for _ in range(a.boot))
    return bs[int(.025 * a.boot)], bs[int(.975 * a.boot)]


print("# P2O wave 0: paired per-task pass (successes / scored), arm - ctl, bootstrap 95 % over tasks\n")
print("| arm | stratum | tasks | ctl | arm | delta | 95 % CI | new / lost | ctx ctl / arm | med turns ctl / arm |")
print("|---|---|---|---|---|---|---|---|---|---|")
for arm in arms:
    for s in strata:
        pairs = []
        for t, d in dev.items():
            if s != "all" and d["stratum"] != s:
                continue
            c = rows.get((t, "ctl")); v = rows.get((t, arm))
            if not c or not v or not scored(c) or not scored(v):
                continue
            pairs.append((t, c, v))
        if not pairs:
            continue
        pc = [rate(c) for _, c, _ in pairs]; pv = [rate(v) for _, _, v in pairs]
        deltas = [y - x for x, y in zip(pc, pv)]
        lo, hi = ci(deltas)
        new = sum(1 for x, y in zip(pc, pv) if x == 0 and y > 0); lost = sum(1 for x, y in zip(pc, pv) if x > 0 and y == 0)
        ctx = lambda rs: sum(int(r["ctx_exceeded"]) for r in rs) / max(1, sum(int(r["attempts"]) for r in rs))
        mt = lambda rs: statistics.median([float(r["med_turns"]) for r in rs if r["med_turns"] not in ("", "None")] or [0])
        print("| %s | %s | %d | %.3f | %.3f | %+.3f | [%+.3f, %+.3f] | %d / %d | %.2f / %.2f | %.1f / %.1f |" % (
            arm, s, len(pairs), sum(pc) / len(pc), sum(pv) / len(pv), sum(deltas) / len(deltas), lo, hi, new, lost,
            ctx([c for _, c, _ in pairs]), ctx([v for _, _, v in pairs]), mt([c for _, c, _ in pairs]), mt([v for _, _, v in pairs])))
print("\nNext readers: wf_feat.py per shard (behaviour: repro / tests / declared done / ctx death / never edited), think-share by turn from"
      " the trajectories, prescreen summary.tsv for the shift metric; pick by lift per unit of shift.")
