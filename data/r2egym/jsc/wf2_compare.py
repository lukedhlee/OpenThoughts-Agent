#!/usr/bin/env python3
"""wf2_compare.py — the fixed-clauses re-probe (ttwf2, 112 tasks) against the header (tthd) and TaskTrove-verbatim (ttwf) arms.
Three-way paired on the 100 band tasks; per-task table for the 18 disagreement tasks (incl. the four trap tasks); and the trap
behaviours per arm from wf_feat.py features: restores/reverts (git checkout/stash, cp *.backup), visible-test runs, repro rerun after
last edit, done-after-own-script, done-on-green-visible-suite proxies. Python 3.9 / stdlib.
"""
import csv, glob, json, os, random, statistics, collections
E = "/e/fscratch/reformo/lee27/experiments"
ARMS = [("hd", "tthd"), ("wf", "ttwf"), ("wf2", "ttwf2")]
TRAPS = {"r2egym-v1-06865": "reverted correct fix on XFAIL noise", "r2egym-v1-05754": "visible test asserts pre-fix, gave up",
         "r2egym-v1-07150": "deadlock on contradicting visible test", "r2egym-v1-06352": "done on non-covering green suite",
         "r2egym-v1-00159": "weak repro assertion (rounding vs truncation)", "r2egym-v1-04611": "repro count assertion locked in wrong fix",
         "r2egym-v1-05055": "gain: early runnable repro", "r2egym-v1-04900": "gain: probing found root cause"}


def load(prefix):
    t = {}
    for p in sorted(glob.glob("%s/%s_s[0-9]/pass8_pass8_table.csv" % (E, prefix))):
        for r in csv.DictReader(open(p)):
            t[r["task"]] = dict(succ=int(r["succ"]), scored=int(r["scored"]), ctx=int(r["ctx_exceeded"]), att=int(r["attempts"]))
    return t


def boot(pairs, n=2000, seed=0):
    rng = random.Random(seed); ds = [b - a for a, b in pairs]
    if not ds: return (None,) * 3
    bs = sorted(sum(ds[rng.randrange(len(ds))] for _ in ds) / len(ds) for _ in range(n))
    return sum(ds) / len(ds), bs[int(.025 * n)], bs[int(.975 * n)]


def f3(x): return "-" if x is None else "%+.3f" % x


rows = {}
with open(E + "/ttwf_sample.tsv") as f:
    h = f.readline().rstrip("\n").split("\t")
    for l in f:
        r = dict(zip(h, l.rstrip("\n").split("\t"))); rows[r["task"]] = r
tab = {a: load(p) for a, p in ARMS}
in2 = sorted(tab["wf2"])
band = [t for t in in2 if rows[t]["stratum"] == "medium" and all(t in tab[a] and tab[a][t]["scored"] for a in tab)]
pr = {a: {t: tab[a][t]["succ"] / tab[a][t]["scored"] for t in band} for a in tab}
print("# Fixed-clauses prompt (wf2) vs header (hd) vs TaskTrove verbatim (wf), %d band tasks paired, k = 8\n" % len(band))
print("| arm | per-task pass | trial pass | ctx deaths |"); print("|---|---|---|---|")
for a, _ in ARMS:
    s = sum(tab[a][t]["succ"] for t in band); n = sum(tab[a][t]["scored"] for t in band); c = sum(tab[a][t]["ctx"] for t in band); at = sum(tab[a][t]["att"] for t in band)
    print("| %s | %.3f | %.3f | %d%% |" % (a, sum(pr[a].values()) / len(band), s / n, round(100 * c / at)))
print("\n| paired delta | mean | 95 % |"); print("|---|---|---|")
for a, b in (("hd", "wf2"), ("wf", "wf2"), ("hd", "wf")):
    m, lo, hi = boot([(pr[a][t], pr[b][t]) for t in band]); print("| %s → %s | **%s** | [%s, %s] |" % (a, b, f3(m), f3(lo), f3(hi)))
d = collections.Counter(tab["wf2"][t]["succ"] - tab["hd"][t]["succ"] for t in band); print("\nΔ successes wf2 − hd per task:", dict(sorted(d.items())))
d = collections.Counter(tab["wf2"][t]["succ"] - tab["wf"][t]["succ"] for t in band); print("Δ successes wf2 − wf per task:", dict(sorted(d.items())))
print("\n## The 18 disagreement tasks (successes of 8)\n\n| task | stratum | repo | hd | wf | wf2 | reading |"); print("|---|---|---|---|---|---|---|")
for t in [l.strip() for l in open(E + "/ttwf_ana/digest_tasks.txt") if l.strip()]:
    print("| %s | %s | %s | %s | %s | %s | %s |" % (t, rows[t]["stratum"], rows[t]["repo"], *["%d/%d" % (tab[a][t]["succ"], tab[a][t]["scored"]) if t in tab[a] else "-" for a in ("hd", "wf", "wf2")], TRAPS.get(t, "")))
# behaviour
F = {}
for a, p in ARMS:
    fp = "%s/ttwf_ana/f_%s.jsonl" % (E, p)
    if os.path.exists(fp): F[a] = [json.loads(l) for l in open(fp)]
if F:
    keep = set(in2)
    print("\n## Trap behaviours, scored trials on the same %d tasks\n" % len(keep))
    feats = [("tc", "declared done"), ("tc_after_pytest", "done after repo tests"), ("tc_after_selftest", "done after own script only"),
             ("repro_written", "wrote reproduce_issue.py"), ("repro_after_edit", "ran repro after last edit"), ("pytest_any", "ran visible tests"),
             ("git_restore_any", "git checkout/stash/reset (revert)"), ("ctx", "context death")]
    hdr = "| behaviour | " + " | ".join("%s WIN | %s LOSS" % (a, a) for a in F) + " |"; print(hdr); print("|---|" + "---|" * (2 * len(F)))
    def prep(r):
        r["pytest_any"] = (r.get("n_pytest") or 0) > 0; r["git_restore_any"] = (r.get("git_restore") or 0) > 0; return r
    groups = {}
    for a in F:
        rs = [prep(r) for r in F[a] if r["task"] in keep and r.get("reward") is not None]
        groups[a] = ([r for r in rs if r["reward"] >= 1], [r for r in rs if r["reward"] < 1])
    for k, lab in feats:
        cells = []
        for a in F:
            for g in groups[a]: cells.append("%d%%" % round(100 * sum(1 for r in g if r.get(k)) / len(g)) if g else "-")
        print("| %s | %s |" % (lab, " | ".join(cells)))
    print("| n | %s |" % " | ".join(str(len(g)) for a in F for g in groups[a]))
    for a in F:
        rs = [r for r in F[a] if r["task"] in keep and r.get("reward") is not None]
        done = [r for r in rs if r.get("tc")]
        print("%s: tc_rate %.3f  P(win|done) %.3f  median turns %s" % (a, len(done) / len(rs), sum(r["reward"] >= 1 for r in done) / max(1, len(done)), statistics.median([r["turns"] for r in rs])))
