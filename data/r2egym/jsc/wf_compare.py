#!/usr/bin/env python3
"""wf_compare.py [--sample ttwf_sample.tsv] [--feat-dir ttwf_ana] > report.md
Paired comparison of the workflow-prompt arm (ttwf) against the header-prompt control (tthd) on the same 300 tasks,
same harness, same hardened verifier, same base model. Per stratum and per repo: per-task pass rate, paired delta with a
bootstrap 95 % interval, trial pass, context deaths, tasks newly solved / lost. Then the behaviour tables from
wf_feat.py: arm x outcome and arm x stratum, including the two metrics the answer-contract page asked for,
tc_rate and P(reward = 1 | task_complete). Python 3.9 / stdlib.
"""
import argparse, collections, csv, glob, json, os, random, statistics
E = "/e/fscratch/reformo/lee27/experiments"
ARMS = [("hd", "tthd"), ("wf", "ttwf")]


def med(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def rate(recs, key):
    xs = [bool(r.get(key)) for r in recs]
    return sum(xs) / len(xs) if xs else None


def f3(x): return "-" if x is None else ("%.3f" % x)
def f1(x): return "-" if x is None else ("%.1f" % x)
def pct(x): return "-" if x is None else ("%d%%" % round(100 * x))


def boot_ci(pairs, n=2000, seed=0):
    rng = random.Random(seed); ds = [b - a for a, b in pairs]
    if not ds: return None, None, None
    m = sum(ds) / len(ds); bs = []
    for _ in range(n):
        s = [ds[rng.randrange(len(ds))] for _ in ds]; bs.append(sum(s) / len(s))
    bs.sort(); return m, bs[int(0.025 * n)], bs[int(0.975 * n)]


def load_tables(prefix):
    t = {}
    for p in sorted(glob.glob("%s/%s_s*/pass8_pass8_table.csv" % (E, prefix))):
        for r in csv.DictReader(open(p)):
            t[r["task"]] = dict(scored=int(r["scored"]), succ=int(r["succ"]), nulls=int(r["nulls"]), ctx=int(r["ctx_exceeded"]),
                                med_turns=float(r["med_turns"]) if r["med_turns"] else None, attempts=int(r["attempts"]))
    return t


def stratum_block(name, tasks, tab, feats):
    """one row of the outcome table for a task subset"""
    paired = [t for t in tasks if t in tab["hd"] and t in tab["wf"] and tab["hd"][t]["scored"] and tab["wf"][t]["scored"]]
    pr = {a: [tab[a][t]["succ"] / tab[a][t]["scored"] for t in paired] for a in ("hd", "wf")}
    m, lo, hi = boot_ci(list(zip(pr["hd"], pr["wf"])))
    trial = {a: (sum(tab[a][t]["succ"] for t in paired), sum(tab[a][t]["scored"] for t in paired)) for a in ("hd", "wf")}
    ctx = {a: (sum(tab[a][t]["ctx"] for t in paired), sum(tab[a][t]["attempts"] for t in paired)) for a in ("hd", "wf")}
    new = sum(1 for t in paired if tab["hd"][t]["succ"] == 0 and tab["wf"][t]["succ"] > 0)
    lost = sum(1 for t in paired if tab["hd"][t]["succ"] > 0 and tab["wf"][t]["succ"] == 0)
    both = sum(1 for t in paired if tab["hd"][t]["succ"] > 0 and tab["wf"][t]["succ"] > 0)
    turns = {a: med([tab[a][t]["med_turns"] for t in paired]) for a in ("hd", "wf")}
    return "| %s | %d | %s | %s | **%s** [%s, %s] | %s / %s | %s / %s | %d / %d / %d | %s / %s |" % (
        name, len(paired), f3(sum(pr["hd"]) / len(pr["hd"]) if pr["hd"] else None), f3(sum(pr["wf"]) / len(pr["wf"]) if pr["wf"] else None),
        f3(m), f3(lo), f3(hi), f3(trial["hd"][0] / trial["hd"][1] if trial["hd"][1] else None), f3(trial["wf"][0] / trial["wf"][1] if trial["wf"][1] else None),
        pct(ctx["hd"][0] / ctx["hd"][1] if ctx["hd"][1] else None), pct(ctx["wf"][0] / ctx["wf"][1] if ctx["wf"][1] else None), new, lost, both, f1(turns["hd"]), f1(turns["wf"]))


BEHAV = [("tc", "declared done"), ("tc_after_pytest", "done after repo tests"), ("tc_after_selftest", "done after own script only"),
         ("repro_written", "wrote reproduce_issue.py"), ("repro_before_edit", "ran repro before 1st src edit"), ("repro_after_edit", "ran repro after last src edit"),
         ("pytest_any", "ran repo tests"), ("pytest_after_edit", "repo tests after last src edit"), ("grader_any", "ran the grader"),
         ("ctx", "context death"), ("never_src_edit", "never edited source"), ("shadow_any", "wrote stdlib-shadowing file"), ("git_clean_any", "git clean")]
MEDS = [("turns", "turns"), ("comp_tokens", "completion tokens"), ("first_src_edit", "first src edit turn"), ("n_src_edit", "src edit turns"), ("rshare", "reasoning share")]


def prep(r):
    r["pytest_any"] = (r.get("n_pytest") or 0) > 0; r["grader_any"] = (r.get("grader_runs") or 0) > 0
    r["never_src_edit"] = (r.get("n_src_edit") or 0) == 0; r["shadow_any"] = bool(r.get("shadow_files")); r["git_clean_any"] = (r.get("git_clean") or 0) > 0
    return r


def behav_table(groups):
    """groups: list of (label, recs)"""
    out = ["| behaviour | " + " | ".join(l for l, _ in groups) + " |", "|---|" + "---|" * len(groups)]
    for k, lab in BEHAV: out.append("| %s | " % lab + " | ".join(pct(rate(recs, k)) for _, recs in groups) + " |")
    for k, lab in MEDS: out.append("| median %s | " % lab + " | ".join(("%.2f" % med([r.get(k) for r in recs])) if k == "rshare" and med([r.get(k) for r in recs]) is not None else f1(med([r.get(k) for r in recs])) for _, recs in groups) + " |")
    out.append("| n trials | " + " | ".join(str(len(recs)) for _, recs in groups) + " |")
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--sample", default=E + "/ttwf_sample.tsv"); ap.add_argument("--feat-dir", default=E + "/ttwf_ana")
    a = ap.parse_args()
    rows = {}
    with open(a.sample) as f:
        hdr = f.readline().rstrip("\n").split("\t")
        for line in f:
            r = dict(zip(hdr, line.rstrip("\n").split("\t"))); rows[r["task"]] = r
    tab = {short: load_tables(prefix) for short, prefix in ARMS}
    strata = ["zero", "medium", "success"]
    print("# Workflow prompt vs header prompt, paired on %d tasks (base Snowball, fixed harness, hardened verifier)\n" % len(rows))
    print("Per-task pass = successes / scored attempts (k = 8). Delta = mean over paired tasks of (workflow - header), bootstrap 95 %% interval over tasks. "
          "Base-probe reference (frozen harness, header prompt, 09-04) per stratum: zero = 0/8 by construction, medium = 2-5/8, success = 6-8/8.\n")
    hdr = "| subset | tasks | header | workflow | delta | trial pass hd / wf | ctx deaths hd / wf | newly solved / lost / both | med turns hd / wf |"
    print(hdr); print("|---|---|---|---|---|---|---|---|---|")
    for s in strata: print(stratum_block(s, [t for t in rows if rows[t]["stratum"] == s], tab, None))
    print(stratum_block("all", list(rows), tab, None))
    print("\n## By repo (all strata pooled)\n"); print(hdr); print("|---|---|---|---|---|---|---|---|---|")
    for repo in sorted({r["repo"] for r in rows.values()}):
        print(stratum_block(repo, [t for t in rows if rows[t]["repo"] == repo], tab, None))
    print("\n## By repo within stratum (counts of tasks with >= 1 success, header -> workflow)\n")
    print("| repo | " + " | ".join(strata) + " |"); print("|---|" + "---|" * len(strata))
    for repo in sorted({r["repo"] for r in rows.values()}):
        cells = []
        for s in strata:
            ts = [t for t in rows if rows[t]["repo"] == repo and rows[t]["stratum"] == s and t in tab["hd"] and t in tab["wf"]]
            cells.append("%d -> %d of %d" % (sum(1 for t in ts if tab["hd"][t]["succ"] > 0), sum(1 for t in ts if tab["wf"][t]["succ"] > 0), len(ts)) if ts else "-")
        print("| %s | %s |" % (repo, " | ".join(cells)))
    feats = {}
    for short, prefix in ARMS:
        p = "%s/f_%s.jsonl" % (a.feat_dir, prefix)
        if os.path.exists(p): feats[short] = [prep(json.loads(l)) for l in open(p)]
    if not feats: return
    for short in feats:
        for r in feats[short]: r["stratum"] = rows.get(r["task"], {}).get("stratum")
    scored = {s: [r for r in feats[s] if r.get("reward") is not None] for s in feats}
    print("\n## Behaviour by arm and outcome (scored trials; last attempt of each trial)\n")
    groups = []
    for s in ("hd", "wf"):
        for lab, cond in (("WIN", lambda r: r["reward"] >= 1), ("LOSS", lambda r: r["reward"] < 1)):
            groups.append(("%s %s" % (s, lab), [r for r in scored.get(s, []) if cond(r)]))
    print("\n".join(behav_table(groups)))
    print("\n## Behaviour by arm and stratum (all scored trials)\n")
    groups = [("%s %s" % (s, st), [r for r in scored.get(s, []) if r["stratum"] == st]) for st in strata for s in ("hd", "wf")]
    print("\n".join(behav_table(groups)))
    print("\n## The two contract metrics: tc_rate and P(reward = 1 | task_complete)\n")
    print("| subset | tc_rate hd | tc_rate wf | P(win \\| done) hd | P(win \\| done) wf | P(win \\| not done) hd | P(win \\| not done) wf |"); print("|---|---|---|---|---|---|---|")
    for st in strata + ["all"]:
        cells = []
        for key in ("tc_rate", "pwd", "pwnd"):
            for s in ("hd", "wf"):
                recs = [r for r in scored.get(s, []) if st == "all" or r["stratum"] == st]
                done = [r for r in recs if r.get("tc")]; notdone = [r for r in recs if not r.get("tc")]
                if key == "tc_rate": cells.append(f3(len(done) / len(recs) if recs else None))
                elif key == "pwd": cells.append(f3(sum(r["reward"] >= 1 for r in done) / len(done) if done else None))
                else: cells.append(f3(sum(r["reward"] >= 1 for r in notdone) / len(notdone) if notdone else None))
        print("| %s | %s |" % (st, " | ".join(cells)))
    shadow = collections.Counter(f for s in feats for r in feats[s] for f in (r.get("shadow_files") or []))
    if shadow: print("\nstdlib-shadowing files written (both arms):", dict(shadow.most_common(10)))
    ex = collections.Counter((s, r.get("exc")) for s in feats for r in feats[s] if r.get("exc"))
    print("\nexceptions by arm:", {"%s:%s" % k: v for k, v in ex.most_common(12)})


if __name__ == "__main__":
    main()
