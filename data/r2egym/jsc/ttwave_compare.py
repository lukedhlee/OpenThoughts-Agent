#!/usr/bin/env python3
"""ttwave_compare.py — join the TaskTrove-side pass@8 tables with the manifest and the raw 28k/60k priors (python3.9-safe, stdlib).
Usage: ttwave_compare.py [--wave ttwave --shards 4] [--sample ttsample_s0]   (tables: <exp>/<name>/pass8_pass8_table.csv, else a fresh
pass8_table.py run on the trace tree is attempted). Prints markdown."""
import argparse, csv, glob, os, subprocess, sys
X = "/e/fscratch/reformo/lee27/experiments"; E = X + "/ttsample"; PT = "/e/project1/transfernetx/lee27/code/snowball/pass8_table.py"
ap = argparse.ArgumentParser(); ap.add_argument("--wave", default="ttwave"); ap.add_argument("--shards", type=int, default=4)
ap.add_argument("--sample", default="ttsample_s0"); ap.add_argument("--no-table-run", action="store_true"); a = ap.parse_args()
def table(name):
    d = "%s/%s" % (X, name); f = d + "/pass8_pass8_table.csv"
    if not os.path.exists(f) and not a.no_table_run and os.path.isdir("%s/%s/trace_jobs" % (d, name)):
        subprocess.run([sys.executable, PT, "%s/%s/trace_jobs" % (d, name), "--k", "8", "--out", d + "/pass8_live"], capture_output=True); f = d + "/pass8_live_pass8_table.csv"
    if not os.path.exists(f): return {}
    return {r["task"]: r for r in csv.DictReader(open(f))}
def pk(r): return float(r["pass_at_k"]) if r and r.get("pass_at_k") not in (None, "") else None
def bucket(v): return "zero" if v == 0 else ("full" if v == 1 else "mixed")
def mean(xs): xs = [x for x in xs if x is not None]; return (sum(xs) / len(xs)) if xs else float("nan")
def med(xs):
    xs = sorted(float(x) for x in xs if x not in (None, "")); return xs[len(xs) // 2] if xs else float("nan")
# ---- wave
man = list(csv.DictReader(open(E + "/ttwave_manifest.tsv"), delimiter="\t"))
res = {}
for i in range(a.shards): res.update(table("%s_s%d" % (a.wave, i)))
print("# TaskTrove-side pass@8 wave `%s` (%d shards) — %d of %d tasks tabled, %d fully sampled\n" % (a.wave, a.shards, sum(t in res for t in (m["task"] for m in man)), len(man), sum(res[m["task"]].get("full") == "True" for m in man if m["task"] in res)))
for grp in ("overlap_tt", "tt_only"):
    rows = [m for m in man if m["group"] == grp and m["task"] in res and res[m["task"]].get("full") == "True"]
    tt = [pk(res[m["task"]]) for m in rows]
    print("## %s — n=%d fully sampled" % (grp, len(rows)))
    print("| metric | TaskTrove copy (this wave) | raw twin @60k | raw twin @28k |\n|---|---|---|---|")
    if grp == "overlap_tt":
        r60 = [float(m["raw_pass8_60k"]) for m in rows]; r28 = [float(m["raw_pass8_28k"]) for m in rows]
        print("| mean pass@8 | %.3f | %.3f | %.3f |" % (mean(tt), mean(r60), mean(r28)))
        for b in ("zero", "mixed", "full"):
            print("| %s tasks | %d | %d | %d |" % (b, sum(bucket(v) == b for v in tt), sum(bucket(v) == b for v in r60), sum(bucket(v) == b for v in r28)))
        print("| median turns | %s | %s | — |" % (med(res[m["task"]].get("med_turns") for m in rows), med(m["raw_med_turns_60k"] for m in rows)))
        print("| ctx-exceeded attempts | %d | — | — |" % sum(int(res[m["task"]].get("ctx_exceeded") or 0) for m in rows))
        print("\n### bucket agreement, raw@60k (rows) vs TaskTrove copy (cols)\n| raw@60k \\ TT | zero | mixed | full |\n|---|---|---|---|")
        for b in ("zero", "mixed", "full"):
            print("| %s | %s |" % (b, " | ".join(str(sum(bucket(r) == b and bucket(t) == c for r, t in zip(r60, tt))) for c in ("zero", "mixed", "full"))))
        d = [t - r for t, r in zip(tt, r60)]; print("\nTT − raw@60k per task: mean %+.3f, |Δ|≥0.25 on %d/%d tasks" % (mean(d), sum(abs(x) >= 0.25 for x in d), len(d)))
        print("\n### by repo (n, TT mean, raw@60k mean)\n| repo | n | TT | raw@60k |\n|---|---|---|---|")
        repos = sorted({m["repo"] for m in rows})
        for rp in repos:
            rr = [m for m in rows if m["repo"] == rp]; print("| %s | %d | %.3f | %.3f |" % (rp, len(rr), mean(pk(res[m["task"]]) for m in rr), mean(float(m["raw_pass8_60k"]) for m in rr)))
    else:
        print("| mean pass@8 | %.3f | — | — |" % mean(tt))
        for b in ("zero", "mixed", "full"): print("| %s tasks | %d | — | — |" % (b, sum(bucket(v) == b for v in tt)))
        print("| median turns | %s | — | — |" % med(res[m["task"]].get("med_turns") for m in rows))
        print("| ctx-exceeded attempts | %d | — | — |" % sum(int(res[m["task"]].get("ctx_exceeded") or 0) for m in rows))
        print("| nulls (infra) | %d | — | — |" % sum(int(res[m["task"]].get("nulls") or 0) for m in rows))
    print()
# ---- same-pool pair check
sm = list(csv.DictReader(open(E + "/ttsample_manifest.tsv"), delimiter="\t")); sr = table(a.sample)
if sr:
    print("## same-pool pair check `%s` — %d/%d tasks tabled\n" % (a.sample, sum(m["task"] in sr for m in sm), len(sm)))
    pairs = {}
    for m in sm:
        if m["pair"]: pairs.setdefault(m["pair"], {})[m["group"]] = m
    print("| pair | repo | TT copy | raw copy | raw prior@60k |\n|---|---|---|---|---|")
    agree = 0; n = 0
    for pid in sorted(pairs):
        pr = pairs[pid]; t = pr.get("overlap_tt"); r = pr.get("overlap_raw")
        if not (t and r and t["task"] in sr and r["task"] in sr): continue
        a1, a2 = pk(sr[t["task"]]), pk(sr[r["task"]]); n += 1; agree += bucket(a1) == bucket(a2)
        print("| %s | %s | %.3f | %.3f | %s |" % (pid, t["repo"], a1, a2, t["prior_pass8"]))
    print("\nbucket agreement TT vs raw copy: %d/%d" % (agree, n))
    for grp in ("tt_only", "raw_only"):
        rows = [m for m in sm if m["group"] == grp and m["task"] in sr]
        print("%s: n=%d mean pass@8 %.3f%s" % (grp, len(rows), mean(pk(sr[m["task"]]) for m in rows), (" (raw prior@60k mean %.3f)" % mean(float(m["prior_pass8"]) for m in rows if m["prior_pass8"])) if grp == "raw_only" else ""))
