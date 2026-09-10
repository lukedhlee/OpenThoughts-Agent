#!/usr/bin/env python3
"""ttwave_noise.py — what bucket agreement / |Δ|≥0.25 rate would two independent k=8 draws of the SAME task produce?
Uses each overlap task's pooled estimate p = (TT successes + raw@60k successes)/16 as the truth; 2000 Monte-Carlo replicates."""
import csv, glob, random, os
X = "/e/fscratch/reformo/lee27/experiments"; E = X + "/ttsample"; random.seed(1)
res = {}
for f in glob.glob(X + "/ttwave_s*/pass8_pass8_table.csv"):
    for r in csv.DictReader(open(f)):
        if r.get("full") == "True": res[r["task"]] = float(r["pass_at_k"])
man = [m for m in csv.DictReader(open(E + "/ttwave_manifest.tsv"), delimiter="\t") if m["group"] == "overlap_tt" and m["task"] in res]
tt = [res[m["task"]] for m in man]; raw = [float(m["raw_pass8_60k"]) for m in man]
b = lambda v: 0 if v == 0 else (2 if v == 1 else 1)
obs_agree = sum(b(t) == b(r) for t, r in zip(tt, raw)) / len(man); obs_big = sum(abs(t - r) >= 0.25 for t, r in zip(tt, raw)) / len(man)
obs_mean = sum(t - r for t, r in zip(tt, raw)) / len(man)
ag, bg, md = [], [], []
for _ in range(2000):
    a = c = d = 0.0
    for t, r in zip(tt, raw):
        p = (t + r) / 2
        x = sum(random.random() < p for _ in range(8)) / 8; y = sum(random.random() < p for _ in range(8)) / 8
        a += b(x) == b(y); c += abs(x - y) >= 0.25; d += x - y
    ag.append(a / len(man)); bg.append(c / len(man)); md.append(d / len(man))
q = lambda v, k: sorted(v)[int(k * len(v))]
print("n=%d overlap tasks tabled" % len(man))
print("bucket agreement: observed %.3f | same-task noise expectation %.3f (95%% band %.3f–%.3f)" % (obs_agree, sum(ag) / len(ag), q(ag, .025), q(ag, .975)))
print("|delta|>=0.25 rate: observed %.3f | noise expectation %.3f (band %.3f–%.3f)" % (obs_big, sum(bg) / len(bg), q(bg, .025), q(bg, .975)))
print("mean(TT - raw@60k): observed %+.3f | noise band %+.3f–%+.3f" % (obs_mean, q(md, .025), q(md, .975)))
