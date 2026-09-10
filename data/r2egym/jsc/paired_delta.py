#!/usr/bin/env python3
"""Paired per-task pass-rate deltas between probes: python3 paired_delta.py label=prefix [label=prefix ...]
N-way paired mean per-task pass on tasks fully sampled by every probe; then each pair on the tasks fully sampled by both:
mean(b) - mean(a) with a 95 % interval (1.96 * sd / sqrt(n)) and the n. First label is the reference for the one-vs-all block."""
import csv, sys, itertools, math, os
EXCLUDE = set(l.strip() for l in open(os.environ["EXCLUDE"])) if os.environ.get("EXCLUDE") else set()
probes = {}
for arg in sys.argv[1:]:
    label, prefix = arg.split("=", 1)
    rows = {r["task"]: r for r in csv.DictReader(open(prefix + "_pass8_table.csv"))}
    probes[label] = {t: int(r["succ"]) / int(r["scored"]) for t, r in rows.items() if r["full"] == "True" and int(r["scored"]) > 0 and t not in EXCLUDE}
def delta(a, b):
    both = sorted(set(a) & set(b)); d = [b[t] - a[t] for t in both]; n = len(d)
    m = sum(d) / n; sd = math.sqrt(sum((x - m) ** 2 for x in d) / (n - 1)); return m, 1.96 * sd / math.sqrt(n), n
common = set.intersection(*(set(p) for p in probes.values()))
print(f"{len(probes)}-way paired on n={len(common)} tasks fully sampled by all:")
for label, p in probes.items():
    print(f"  {label:16s} {sum(p[t] for t in common) / len(common):.3f}   (own n={len(p)}: {sum(p.values()) / len(p):.3f})")
print("\npairwise (b - a), 95 % interval, n:")
for a, b in itertools.combinations(probes, 2):
    m, ci, n = delta(probes[a], probes[b]); print(f"  {a:14s} -> {b:14s} {m:+.3f} +- {ci:.3f}  n={n}")
