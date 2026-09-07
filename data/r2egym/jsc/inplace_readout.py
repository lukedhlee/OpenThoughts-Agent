#!/usr/bin/env python3
"""inplace_readout.py — the in-place-clause A/B read-out: does naming "there is no patch file, edit in place"
change where the model writes, and does that change the score?

Input: one or more `<probe>_turns.jsonl` extracts (hist_extract.py, which now carries the per-turn write flags
es/et/ws/pf/uw from edit_target.py). Pass `label=path` per arm; the first label is the reference.

Reports, per arm and then paired:
  1. per-task pass on tasks fully sampled in every arm  (the score question)
  2. where writes landed, over SCORED attempts: repo source / repo test / scratch only / patch file / nothing
  3. the same partition over LOSING attempts only -- directly comparable to the 09-03 report's
     "edited repo source 61 % / wrote only to /tmp 20 % / wrote nothing 19 %" and to the 09-07 loser taxonomy
  4. P(win | wrote repo source), the conversion rate the +5 pt ceiling was computed from

Three readings to refuse, carried over from the 09-07 spin-off:
  - context deaths are NOT diagnostic here (94 % of base losses die on context; every subgroup sits at that rate)
  - a shift in `pf` with no shift in `es` is a change of habit, not of outcome; only `es` and pass can move the score
  - `uw` (a write we can see but cannot resolve) is a measurement limit, not a behaviour -- never merge it into a class

Usage:  python3 inplace_readout.py ctl=<...>_turns.jsonl inp=<...>_turns.jsonl [--tasks <file>] [--first8]
"""
import argparse, json, math, os, sys
from collections import defaultdict

BRIDGE = {"BridgeOutageError", "BridgeOperationTimeoutError", "BridgeOperationError"}

ap = argparse.ArgumentParser()
ap.add_argument("arms", nargs="+", help="label=path to <probe>_turns.jsonl")
ap.add_argument("--tasks", default=None, help="restrict to these task ids")
ap.add_argument("--first8", action="store_true", help="use the first 8 scored attempts per task (see SCHEMA conventions)")
ap.add_argument("--min-scored", type=int, default=8, help="a task is fully sampled at this many scored attempts")
a = ap.parse_args()
ONLY = {l.strip() for l in open(a.tasks) if l.strip()} if a.tasks else None


def load(path):
    """-> {task: [attempt, ...]} with one dict per SCORED attempt (reward not null, no bridge exception)."""
    by = defaultdict(list)
    n_raw = n_null = n_bridge = 0
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            n_raw += 1
            if ONLY is not None and r["task"] not in ONLY:
                continue
            if r.get("exc") in BRIDGE:
                n_bridge += 1
                continue
            if r.get("reward") is None:
                n_null += 1
                continue
            t = r["turns"] or []
            by[r["task"]].append(dict(
                win=float(r["reward"]) > 0.5,
                es=any(x.get("es") for x in t), et=any(x.get("et") for x in t),
                ws=any(x.get("ws") for x in t), pf=any(x.get("pf") for x in t),
                uw=any(x.get("uw") for x in t),
                ctx=(r.get("exc") == "ContextLengthExceededError"),
                turns=r.get("n_turns") or 0,
            ))
    if a.first8:
        by = {t: v[:8] for t, v in by.items()}
    return by, n_raw, n_null, n_bridge


def share(atts, pred):
    n = len(atts)
    return (sum(1 for x in atts if pred(x)) / n if n else float("nan")), n


def partition(atts):
    """where the writes landed, one class per attempt, most-committing first."""
    c = defaultdict(int)
    for x in atts:
        if x["es"]:
            c["repo source"] += 1
        elif x["et"]:
            c["repo test only"] += 1
        elif x["ws"]:
            c["scratch only"] += 1
        elif x["uw"]:
            c["unresolvable write"] += 1
        else:
            c["no write seen"] += 1
    return c


def fmt_partition(c, n, indent="    "):
    out = []
    for k in ("repo source", "repo test only", "scratch only", "unresolvable write", "no write seen"):
        out.append("%s%-20s %5d  %5.1f %%" % (indent, k, c[k], 100.0 * c[k] / n if n else float("nan")))
    return "\n".join(out)


arms = {}
for spec in a.arms:
    label, path = spec.split("=", 1)
    arms[label], nr, nn, nb = load(path)
    tot = sum(len(v) for v in arms[label].values())
    full = sum(1 for v in arms[label].values() if len(v) >= a.min_scored)
    print("%-8s %s\n         %d attempt records, %d scored, %d tasks (%d fully sampled at >=%d), %d infra-null, %d bridge"
          % (label, os.path.basename(path), nr, tot, len(arms[label]), full, a.min_scored, nn, nb))

print("\n" + "=" * 96 + "\n1. SCORE  (per-task pass = wins / scored, on tasks fully sampled in every arm)\n")
common = None
for label, by in arms.items():
    s = {t for t, v in by.items() if len(v) >= a.min_scored}
    common = s if common is None else (common & s)
common = sorted(common or [])
rate = {label: {t: sum(1 for x in by[t] if x["win"]) / len(by[t]) for t in common} for label, by in arms.items()}
print("   paired on n=%d tasks" % len(common))
for label in arms:
    print("     %-8s %.3f" % (label, sum(rate[label].values()) / len(common) if common else float("nan")))
ref = list(arms)[0]
for label in list(arms)[1:]:
    d = [rate[label][t] - rate[ref][t] for t in common]
    n = len(d)
    if n > 1:
        m = sum(d) / n
        sd = math.sqrt(sum((x - m) ** 2 for x in d) / (n - 1))
        ci = 1.96 * sd / math.sqrt(n)
        moved = sum(1 for x in d if x != 0)
        print("     %-8s - %-8s %+.3f [%+.3f, %+.3f]  n=%d  (%d tasks moved)" % (label, ref, m, m - ci, m + ci, n, moved))

for title, sel in (("2. WHERE WRITES LANDED  (all scored attempts)", lambda x: True),
                   ("3. WHERE WRITES LANDED  (LOSING attempts only)", lambda x: not x["win"])):
    print("\n" + "=" * 96 + "\n%s\n" % title)
    for label, by in arms.items():
        atts = [x for t in common for x in by[t] if sel(x)]
        n = len(atts)
        print("  %-8s n=%d" % (label, n))
        print(fmt_partition(partition(atts), n))
        pf, _ = share(atts, lambda x: x["pf"])
        es, _ = share(atts, lambda x: x["es"])
        ctx, _ = share(atts, lambda x: x["ctx"])
        print("    %-20s %5.1f %%   (ever wrote a patch file)" % ("patch file", 100 * pf))
        print("    %-20s %5.1f %%   (ever wrote repo source, incl. attempts that also wrote elsewhere)" % ("edited source", 100 * es))
        print("    %-20s %5.1f %%   (context deaths -- NOT diagnostic, read against the arm's own base rate)" % ("ctx death", 100 * ctx))

print("\n" + "=" * 96 + "\n4. CONVERSION  P(win | class), all scored attempts on the paired tasks\n")
print("  %-8s %-22s %8s %8s" % ("arm", "class", "n", "P(win)"))
for label, by in arms.items():
    atts = [x for t in common for x in by[t]]
    for name, pred in (("wrote repo source", lambda x: x["es"]),
                       ("scratch only", lambda x: not x["es"] and not x["et"] and x["ws"]),
                       ("no write seen", lambda x: not (x["es"] or x["et"] or x["ws"] or x["uw"])),
                       ("wrote a patch file", lambda x: x["pf"])):
        g = [x for x in atts if pred(x)]
        print("  %-8s %-22s %8d %7.3f" % (label, name, len(g), (sum(1 for x in g if x["win"]) / len(g)) if g else float("nan")))
