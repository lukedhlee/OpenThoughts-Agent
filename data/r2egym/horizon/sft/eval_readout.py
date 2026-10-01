#!/usr/bin/env python3
"""Readout of eval_sft.sh: pass@1 per set (TB2.1, SWE-bench Verified random-100, TB-lite) for each model tag, from
REPS independent runs, and the paired difference between two tags.

Outcomes are paired_eval.py's TB2 rule, unchanged: every trial of a harbor job dir, usable = the verifier returned a
reward, pass = reward >= 1. A set's shards (swe_s0 + swe_s1) merge per task within a rep. Per rep: the pass rate over
usable tasks (Wilson 95 % CI). Across reps: a task's score is its mean over the reps where it was usable, pass@1 = the
mean over tasks, 95 % CI by bootstrap over tasks (10,000 resamples, seed 20260928). Two tags: paired difference of the
per-task scores over tasks usable for both, same bootstrap. --ref TAG=SET:<job dir>[,<job dir>] adds a single-run
reference (e.g. Jupiter's arm A trials) to compare against.

    python eval_readout.py --tags hzA hzB [--day 20260930] [--jobs <dir>] [--ref jupA=tb21:<dir> ...] [--out r.json]
"""
import argparse
import collections
import glob
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', '..', 'relay', 'sft'))
import paired_eval as pe  # noqa: E402

SETS = {'tb21': ('tb21',), 'swe': ('swe_s0', 'swe_s1'), 'tblite': ('tblite_s0', 'tblite_s1')}
SEED, NBOOT = 20260928, 10000


def boot(vals):
    rng = random.Random(SEED)
    ms = sorted(sum(rng.choice(vals) for _ in vals) / len(vals) for _ in range(NBOOT))
    return [round(ms[int(0.025 * NBOOT)], 4), round(ms[int(0.975 * NBOOT) - 1], 4)]


def se(vals):
    """standard error of the mean over tasks"""
    n = len(vals)
    if n < 2:
        return None
    m = sum(vals) / n
    return round((sum((v - m) ** 2 for v in vals) / (n - 1) / n) ** 0.5, 4)


def merged(dirs):
    o, c = {}, collections.Counter()
    for d in dirs:
        x, n = pe.tb2_outcomes(d)
        o.update(x)
        c.update(n)
    return o, dict(c)


def model_set(jobs, tag, fam, day):
    """{rep: outcomes} for one tag and set family, from <jobs>/<shard>_<tag>_r<rep>_<day>."""
    reps = collections.defaultdict(list)
    for sh in SETS[fam]:
        for d in sorted(glob.glob(f'{jobs}/{sh}_{tag}_r*_{day}')):
            rep = d.rsplit('_r', 1)[1].split('_')[0]
            reps[rep].append(d)
    return {r: merged(ds) for r, ds in sorted(reps.items())}


def scores(rep_outcomes):
    per = collections.defaultdict(list)
    for o, _ in rep_outcomes.values():
        for t, v in o.items():
            per[t].append(v)
    return {t: sum(v) / len(v) for t, v in per.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--tags', nargs='+', required=True)
    ap.add_argument('--day', default='*')
    ap.add_argument('--jobs', default=f"/scratch/11584/{os.environ.get('USER')}/experiments/sft_eval/tb2_jobs")
    ap.add_argument('--ref', nargs='*', default=[], help='TAG=SET:<job dir>[,<job dir>] single-run references')
    ap.add_argument('--out')
    ap.add_argument('--exclude', help='regex of task names to drop from every set (e.g. ^sympy__ : the RL pool shares that '
                    'repo with SWE-bench random-100, so report the gain with and without it)')
    a = ap.parse_args()
    import re
    drop = re.compile(a.exclude) if a.exclude else None
    res, sc = {}, {}
    for tag in a.tags:
        for fam in SETS:
            ro = model_set(a.jobs, tag, fam, a.day)
            if not ro:
                continue
            s = scores(ro)
            if drop:
                s = {t: v for t, v in s.items() if not drop.search(t)}
            sc[(tag, fam)] = s
            res[f'{tag}/{fam}'] = dict(
                reps={r: dict(pe.rate(o), counts=c) for r, (o, c) in ro.items()},
                tasks=len(s), pass_at_1=round(sum(s.values()) / len(s), 4) if s else None,
                se=se(list(s.values())), ci95=boot(list(s.values())) if s else None)
    for spec in a.ref:
        tag, rest = spec.split('=', 1)
        fam, dirs = rest.split(':', 1)
        o, c = merged(dirs.split(','))
        sc[(tag, fam)] = {t: float(v) for t, v in o.items()}
        res[f'{tag}/{fam}'] = dict(pe.rate(o), counts=c)
    tags = list(dict.fromkeys(a.tags + [s.split('=', 1)[0] for s in a.ref]))
    for i, x in enumerate(tags):
        for y in tags[i + 1:]:
            for fam in SETS:
                if (x, fam) in sc and (y, fam) in sc:
                    X, Y = sc[(x, fam)], sc[(y, fam)]
                    both = sorted(set(X) & set(Y))
                    d = [X[t] - Y[t] for t in both]
                    if d:
                        res[f'{x}-{y}/{fam}'] = dict(tasks=len(both), diff=round(sum(d) / len(d), 4), se=se(d), ci95=boot(d))
    for k, v in res.items():
        if 'pass_at_1' in v:
            reps = ' '.join(f"r{r}={x['rate']}({x['n']})" for r, x in v['reps'].items())
            print(f"{k:16s} pass@1 {v['pass_at_1']} ± {v['se']} {v['ci95']} over {v['tasks']} tasks; {reps}")
        elif 'diff' in v:
            print(f"{k:16s} diff {v['diff']:+.4f} ± {v['se']} {v['ci95']} over {v['tasks']} tasks")
        else:
            print(f"{k:16s} {v['rate']} {v['ci95']} ({v['n']}) [single run]")
    if a.out:
        json.dump(res, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
