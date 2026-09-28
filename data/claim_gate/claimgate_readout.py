#!/usr/bin/env python3
"""Claim gate readout: router episodes (arm, judge score, gate) joined to harbor trials (reward) by session id.

Rules (research note 2026-09-28_opd_ref_evidence.md § "Claim gate test", fixed before any result):
  primary  among LOW first claims (judge logit < threshold), pass rate treat - control >= +0.10 with the 95 % bootstrap
           CI (resampling tasks) above 0; fewer than 60 low claims per arm = uninformative
  (a harm check on all episodes was dropped before any run: the gate acts only on low claims, so the overall
  difference is the primary one diluted; the untouched groups serve as a randomization check instead)
Harness errors (no reward, or an exception that is not an agent ending) and deadline-censored episodes are excluded
from every rate and counted.

    python claimgate_readout.py --run <run dir>   (reads <run>/router/episodes.json and <run>/jobs/*/*/result.json)
"""
import argparse
import collections
import glob
import json
import os
import random
import statistics
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'relay', 'pilot'))
from readout import trials  # noqa: E402

MIN_LOW_PER_ARM = 60


def rate(rows):
    return sum(r['pass'] for r in rows) / len(rows) if rows else float('nan')


def diff_ci(rows, n_boot=2000, seed=0):
    """treat - control pass rate, and its 95 % CI from resampling tasks (all of a task's episodes together)."""
    by_task = collections.defaultdict(list)
    for r in rows:
        by_task[r['task']].append(r)
    tasks = sorted(by_task)

    def d(sample):
        t = [r for k in sample for r in by_task[k] if r['arm'] == 'treat']
        c = [r for k in sample for r in by_task[k] if r['arm'] == 'control']
        return rate(t) - rate(c) if t and c else float('nan')

    rng = random.Random(seed)
    boots = sorted(x for x in (d([rng.choice(tasks) for _ in tasks]) for _ in range(n_boot)) if x == x)
    lo, hi = (boots[int(0.025 * len(boots))], boots[int(0.975 * len(boots)) - 1]) if boots else (float('nan'),) * 2
    return d(tasks), lo, hi


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', required=True, nargs='+', help='one or more run dirs (chunks of one experiment)')
    a = p.parse_args()
    eps, counts = {}, collections.Counter()
    for run in a.run:
        snap = json.load(open(os.path.join(run, 'router', 'episodes.json')))
        eps.update({e['sid']: e for e in snap['episodes'] if e.get('gate_arm')})
        counts.update({k: v for k, v in snap.get('counts', {}).items() if isinstance(v, (int, float))})
    rows, miss = [], collections.Counter()
    for jd in sorted(g for run in a.run for g in glob.glob(os.path.join(run, 'jobs', '*'))):
        for t in trials(jd):
            e = eps.get(t['sid'])
            if e is None:
                miss['no_router_episode'] += 1
                continue
            if t['harness_error']:
                miss['harness_error'] += 1
                continue
            if e.get('ending') == 'deadline':
                miss['deadline_censored'] += 1
                continue
            j = e.get('judge') or {}
            rows.append(dict(task=t['task'], arm=e['gate_arm'], pass_=None, pass_flag=(t['reward'] or 0) >= 1,
                             judged=j.get('logit') is not None, low=bool(j.get('low')), gated=bool(e.get('gated')),
                             logit=j.get('logit'), turns=e.get('turns'), gate_turn=e.get('gate_turn'), exc=t['exc']))
    for r in rows:
        r['pass'] = r.pop('pass_flag')
        r.pop('pass_')
    L = ['# claim gate readout: %s' % ', '.join(os.path.basename(r.rstrip('/')) for r in a.run), '',
         'router counts: %s' % json.dumps(dict(counts)),
         'episodes scored: %d (excluded: %s)' % (len(rows), dict(miss)), '']
    L += ['| group | treat n | treat pass | control n | control pass |', '|---|---|---|---|---|']
    groups = [('all episodes', lambda r: True), ('claimed (judged)', lambda r: r['judged']),
              ('low claims', lambda r: r['judged'] and r['low']), ('high claims', lambda r: r['judged'] and not r['low']),
              ('never claimed / judge error', lambda r: not r['judged'])]
    for name, f in groups:
        t = [r for r in rows if f(r) and r['arm'] == 'treat']
        c = [r for r in rows if f(r) and r['arm'] == 'control']
        L.append('| %s | %d | %.3f | %d | %.3f |' % (name, len(t), rate(t), len(c), rate(c)))
    low = [r for r in rows if r['judged'] and r['low']]
    d_low, lo_low, hi_low = diff_ci(low)
    d_all, lo_all, hi_all = diff_ci(rows)
    n_t = sum(r['arm'] == 'treat' for r in low)
    n_c = sum(r['arm'] == 'control' for r in low)
    untouched = [r for r in rows if not (r['judged'] and r['low'])]
    d_u, lo_u, hi_u = diff_ci(untouched)
    L += ['', 'randomization check, episodes the gate cannot touch, treat - control: %+.3f [%+.3f, %+.3f]' % (d_u, lo_u, hi_u)]
    L += ['low claims, treat - control: %+.3f [%+.3f, %+.3f] (n %d / %d)' % (d_low, lo_low, hi_low, n_t, n_c),
          'all episodes, treat - control: %+.3f [%+.3f, %+.3f]' % (d_all, lo_all, hi_all)]
    gated = [r for r in rows if r['gated']]
    if gated:
        extra = [r['turns'] - r['gate_turn'] for r in gated if r['turns'] is not None and r['gate_turn'] is not None]
        L += ['gated episodes: %d; turns after the gate median %s (p90 %s); agent endings after the gate: %s' % (
            len(gated), statistics.median(extra) if extra else None,
            sorted(extra)[int(0.9 * (len(extra) - 1))] if extra else None,
            dict(collections.Counter(r['exc'] for r in gated if r['exc'])))]
    if min(n_t, n_c) < MIN_LOW_PER_ARM:
        verdict = 'UNINFORMATIVE (fewer than %d low claims per arm)' % MIN_LOW_PER_ARM
    else:
        verdict = 'PASS' if d_low >= 0.10 and lo_low > 0 else 'FAIL'
    L += ['', '**Verdict: %s**' % verdict]
    print('\n'.join(L))


if __name__ == '__main__':
    sys.exit(main())
