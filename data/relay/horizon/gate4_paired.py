#!/usr/bin/env python3
"""Port gate 4 (ai_memory/active/horizon-port/objective.md): the same checkpoint's pass@1 on Horizon and on Jupiter, on a
task set Jupiter already scored, paired per task. PASS iff the paired 95 % CI of Horizon - Jupiter includes 0 and
|diff| < 0.03.

A thin adapter over data/relay/sft/paired_eval.py, whose outcome rules, Wilson CI and paired bootstrap (10,000
resamples over tasks, seed 20260928) are used unchanged; only its fixed model keys (0921/A/B/C) do not fit a
site-vs-site comparison, so this script calls its functions directly.
  --kind heldout: each dir is a run dir (<run>/jobs/<run>_student_only[_p2]); usable = paired_eval.heldout_outcomes.
                  A bare harbor job dir <x>_student_only is accepted too (the Jupiter reference tarballs hold only that).
  --kind tb2:     each dir is a harbor job dir; usable = the verifier returned a reward (paired_eval.tb2_outcomes).
Several dirs per side (shards, or a retry) merge per task, the later dir winning, as paired_eval.py does.

    python gate4_paired.py --set heldout300 --kind heldout --jupiter <dir> --horizon <dir> [--out x.json]
"""
import argparse
import collections
import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'sft'))
import paired_eval as pe  # noqa: E402


def heldout_dir(d):
    """A run dir as paired_eval.heldout_outcomes wants it; a bare <run>_student_only job dir gets a temporary run dir."""
    d = os.path.normpath(d)
    base = os.path.basename(d)
    if base.endswith('_student_only') and os.path.isdir(d) and not os.path.isdir(os.path.join(d, 'jobs')):
        run = base[:-len('_student_only')]
        tmp = tempfile.mkdtemp(prefix='gate4_')
        os.makedirs(os.path.join(tmp, run))
        os.symlink(os.path.dirname(d), os.path.join(tmp, run, 'jobs'))
        return os.path.join(tmp, run)
    return d


def outcomes(kind, dirs):
    merged, cnt = {}, collections.Counter()
    for d in dirs:
        o, c = pe.heldout_outcomes(heldout_dir(d)) if kind == 'heldout' else pe.tb2_outcomes(d)
        merged.update(o)
        cnt.update(c)
    return merged, dict(cnt)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--set', required=True)
    ap.add_argument('--kind', choices=('heldout', 'tb2'), required=True)
    ap.add_argument('--jupiter', nargs='+', required=True)
    ap.add_argument('--horizon', nargs='+', required=True)
    ap.add_argument('--max-abs-diff', type=float, default=0.03)
    ap.add_argument('--out')
    a = ap.parse_args()
    jo, jc = outcomes(a.kind, a.jupiter)
    ho, hc = outcomes(a.kind, a.horizon)
    p = pe.paired(ho, jo)
    ok = bool(p.get('ci95')) and p['ci95'][0] <= 0 <= p['ci95'][1] and abs(p['diff']) < a.max_abs_diff
    res = dict(set=a.set, kind=a.kind, jupiter=dict(pe.rate(jo), counts=jc, dirs=a.jupiter),
               horizon=dict(pe.rate(ho), counts=hc, dirs=a.horizon), paired_horizon_minus_jupiter=p,
               rule=f'PASS iff the paired 95 % CI includes 0 and |diff| < {a.max_abs_diff}',
               verdict='PASS' if ok else 'FAIL')
    if a.out:
        json.dump(res, open(a.out, 'w'), indent=1)
    f = lambda r: f"{r['rate']:.3f} [{r['ci95'][0]:.3f}, {r['ci95'][1]:.3f}] ({r['n']} scored, {r['passes']} pass)"  # noqa: E731
    print(f"{a.set}: Jupiter {f(res['jupiter'])}; Horizon {f(res['horizon'])}")
    if p.get('ci95'):
        print(f"  paired Horizon - Jupiter over {p['tasks']} tasks: {p['diff']:+.4f} [{p['ci95'][0]:+.4f}, {p['ci95'][1]:+.4f}]"
              f" (Horizon-only passes {p['x_only']}, Jupiter-only {p['y_only']}) -> gate 4 {res['verdict']}")
    else:
        print('  no task scored on both sides -> gate 4 FAIL')


if __name__ == '__main__':
    main()
