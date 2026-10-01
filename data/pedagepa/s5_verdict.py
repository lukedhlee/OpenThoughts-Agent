#!/usr/bin/env python3
"""PedaGEPA stage 5 verdict, by the rule fixed 2026-09-30 before any stage-5 result.

Behaviour (primary): per trajectory, composite = mean of the numeric scores of P7, P8, P9, P10 (NA / LD left out; a
trajectory with none is dropped). P - C1 paired over the tasks where both have a composite, task bootstrap (10,000,
seed 20260928). PASS iff >= +0.20 with the 95 % CI above 0.
Pass rate (secondary): mean of the TB2.1 and SWE paired differences of per-task 3-run means (eval_readout.py's per-task
scores), each set resampled on its own and the two means averaged; "helps" iff CI above 0, "hurts" iff below.
Verdict: behaviour PASS + pass helps -> PASS; behaviour PASS, pass not helps -> PARTIAL; behaviour FAIL -> FAIL;
P - C0 pass hurts -> FAIL.

    python s5_verdict.py --opus <opus_eval dir> --readout <readout dir> [--out verdict.json]
"""
import argparse
import glob
import json
import os
import random
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'r2egym', 'horizon', 'sft'))
sys.path.insert(0, os.path.join(HERE, '..', 'relay', 'sft'))
ITEMS = ('P7', 'P8', 'P9', 'P10')
SEED, NB = 20260928, 10000


def boot_mean(diffs, rng):
    n = len(diffs)
    ms = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(NB))
    return sum(diffs) / n, ms[int(.025 * NB)], ms[int(.975 * NB) - 1]


def composites(opus_dir):
    mp = json.load(open(os.path.join(opus_dir, 'id_map_PRIVATE.json')))
    out, items = {}, {}
    for f in glob.glob(os.path.join(opus_dir, 'opus', 'batch*.jsonl')):
        for l in open(f):
            if not l.strip():
                continue
            j = json.loads(l)
            m = mp[j['id']]
            sc = j.get('scores', {})
            vals = {k: sc.get(k, {}).get('score') for k in ITEMS}
            num = [v for v in vals.values() if isinstance(v, (int, float))]
            key = (m['set'], m['task'])
            items.setdefault(m['arm'], {})[key] = vals
            if num:
                out.setdefault(m['arm'], {})[key] = sum(num) / len(num)
    return out, items


def paired(a, b, rng):
    ks = sorted(set(a) & set(b))
    return boot_mean([a[k] - b[k] for k in ks], rng) + (len(ks),)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--opus', required=True)
    ap.add_argument('--readout', required=True, help='dir with eval_readout --out json files <a>_vs_<b>.json')
    ap.add_argument('--out')
    a = ap.parse_args()
    comp, items = composites(a.opus)
    rep = {'n_composite': {k: len(v) for k, v in comp.items()}}
    rng = random.Random(SEED)
    for x, y in (('pgp', 'pgc1'), ('pgp', 'pgc0'), ('pgc1', 'pgc0')):
        m, lo, hi, n = paired(comp[x], comp[y], rng)
        rep[f'behaviour_{x}-{y}'] = dict(diff=round(m, 3), ci=[round(lo, 3), round(hi, 3)], tasks=n)
        for it in ITEMS:
            ax = {k: v[it] for k, v in items[x].items() if isinstance(v[it], (int, float))}
            ay = {k: v[it] for k, v in items[y].items() if isinstance(v[it], (int, float))}
            if len(set(ax) & set(ay)) >= 5:
                mm, l2, h2, n2 = paired(ax, ay, rng)
                rep[f'item_{it}_{x}-{y}'] = dict(diff=round(mm, 3), ci=[round(l2, 3), round(h2, 3)], tasks=n2)
    for arm, d in comp.items():
        rep[f'composite_mean_{arm}'] = round(st.mean(d.values()), 3)
    b = rep['behaviour_pgp-pgc1']
    rep['behaviour_clause'] = 'PASS' if b['diff'] >= 0.20 and b['ci'][0] > 0 else 'FAIL'
    # pass rate: eval_readout.py's per-set paired bootstraps (each set resampled on its own); the average of two
    # independent bootstrap means has SE = sqrt(se1^2 + se2^2) / 2 (normal approximation of the averaged bootstrap)
    for x, y in (('pgp', 'pgc0'), ('pgp', 'pgc1'), ('pgc1', 'pgc0')):
        r = json.load(open(os.path.join(a.readout, f'{x}_vs_{y}.json')))
        t, w = r[f'{x}-{y}/tb21'], r[f'{x}-{y}/swe']
        d, se = (t['diff'] + w['diff']) / 2, (t['se'] ** 2 + w['se'] ** 2) ** .5 / 2
        lo, hi = d - 1.96 * se, d + 1.96 * se
        rep[f'pass_{x}-{y}'] = dict(tb21=t['diff'], swe=w['diff'], mean=round(d, 4), ci=[round(lo, 4), round(hi, 4)],
                                    call='helps' if lo > 0 else ('hurts' if hi < 0 else 'no detectable effect'))
    pc = rep['pass_pgp-pgc0']['call']
    beh = rep['behaviour_clause']
    rep['verdict'] = ('FAIL' if pc == 'hurts' or beh == 'FAIL' else ('PASS' if pc == 'helps' else 'PARTIAL'))
    json.dump(rep, open(a.out, 'w'), indent=1) if a.out else None
    print(json.dumps(rep, indent=1))


if __name__ == '__main__':
    main()
