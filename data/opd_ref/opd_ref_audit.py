"""Judge the judge: do the per-turn changes of the teacher judge's value (opd_ref_judge.py dense) credit the turns an
independent auditor marks as progress or mistakes? CompassOPD's per-turn score on the same turns is the control.

Auditor labels: one JSON per episode {"trial", "passed", "labels": [{"turn", "label": +1/0/-1, "reason"}],
"pivotal_turn", ...}, written blind to the judge with hindsight of the outcome (opd_ref_judge.py export makes the
transcripts). Turn numbers are the judge's k (assistant turns from 1).

    python opd_ref_audit.py --labels <dir> --dense <judge run dir> [--compass <opd_ref_probe run dir> --refs 2B,9B]
"""
import argparse
import glob
import json
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from opd_ref_probe import load_scores  # noqa: E402


def auc_pm(rows, key, n_boot=2000, seed=0):
    """P(score on a +1 turn > score on a -1 turn), pooled over turns; CI from a bootstrap over episodes."""
    ok = [r for r in rows if r['label'] != 0 and np.isfinite(r.get(key, np.nan))]
    eps = sorted({r['trial'] for r in ok})

    def one(sel):
        pos = np.array([r[key] for r in sel if r['label'] > 0])
        neg = np.array([r[key] for r in sel if r['label'] < 0])
        if len(pos) == 0 or len(neg) == 0:
            return np.nan
        return float(np.mean((pos[:, None] > neg[None, :]) + 0.5 * (pos[:, None] == neg[None, :])))
    est = one(ok)
    by = {t: [r for r in ok if r['trial'] == t] for t in eps}
    rng = np.random.default_rng(seed)
    bs = [one([r for t in rng.choice(eps, len(eps)) for r in by[t]]) for _ in range(n_boot)]
    bs = [b for b in bs if np.isfinite(b)]
    return dict(auc=est, lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)),
                n_pos=sum(r['label'] > 0 for r in ok), n_neg=sum(r['label'] < 0 for r in ok), n_eps=len(eps))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--labels', required=True)
    p.add_argument('--dense', required=True)
    p.add_argument('--compass', default='')
    p.add_argument('--refs', default='2B,9B')
    p.add_argument('--out', default='')
    a = p.parse_args()

    labs = {}
    for f in sorted(glob.glob(os.path.join(a.labels, '*.json'))):
        d = json.load(open(f))
        labs[d['trial']] = d
    V = {}
    for f in glob.glob(os.path.join(a.dense, 'judge', 'dense.*.pkl')):
        for r in pickle.load(open(f, 'rb')):
            if not r.get('para') and r['trial'] in labs:
                V[(r['trial'], r['k'])] = float(1 / (1 + np.exp(-r['logit'])))
    comp = {}
    if a.compass:
        eps = pickle.load(open(os.path.join(a.compass, 'episodes.pkl'), 'rb'))
        refs = a.refs.split(',')
        sc = {m: load_scores(a.compass, m)[0] for m in ['teacher'] + refs}
        for i, e in enumerate(eps):
            if e['trial'] not in labs or not all(i in sc[m] for m in sc):
                continue
            ch = e['chunks']
            t, kind, sl, sh, ql, qh = [ch[:, j] for j in range(6)]
            lq = {m: (lambda c: c[qh] - c[ql])(np.cumsum(np.r_[0, np.nan_to_num(sc[m][i])])) for m in sc}
            n = (sh - sl).astype(float)
            for k in range(e['n_turns']):
                m = (t == k) & (kind == 1)
                if m.any():
                    comp[(e['trial'], k + 1)] = dict(
                        teacher=float(lq['teacher'][m].sum() / n[m].sum()),
                        **{'compass_' + r: float((lq['teacher'][m] - lq[r][m]).sum() / n[m].sum()) for r in refs})
    rows = []
    for trial, d in labs.items():
        for x in d['labels']:
            k = int(x['turn'])
            r = dict(trial=trial, passed=bool(d['passed']), k=k, label=int(x['label']),
                     pivotal=k == d.get('pivotal_turn'))
            if (trial, k) in V:
                r['V'] = V[(trial, k)]
                if (trial, k - 1) in V:
                    r['dV'] = V[(trial, k)] - V[(trial, k - 1)]
            r.update(comp.get((trial, k), {}))
            rows.append(r)
    keys = ['dV', 'V'] + sorted({k for c in comp.values() for k in c})
    M: dict = dict(n_episodes=len(labs), n_turns=len(rows),
             label_counts={str(l): sum(r['label'] == l for r in rows) for l in (1, 0, -1)}, auc={}, abs_dv={})
    L = ['# judge the judge: auditor labels vs per-turn scores', '',
         '%d episodes, %d labelled turns (+1: %d, 0: %d, -1: %d); %d turns with a judge dV' % (
             len(labs), len(rows), *[sum(r['label'] == l for r in rows) for l in (1, 0, -1)],
             sum('dV' in r for r in rows)), '',
         '| score | AUC(+1 turns above -1 turns) | +1 / -1 turns | episodes |', '|---|---|---|---|']
    for k in keys:
        m = auc_pm(rows, k)
        M['auc'][k] = m
        L.append('| %s | %.2f [%.2f, %.2f] | %d / %d | %d |' % (k, m['auc'], m['lo'], m['hi'], m['n_pos'], m['n_neg'],
                                                               m['n_eps']))
    L += ['', '## size of the judge\'s per-turn change by auditor label', '', '| label | n | median abs dV | mean dV |',
          '|---|---|---|---|']
    for lab in (1, 0, -1):
        d = np.array([r['dV'] for r in rows if r['label'] == lab and 'dV' in r])
        M['abs_dv'][str(lab)] = dict(n=len(d), med_abs=float(np.median(np.abs(d))) if len(d) else None,
                                     mean=float(d.mean()) if len(d) else None)
        if len(d):
            L.append('| %+d | %d | %.3f | %+.3f |' % (lab, len(d), np.median(np.abs(d)), d.mean()))
    # pivotal turns: where does the auditor's deciding turn rank among the episode's |dV|, and with what sign?
    piv = []
    for trial, d in labs.items():
        er = [r for r in rows if r['trial'] == trial and 'dV' in r]
        pr = [r for r in er if r['pivotal']]
        if not pr or len(er) < 3:
            continue
        mags = np.array([abs(r['dV']) for r in er])
        piv.append(dict(trial=trial, passed=d['passed'], pct=float((mags < abs(pr[0]['dV'])).mean()),
                        dv=pr[0]['dV'], label=pr[0]['label']))
    if piv:
        M['pivotal'] = piv
        L += ['', '## the auditor\'s pivotal turn', '',
              '%d episodes: its |dV| is at the %.0fth percentile of the episode\'s turns on average (50 = typical); the '
              'judge moves the way the auditor\'s label points on %d of %d pivotal turns labelled +1/-1' % (
                  len(piv), 100 * np.mean([x['pct'] for x in piv]),
                  sum(np.sign(x['dv']) == np.sign(x['label']) for x in piv if x['label'] != 0),
                  sum(x['label'] != 0 for x in piv))]
    out = a.out or a.dense
    json.dump(M, open(os.path.join(out, 'audit.json'), 'w'), indent=1)
    pickle.dump(rows, open(os.path.join(out, 'audit_rows.pkl'), 'wb'))
    open(os.path.join(out, 'audit.md'), 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    sys.exit(main())
