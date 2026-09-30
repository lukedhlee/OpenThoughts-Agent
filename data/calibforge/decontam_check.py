#!/usr/bin/env python3
"""Text-overlap decontamination check between harbor task sets: for every task of a query set, the largest share of
its word n-grams found in any single task of a reference set (containment), on the instruction and on the tests.

Word n-grams: lowercase tokens [a-z0-9_]+, instructions 8-grams, tests 10-grams (tests/*.py, *.sh). Boilerplate is
removed first: an n-gram in more than --boiler of the reference tasks (harness templates, pytest scaffolding) does not
count. A query task's containment is |its n-grams in reference task r| / |its n-grams|, maximised over r; 1.0 = all of
its text appears in one reference task, 0 = none.

    python decontam_check.py --set heldout=<dir or list>:<cf root> ... (see --help)

Sets are given as NAME=SPEC, SPEC one of:
  tree:<dir>                    every <dir>/<task>/ with instruction.md
  cf:<calibforge root>          every task of a CalibForge checkout (<family>/<task>/...), id calibforge__<family>__<task>
  cf:<calibforge root>@<list>   those CalibForge tasks whose id is in <list> (one id per line)
  cfmanifest:<root>@<jsonl>     the CalibForge tasks named by a manifest's 'task' field
Pairs to compare: --pair QUERY:REF (repeatable). Output: a summary per pair and field, and the top pairs.
"""
import argparse
import collections
import json
import os
import re
import sys

TOK = re.compile(r'[a-z0-9_]+')


def load_task(d):
    ins = open(os.path.join(d, 'instruction.md'), errors='replace').read() if os.path.exists(os.path.join(d, 'instruction.md')) else ''
    tests = []
    td = os.path.join(d, 'tests')
    if os.path.isdir(td):
        for root, _, files in os.walk(td):
            for f in sorted(files):
                if f.endswith(('.py', '.sh')):
                    tests.append(open(os.path.join(root, f), errors='replace').read())
    return dict(instruction=ins, tests='\n'.join(tests))


def load_set(spec):
    kind, _, arg = spec.partition(':')
    out = {}
    if kind == 'tree':
        for t in sorted(os.listdir(arg)):
            if os.path.exists(os.path.join(arg, t, 'instruction.md')):
                out[t] = load_task(os.path.join(arg, t))
        return out
    root, _, sel = arg.partition('@')
    want = None
    if kind == 'cf' and sel:
        want = {l.strip() for l in open(sel) if l.strip()}
    elif kind == 'cfmanifest':
        want = {json.loads(l)['task'] for l in open(sel) if l.strip()}
    for fam in sorted(os.listdir(root)):
        fd = os.path.join(root, fam)
        if not os.path.isdir(fd) or fam.startswith('.'):
            continue
        for t in sorted(os.listdir(fd)):
            tid = f'calibforge__{fam}__{t}'
            if want is not None and tid not in want:
                continue
            if os.path.exists(os.path.join(fd, t, 'instruction.md')):
                out[tid] = load_task(os.path.join(fd, t))
    if want is not None and len(out) != len(want):
        print(f'warning: {spec}: {len(want) - len(out)} of {len(want)} listed tasks not found', file=sys.stderr)
    return out


def grams(text, n):
    w = TOK.findall(text.lower())
    return {hash(' '.join(w[i:i + n])) for i in range(len(w) - n + 1)}


def compare(Q, R, field, n, boiler, same_ok=False):
    rg = {r: grams(R[r][field], n) for r in R}
    df = collections.Counter(g for s in rg.values() for g in s)
    cut = max(2, int(boiler * len(R)))
    common = {g for g, c in df.items() if c > cut}
    index = collections.defaultdict(list)
    for r, s in rg.items():
        for g in s - common:
            index[g].append(r)
    res = {}
    for q in Q:
        s = grams(Q[q][field], n) - common
        if not s:
            res[q] = (0.0, None, 0)
            continue
        c = collections.Counter(r for g in s for r in index.get(g, ()) if same_ok or r != q)
        if not c:
            res[q] = (0.0, None, len(s))
            continue
        r, k = c.most_common(1)[0]
        res[q] = (k / len(s), r, len(s))
    return res, len(common)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--set', action='append', required=True, help='NAME=SPEC')
    ap.add_argument('--pair', action='append', required=True, help='QUERY:REF')
    ap.add_argument('--boiler', type=float, default=0.05)
    ap.add_argument('--top', type=int, default=8)
    ap.add_argument('--out')
    a = ap.parse_args()
    S = {}
    for s in a.set:
        name, spec = s.split('=', 1)
        S[name] = load_set(spec)
        print(f'set {name}: {len(S[name])} tasks', file=sys.stderr)
    report = {}
    for p in a.pair:
        qn, rn = p.split(':')
        for field, n in (('instruction', 8), ('tests', 10)):
            res, nb = compare(S[qn], S[rn], field, n, a.boiler)
            v = sorted((x[0] for x in res.values()), reverse=True)
            bands = {f'>={t}': sum(1 for x in v if x >= t) for t in (0.1, 0.2, 0.5, 0.8)}
            top = sorted(res.items(), key=lambda kv: -kv[1][0])[:a.top]
            report[f'{qn}:{rn}:{field}'] = dict(n=len(v), boiler_grams=nb, max=round(v[0], 3) if v else None,
                                               p99=round(v[int(0.01 * len(v))], 3) if v else None, bands=bands,
                                               top=[(q, round(x[0], 3), x[1], x[2]) for q, x in top])
            print(f'\n{qn} vs {rn} [{field}, {n}-grams, {nb} boilerplate grams dropped]: {len(v)} tasks; max {v[0]:.3f}, '
                  f'p99 {v[int(0.01 * len(v))]:.3f}; tasks with containment {bands}')
            for q, x in top:
                print(f'  {x[0]:.3f}  {q}  <-  {x[1]}  ({x[2]} grams)')
    if a.out:
        json.dump(report, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
