#!/usr/bin/env python3
"""Tabulate paired_rollout_probe.py timing files (tokenize / terminal_exec / generation) and stats json.

usage: compare_pair_timing.py <name>=<run_dir>/<name>_timing.jsonl ...  (stats json is looked up next to it)
"""
import collections, json, sys
from pathlib import Path


def q(xs, p):
    xs = sorted(xs)
    return round(xs[int((len(xs) - 1) * p)], 3) if xs else None


def load(path):
    rows = [json.loads(l) for l in Path(path).read_text().splitlines()]
    by = collections.defaultdict(list)
    for r in rows:
        if r['kind'] in ('tokenize', 'terminal_exec', 'generation'):
            by[r['kind']].append(r)
    created = sum(r['kind'] == 'environment_created' for r in rows)
    out = {}
    for k in ('tokenize', 'terminal_exec', 'generation'):
        rs = by[k]
        ok = [r['seconds'] for r in rs if not r.get('error')]
        errs = collections.Counter(r['error'] for r in rs if r.get('error'))
        out[k] = dict(calls=len(rs), errors=sum(errs.values()), error_types=dict(errs),
                      p50=q(ok, .5), p90=q(ok, .9), p99=q(ok, .99), max=round(max(ok), 3) if ok else None)
    stats_path = Path(path).with_name(Path(path).name.replace('_timing.jsonl', '_stats.json'))
    stats = json.loads(stats_path.read_text()) if stats_path.exists() else None
    return dict(environments_created=created, kinds=out, stats=stats)


if __name__ == '__main__':
    runs = {a.split('=', 1)[0]: load(a.split('=', 1)[1]) for a in sys.argv[1:]}
    print('| backend | env created | tokenize calls / errors / p50 / p99 s | terminal calls / errors / p50 / p99 s | generation calls / errors / p50 / p99 s | trials done / errored | reward mean |')
    print('|---|---:|---|---|---|---|---:|')
    for name, r in runs.items():
        k = r['kinds']
        st = (r['stats'] or {}).get('stats') or {}
        ev = next(iter((st.get('evals') or {}).values()), {}) if st else {}
        mean = (ev.get('metrics') or [{}])[0].get('mean') if ev else None
        cell = lambda x: f"{x['calls']} / {x['errors']} / {x['p50']} / {x['p99']}"
        print(f"| {name} | {r['environments_created']} | {cell(k['tokenize'])} | {cell(k['terminal_exec'])} | {cell(k['generation'])} | "
              f"{st.get('n_completed_trials')} / {st.get('n_errored_trials')} | {mean} |")
        for kind, x in k.items():
            if x['error_types']:
                print(f"    {name} {kind} errors: {x['error_types']}")
        if ev and ev.get('exception_stats'):
            print(f"    {name} trial exceptions: {{ {', '.join(f'{k}: {len(v)}' for k, v in ev['exception_stats'].items())} }}")
