#!/usr/bin/env python3
"""Trim a kept manifest (select_kept.py output) to an exact pass/fail count, so two SFT arms match by size
(Luke 2026-09-27: the baseline and relay arms use different task pools by design, so they are matched by count, not
by task). Reads the manifest only; the original is left untouched.

    python match_kept.py --manifest kept_manifest.jsonl --quarantine quarantine_tasks.txt \
        --passes 591 --failures 591 --weak-timeouts fill --out kept_manifest_matched.jsonl

  1. Rows of quarantined tasks are dropped (the relay arm's always-erroring tasks, so neither arm trains on them).
  2. A weak timeout is a timeout failure that had not stalled (select_kept's `stalled` is empty): under CLOCK=wall
     the agent budget also pays for the model's own reply time, so such an episode was usually cut mid-work by slow
     replies, not by the agent getting stuck. --weak-timeouts:
       keep  treat them like any other failure;
       drop  never keep them (fails if the rest cannot fill --failures);
       fill  keep every other failure first and fill only the shortfall with weak timeouts (marked
             weak_timeout_fill=true).
  3. Passes, and failures within each pool, are sampled uniformly with a fixed seed; causes are not balanced by hand.
"""
import argparse
import collections
import json
import random


def weak_timeout(r):
    return r['cause'] == 'timeout' and not r.get('stalled')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--quarantine', action='append', default=[], help='file of task names to drop (repeatable)')
    ap.add_argument('--passes', type=int, required=True)
    ap.add_argument('--failures', type=int, required=True)
    ap.add_argument('--weak-timeouts', choices=['keep', 'drop', 'fill'], default='keep')
    ap.add_argument('--seed', type=int, default=20260927)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    rows = [json.loads(line) for line in open(a.manifest) if line.strip()]
    quarantine = {t.strip() for q in a.quarantine for t in open(q) if t.strip()}
    rows_q = [r for r in rows if r['task'] not in quarantine]
    rng = random.Random(a.seed)
    passes = sorted((r for r in rows_q if r['passed']), key=lambda r: r['trial'])
    fails = sorted((r for r in rows_q if not r['passed']), key=lambda r: r['trial'])
    strong = [r for r in fails if a.weak_timeouts == 'keep' or not weak_timeout(r)]
    weak = [r for r in fails if a.weak_timeouts != 'keep' and weak_timeout(r)]
    if len(passes) < a.passes:
        raise SystemExit(f'only {len(passes)} passes after the quarantine, {a.passes} asked')
    kept_p = rng.sample(passes, a.passes)
    if len(strong) >= a.failures:
        kept_f = rng.sample(strong, a.failures)
    elif a.weak_timeouts == 'fill' and len(strong) + len(weak) >= a.failures:
        kept_f = list(strong) + [dict(r, weak_timeout_fill=True) for r in rng.sample(weak, a.failures - len(strong))]
    else:
        raise SystemExit(f'only {len(strong)} failures (+{len(weak)} weak timeouts) after the quarantine, '
                         f'{a.failures} asked with --weak-timeouts {a.weak_timeouts}')
    with open(a.out, 'w') as f:
        for r in kept_p + kept_f:
            f.write(json.dumps(r) + '\n')
    print(json.dumps(dict(
        manifest=a.manifest, quarantined_tasks=len(quarantine), rows_in=len(rows),
        dropped_quarantine=len(rows) - len(rows_q), pass_pool=len(passes), failure_pool=len(strong),
        weak_timeout_pool=len(weak), weak_timeouts=a.weak_timeouts, seed=a.seed,
        kept_passes=len(kept_p), kept_failures=len(kept_f),
        weak_timeout_fill=sum(bool(r.get('weak_timeout_fill')) for r in kept_f),
        failure_causes=dict(collections.Counter(r['cause'] for r in kept_f)),
        kept_timeouts_stalled=dict(collections.Counter(str(r.get('stalled')) for r in kept_f if r['cause'] == 'timeout')),
        out=a.out), indent=1))


if __name__ == '__main__':
    main()
