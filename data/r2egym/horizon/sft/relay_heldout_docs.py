#!/usr/bin/env python3
"""A held-out set in the relay SFT arms' own format, for scoring exports with batch0_nll.py (score mode).

Rendered relay candidates (rendered.jsonl: every kept candidate of a relay run, ids + per-token loss) whose episode is in
neither arm's training rows (final_v2 manifests), N per run drawn with a fixed seed, rows over --max-tokens skipped. The
tasks can overlap the training tasks (rendered.jsonl carries no task id), so this measures fit to the arms' distribution
on unseen episodes, which is what a Horizon-vs-Jupiter comparison of the same arm needs. One JSON line per document:
ids, w (w[i] = loss[i + 1]: position i scores token i + 1, batch0_nll.py's convention), next = null.

    python relay_heldout_docs.py --run <relay run dir> [--run ...] --exclude <final_v2_manifest.jsonl> [...] --n 60 --out docs.jsonl
"""
import argparse
import json
import random


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run', action='append', required=True, help='dir with rendered.jsonl')
    ap.add_argument('--exclude', action='append', required=True, help='manifest jsonl whose sids were trained on')
    ap.add_argument('--n', type=int, default=60, help='documents per run')
    ap.add_argument('--max-tokens', type=int, default=65536)
    ap.add_argument('--seed', type=int, default=20260930)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    trained = set()
    for m in a.exclude:
        trained |= {json.loads(l)['sid'] for l in open(m)}
    census = {}
    with open(a.out, 'w') as out:
        for k, run in enumerate(a.run):
            path = f'{run}/rendered.jsonl'
            cand = []
            for i, line in enumerate(open(path)):   # pass 1: eligible line numbers (sid + length only)
                head = json.loads(line[:line.index('"ids"') - 2] + '}') if '"ids"' in line[:300] else json.loads(line)
                if head['sid'] not in trained and head.get('fits', True) and head['n_tokens'] <= a.max_tokens:
                    cand.append(i)
            pick = set(random.Random(a.seed + k).sample(cand, min(a.n, len(cand))))
            w_total = 0
            for i, line in enumerate(open(path)):   # pass 2: the picked rows
                if i not in pick:
                    continue
                r = json.loads(line)
                ids, loss = r['ids'], r['loss']
                assert len(ids) == len(loss)
                w = loss[1:] + [0]
                w_total += sum(w)
                out.write(json.dumps({'pack': k, 'sid': r['sid'], 'ids': ids, 'w': w, 'next': None}) + '\n')
            census[run] = dict(rendered=i + 1, eligible=len(cand), picked=len(pick), trained_tokens=w_total)
    print(json.dumps(dict(out=a.out, seed=a.seed, excluded_sids=len(trained), runs=census), indent=1))


if __name__ == '__main__':
    main()
