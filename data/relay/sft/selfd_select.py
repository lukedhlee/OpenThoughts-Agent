#!/usr/bin/env python3
"""Self-distillation rows: the student_only arm's own MSA episodes (render_msa.py --arm student_only) -> SFT rows.

Per task, the passed episode that ends on a trained submit (msa_select.ends_submit) with the SHORTEST eval view (tokens
the no-refeed eval held at the end: every <|start_think|>..<|end_think|> span removed); verifier ran, no weak timeout,
no leak / hunt / canary tag. The rendered rows keep every earlier turn's reasoning (cut to the cap), which the no-refeed
eval never re-sends, so most of them exceed the 65,536-token row limit; each kept episode is re-rendered with
nr_window.windows (think-stripped history, windows of --k trained turns) and windows over the limit are dropped.

    python selfd_select.py --rows <rows.jsonl> [...] --out <rows_out.jsonl> [--k 8]
Writes <out> and <out>.json (counts).
"""
import argparse
import collections
import json
import os
import random

import msa_select
import nr_window

MAX_TOKENS = 65536


def eval_len(ids):
    think, inside, s = 0, False, 0
    for i, t in enumerate(ids):
        if t == nr_window.ST:
            inside, s = True, i
        elif t == nr_window.ET and inside:
            think += i - s + 1
            inside = False
    return len(ids) - think


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--rows', nargs='+', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--k', type=int, default=8)
    a = ap.parse_args()
    from tokenizers import Tokenizer
    msa_select.TOKENIZER = Tokenizer.from_file(os.path.join(msa_select.TOKDIR, 'tokenizer.json'))
    best, n = {}, collections.Counter()
    for p in a.rows:
        for line in open(p):
            r = json.loads(line)
            n['rows'] += 1
            if not r.get('passed'):
                continue
            n['passed'] += 1
            if not (r.get('verifier_ran') and not r.get('weak_timeout') and not (r.get('leak') or r.get('hunt') or r.get('canary'))
                    and r.get('trained_tokens', 0) > 0):
                n['tagged'] += 1
                continue
            if not msa_select.ends_submit(r):
                n['no_submit_end'] += 1
                continue
            ev = eval_len(r['ids'])
            if r['task'] not in best or ev < best[r['task']][0]:
                best[r['task']] = (ev, r)
    n['tasks'] = len(best)
    rng = random.Random(0)
    evs, out = [], []
    for task in sorted(best):
        ev, r = best[task]
        evs.append(ev)
        for w in nr_window.windows(r, a.k, 'all', rng):
            if w['n_tokens'] > MAX_TOKENS:
                n['windows_over_limit'] += 1
                continue
            out.append(dict(w, eval_len=ev, selfd=True))
    with open(a.out, 'w') as f:
        for w in out:
            f.write(json.dumps(w) + '\n')
    evs.sort()
    rep = dict(n, windows=len(out), trained_tokens=sum(w['trained_tokens'] for w in out),
               tokens=sum(w['n_tokens'] for w in out), tasks_with_windows=len({w['task'] for w in out}),
               eval_len_p50=evs[len(evs) // 2] if evs else None, eval_len_p90=evs[9 * len(evs) // 10] if evs else None, k=a.k)
    json.dump(rep, open(a.out + '.json', 'w'), indent=1)
    print(json.dumps(rep))


if __name__ == '__main__':
    main()
