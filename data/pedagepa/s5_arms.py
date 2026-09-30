#!/usr/bin/env python3
"""PedaGEPA stage 5: rows for arms pgc1 and pgp from the judged recovery pool (s5_pool.py + Opus readers).

R   = pool episodes an Opus reader labelled mistake_real=yes AND recovery=genuine (rubric s5_recovery_rubric.md v1),
      at most 2 per task (seeded).
C1  = |R| pool episodes drawn at random from every judged pool episode (labels ignored), at most 2 per task (seeded):
      the size-matched unjudged control.
pgp  rows = pgc0 rows + R rows;  pgc1 rows = pgc0 rows + C1 rows.
Checks: every trained span of every added row is free of the leak pattern (the pool build masked them); the gate sample
(20 random R episodes, seeded) is written for a fresh reader.

    python s5_arms.py --pool <pool dir> --pgc0 <pgc0_rows.jsonl> --tokenizer-dir <09-21> --out-dir <rows dir>
"""
import argparse
import collections
import glob
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from s5_pool import LEAK, spans  # noqa: E402

SEED = 20260930


def cap(items, rng, per_task=2):
    by = collections.defaultdict(list)
    for x in items:
        by[x['task']].append(x)
    out = []
    for t in sorted(by):
        rng.shuffle(by[t])
        out += by[t][:per_task]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--pool', required=True)
    ap.add_argument('--pgc0', required=True)
    ap.add_argument('--tokenizer-dir', required=True)
    ap.add_argument('--out-dir', required=True)
    a = ap.parse_args()
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.path.join(a.tokenizer_dir, 'tokenizer.json'))
    man = {json.loads(l)['sid']: json.loads(l) for l in open(os.path.join(a.pool, 'pool_manifest.jsonl'))}
    lab = {}
    for f in sorted(glob.glob(os.path.join(a.pool, 'judge', 'out', 'batch*.jsonl'))):
        for l in open(f):
            if l.strip():
                j = json.loads(l)
                if j['sid'] in man:
                    lab[j['sid']] = j
    judged = [dict(man[s], **{'label': lab[s]}) for s in lab]
    rng = random.Random(SEED)
    R = cap([x for x in judged if x['label']['mistake_real'] == 'yes' and x['label']['recovery'] == 'genuine'], rng)
    rng = random.Random(SEED + 1)
    pool_c = cap(judged, rng)
    rng.shuffle(pool_c)
    C1 = pool_c[:len(R)]
    want = {'pgp': {x['sid'] for x in R}, 'pgc1': {x['sid'] for x in C1}}
    rep = dict(judged=len(judged), labels=collections.Counter(f"{x['label']['mistake_real']}/{x['label']['recovery']}"
                                                              for x in judged),
               R=len(R), R_pass=sum(x['passed'] for x in R), R_tasks=len({x['task'] for x in R}),
               C1=len(C1), C1_pass=sum(x['passed'] for x in C1), C1_in_R=len(want['pgp'] & want['pgc1']),
               C1_labels=collections.Counter(f"{x['label']['mistake_real']}/{x['label']['recovery']}" for x in C1))
    os.makedirs(a.out_dir, exist_ok=True)
    base = open(a.pgc0).readlines()
    outs = {k: open(os.path.join(a.out_dir, f'{k}_rows.jsonl'), 'w') for k in want}
    for fo in outs.values():
        fo.writelines(base)
    leaks = collections.Counter()
    tokens = collections.Counter()
    for line in open(os.path.join(a.pool, 'pool_rows.jsonl')):
        i = line.find('"ids"')
        sid = json.loads(line[:i].rstrip(', ') + '}')['sid']
        ks = [k for k, w in want.items() if sid in w]
        if not ks:
            continue
        r = json.loads(line)
        bad = any(LEAK.search(tok.decode(r['ids'][s:e], skip_special_tokens=False)) for s, e in spans(r['loss']))
        for k in ks:
            leaks[k] += bad
            tokens[k] += sum(r['loss'])
            outs[k].write(line)
    for fo in outs.values():
        fo.close()
    rep['added_trained_tokens'] = dict(tokens)
    rep['added_rows_with_leak_in_trained_text'] = dict(leaks)
    rng = random.Random(SEED + 2)
    gate = rng.sample(R, min(20, len(R)))
    open(os.path.join(a.out_dir, 'gate_sample.txt'), 'w').write(
        '\n'.join(os.path.join(a.pool, 'views', x['sid'] + '.txt') for x in gate) + '\n')
    json.dump(dict(R=[x['sid'] for x in R], C1=[x['sid'] for x in C1]), open(os.path.join(a.out_dir, 'arm_sids.json'), 'w'))
    json.dump(rep, open(os.path.join(a.out_dir, 'arms_report.json'), 'w'), indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == '__main__':
    main()
