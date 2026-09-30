#!/usr/bin/env python3
"""Rows for the relay_hzrelay (H1) and relay_hzpass (H4) arms: arm A's 09-28 rows + the Horizon relay rows (09-29, student
= arm A), both filtered the same way (Luke 2026-09-30): drop grader hunting, benchmark canary and verify-note leak rows.

Horizon part: rendered candidates (render_think_limit.py, 09-28 settings) that fit 65,536 tokens, tagged by
hz_pool_census.py from the router's records (leak / hunt / canary), dropped when tagged; then at most 2 rows per task
(seeded). hzrelay keeps N passes + N failures (N = min, seeded: 09-28's final_v2 rule); hzpass keeps every capped row.
Arm A part (raw episodes are on Jupiter): the same three tags from the rendered rows themselves: leak = the regex in any
TRAINED span (teacher text), hunt = the regex in a "keystrokes" value of any assistant turn, canary = anywhere in the row;
tagged rows dropped, then the larger class trimmed at random so the part stays 1:1. Both arms get the same arm A part.

    python build_hz_arms.py --census census.jsonl --rendered <dir with *.rendered.jsonl> --a-rows <arm A rows> \
        --a-manifest <final_v2_manifest.jsonl> --tokenizer-dir <09-21> --out-dir <dir>
"""
import argparse
import collections
import glob
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hz_pool_census import CANARY, HUNT, LEAK  # noqa: E402

KEYS = re.compile(r'"keystrokes"\s*:\s*"((?:[^"\\]|\\.)*)"')
SEED = 20260930


def tag_rendered(row, tok):
    ids, loss = row['ids'], row['loss']
    text = tok.decode(ids, skip_special_tokens=False)
    trained, i = [], 0
    while i < len(loss):
        if loss[i]:
            j = i
            while j < len(loss) and loss[j]:
                j += 1
            trained.append(tok.decode(ids[i:j], skip_special_tokens=False))
            i = j
        else:
            i += 1
    leak = any(LEAK.search(s) for s in trained)
    hunt = False
    for seg in text.split('<|start_header_id|>assistant<|end_header_id|>')[1:]:
        seg = seg.split('<|eot_id|>')[0]
        if any(HUNT.search(k) for k in KEYS.findall(seg)):
            hunt = True
            break
    return dict(leak=leak, hunt=hunt, canary=bool(CANARY.search(text)))


def cap_per_task(rows, rng, per_task=2):
    by = collections.defaultdict(list)
    for r in rows:
        by[r['task']].append(r)
    out = []
    for t in sorted(by):
        rng.shuffle(by[t])
        out += by[t][:per_task]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--census', required=True)
    ap.add_argument('--rendered', required=True)
    ap.add_argument('--a-rows', required=True)
    ap.add_argument('--a-manifest', required=True)
    ap.add_argument('--tokenizer-dir', required=True)
    ap.add_argument('--out-dir', required=True)
    a = ap.parse_args()
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.path.join(a.tokenizer_dir, 'tokenizer.json'))
    os.makedirs(a.out_dir, exist_ok=True)
    rep = {}

    # arm A part
    passed = {json.loads(l)['sid']: json.loads(l)['passed'] for l in open(a.a_manifest)}
    a_keep, a_tags = {True: [], False: []}, collections.Counter()
    for line in open(a.a_rows):
        r = json.loads(line)
        t = tag_rendered(r, tok)
        a_tags.update(k for k, v in t.items() if v)
        if not any(t.values()):
            a_keep[passed[r['sid']]].append(line)
    rng = random.Random(SEED)
    n = min(len(a_keep[True]), len(a_keep[False]))
    for k in (True, False):
        rng.shuffle(a_keep[k])
        a_keep[k] = a_keep[k][:n]
    a_lines = a_keep[True] + a_keep[False]
    rep['arm_a'] = dict(rows_in=len(passed), tagged=dict(a_tags), kept_pass=n, kept_fail=n)

    # Horizon part
    cen = {}
    for l in open(a.census):
        c = json.loads(l)
        cen[c['sid']] = c
    pool, st = [], collections.Counter()
    for f in sorted(glob.glob(os.path.join(a.rendered, '*.rendered.jsonl'))):
        for line in open(f):
            i = line.find('"ids"')
            h = json.loads(line[:i].rstrip(', ') + '}')
            c = cen.get(h['sid'])
            st['rendered'] += 1
            if c is None:
                st['no_census'] += 1
                continue
            if not h['fits']:
                st['over_64k'] += 1
                continue
            j = line.find('"loss": [')
            if '1' not in line[j + 9:line.find(']', j)]:   # every turn masked (e.g. one autofixed teacher turn): nothing to train
                st['no_trained_token'] += 1
                continue
            tags = dict(leak=c['leak_turns'] > 0, hunt=c['hunt_turns'] > 0, canary=c['canary'])
            st.update(f'tag_{k}' for k, v in tags.items() if v)
            if any(tags.values()):
                st['dropped'] += 1
                continue
            pool.append(dict(sid=h['sid'], task=c['task'], passed=c['passed'], file=f))
    rng = random.Random(SEED)
    capped = cap_per_task(pool, rng)
    P = [r for r in capped if r['passed']]
    F = [r for r in capped if not r['passed']]
    n = min(len(P), len(F))
    rng.shuffle(P)
    rng.shuffle(F)
    arms = dict(hzrelay={r['sid'] for r in P[:n] + F[:n]}, hzpass={r['sid'] for r in capped})
    rep['horizon'] = dict(st, pool=len(pool), capped_pass=len(P), capped_fail=len(F),
                          hzrelay=dict(pass_=n, fail=n, tasks=len({r['task'] for r in P[:n] + F[:n]})),
                          hzpass=dict(pass_=len(P), fail=len(F), tasks=len({r['task'] for r in capped})))
    outs = {k: open(os.path.join(a.out_dir, f'{k}_rows.jsonl'), 'w') for k in arms}
    for fo in outs.values():
        fo.writelines(a_lines)
    for f in sorted({r['file'] for r in pool}):
        for line in open(f):
            i = line.find('"ids"')
            sid = json.loads(line[:i].rstrip(', ') + '}')['sid']
            for k, want in arms.items():
                if sid in want:
                    outs[k].write(line)
    for fo in outs.values():
        fo.close()
    json.dump(rep, open(os.path.join(a.out_dir, 'build.json'), 'w'), indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == '__main__':
    main()
