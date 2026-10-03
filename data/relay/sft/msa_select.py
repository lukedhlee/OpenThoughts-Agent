#!/usr/bin/env python3
"""SFT arms from MSA relay rows (render_msa.py output of the relay_repair and control arms), with the Terminus-2 arms'
rules (build_hz_arms.py / build_matched_arms.py):

Eligibility, identical for both arms: the verifier ran, not a weak timeout, no leak / hunt / canary tag, fits in
65,536 tokens, at least one trained token (the parquet converter refuses a row without one).
  msaall   every task's eligible relay rows, at most 2 drawn at random (H4's cap);
  msarel / msaqwen  matched: per task the relay draw (p passes, f failures) is the target; the Qwen-alone arm must
           supply the same outcomes on that task, else the task's slots are cut to what both arms have; then each
           arm draws its rows of each outcome uniformly. Same task slots and pass / fail mix in both.

    python msa_select.py --relay <rows.jsonl> [...] --control <rows.jsonl> [...] --out-dir <dir> [--seed 0]
"""
import argparse
import collections
import json
import os
import random

MAX_TOKENS = 65_536
PER_TASK = 2


def eligible(r):
    return (r.get('verifier_ran') and not r.get('weak_timeout') and not (r.get('leak') or r.get('hunt') or r.get('canary'))
            and r.get('fits') and r['n_tokens'] <= MAX_TOKENS and r.get('trained_tokens', 0) > 0)


def by_task(paths):
    out = collections.defaultdict(list)
    seen = 0
    for p in paths:
        for line in open(p):
            r = json.loads(line)
            seen += 1
            if eligible(r):
                out[r['task']].append(r)
    return out, seen


def write(rows, path):
    with open(path, 'w') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')


def summary(rows):
    return dict(rows=len(rows), tasks=len({r['task'] for r in rows}), passes=sum(bool(r['passed']) for r in rows),
                trained_tokens=sum(r['trained_tokens'] for r in rows), tokens=sum(r['n_tokens'] for r in rows))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--relay', nargs='+', required=True)
    ap.add_argument('--control', nargs='+', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    relay, n_relay = by_task(a.relay)
    control, n_control = by_task(a.control)

    msaall, msarel, msaqwen = [], [], []
    for task in sorted(relay):
        draw = rng.sample(relay[task], min(PER_TASK, len(relay[task])))
        msaall += draw
        want = collections.Counter(bool(r['passed']) for r in draw)
        have = collections.Counter(bool(r['passed']) for r in control.get(task, []))
        for outcome in (True, False):
            k = min(want[outcome], have[outcome])
            if not k:
                continue
            msarel += rng.sample([r for r in draw if bool(r['passed']) == outcome], k)
            msaqwen += rng.sample([r for r in control[task] if bool(r['passed']) == outcome], k)

    os.makedirs(a.out_dir, exist_ok=True)
    for name, rows in (('msaall', msaall), ('msarel', msarel), ('msaqwen', msaqwen)):
        write(rows, os.path.join(a.out_dir, f'{name}_rows.jsonl'))
    report = dict(seed=a.seed, relay_rows_seen=n_relay, relay_eligible=sum(map(len, relay.values())),
                  control_rows_seen=n_control, control_eligible=sum(map(len, control.values())),
                  msaall=summary(msaall), msarel=summary(msarel), msaqwen=summary(msaqwen))
    json.dump(report, open(os.path.join(a.out_dir, 'select.json'), 'w'), indent=1)
    print(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
