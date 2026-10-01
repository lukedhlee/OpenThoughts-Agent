#!/usr/bin/env python3
"""Split export_relay_dataset.py's output into the three tabs of the shared CalibForge relay dataset (Luke 2026-10-01):

  used             the relay rows of our best model H8 (used_in contains H8): every row that passed the QA filters
                   and the sampling rules (at most 2 per task; Jupiter: arm A's 1:1 set);
  unused_clean     passed every QA filter, fits 65,536 tokens and has a trained token, but was left out by the sampling
                   rules (mostly the 2-per-task cap);
  unused_filtered  dropped by a filter: leak / hunt tag, over 65,536 tokens, or nothing to train.

Rows tagged `canary` (trajectories that contain Terminal-Bench's canary GUID, because some CalibForge starter files carry
its header) are left out of every split by default: the GUID would make the public dataset trip canary-based
contamination checks. Also writes kimi_trials.txt: the trial names of the 4,392 Kimi rows H8 / H9 trained on, from
open-athena/Kimi-2.5-swesmith-sandboxes-with_tests-oracle_verified_120s-maxeps-32k.

    python split_relay_dataset.py --export <export dir> --h8-rows <h8_rows.jsonl> --out <dir> [--keep-canary]
"""
import argparse
import collections
import glob
import json
import os


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--export', required=True)
    ap.add_argument('--h8-rows', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--keep-canary', action='store_true')
    ap.add_argument('--shard-rows', type=int, default=1000)
    a = ap.parse_args()
    import pyarrow as pa
    import pyarrow.parquet as pq
    os.makedirs(os.path.join(a.out, 'data'), exist_ok=True)
    bufs, shards, stats = collections.defaultdict(list), collections.Counter(), collections.Counter()

    def flush(split):
        if bufs[split]:
            pq.write_table(pa.Table.from_pylist(bufs[split]),
                           os.path.join(a.out, 'data', f'{split}-{shards[split]:05d}.parquet'), compression='zstd')
            shards[split] += 1
            bufs[split] = []
    for f in sorted(glob.glob(os.path.join(a.export, 'data', '*.parquet'))):
        for r in pq.read_table(f).to_pylist():
            if r['canary'] and not a.keep_canary:
                stats['dropped_canary'] += 1
                continue
            if 'H8' in r['used_in']:
                split = 'used'
            elif r['leak'] or r['hunt'] or r['canary'] or r['n_tokens'] > 65536 or not r['trained_tokens']:
                split = 'unused_filtered'
            else:
                split = 'unused_clean'
            bufs[split].append(r)
            stats[f"{split}/{r['source']}/{'pass' if r['passed'] else 'fail'}"] += 1
            if len(bufs[split]) >= a.shard_rows:
                flush(split)
    for s in list(bufs):
        flush(s)
    trials = []
    for line in open(a.h8_rows):
        i = line.find('"ids"')
        sid = json.loads(line[:i].rstrip(', ') + '}')['sid']
        if sid.startswith('kimi:'):
            trials.append(sid.rsplit(':', 1)[1])
    with open(os.path.join(a.out, 'kimi_trials.txt'), 'w') as fo:
        fo.write('\n'.join(sorted(trials)) + '\n')
    stats['kimi_trials'] = len(trials)
    json.dump(dict(sorted(stats.items())), open(os.path.join(a.out, 'splits.json'), 'w'), indent=1)
    print(json.dumps(dict(sorted(stats.items())), indent=1))


if __name__ == '__main__':
    main()
