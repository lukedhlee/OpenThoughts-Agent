#!/usr/bin/env python3
"""Steps per real pass over a relay SFT shard at the chain's packing, for SNOWBALL_EPOCH_STEPS.

    <marin env python> pack_epoch_steps.py <train-00000-of-00001.parquet> [--seq-len 65536] [--batch 16]

The chain derives an epoch as ceil(tokens / (seq_len x batch)), which assumes full packs. Relay rows are long (p50
22k-36k tokens), so the greedy packer fills only ~75 % of each 65,536 sequence and a token-epoch covers ~0.75 of the
packs: "3 epochs" was ~2.2-2.3 real passes, and not the same number in the two arms. This counts the packs with
Levanter's own packer (levanter.data.packing.pack_documents, 64 segments, "left"; rows in shard order, which is the
cache's order for a one-shard cache) and prints the steps per pass, ceil(packs / batch), plus the token-derived value.
Login-node safe (JAX on CPU, one thread).
"""
import argparse
import json
import math
import os

os.environ.setdefault('JAX_PLATFORMS', 'cpu')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('XLA_FLAGS', '--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1')

import numpy as np  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
from levanter.data.packing import pack_documents  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('shard')
    ap.add_argument('--seq-len', type=int, default=65536)
    ap.add_argument('--batch', type=int, default=16)
    a = ap.parse_args()
    lens = np.asarray(pq.read_table(a.shard, columns=['n_tokens'])['n_tokens'].to_numpy(), dtype=np.int64)
    packs = len(pack_documents(lens, a.seq_len, max_segments_per_example=64, slice_strategy='left'))
    tokens = int(lens.sum())
    print(json.dumps(dict(rows=len(lens), tokens=tokens, packs=packs, fill=round(tokens / (packs * a.seq_len), 3),
                          pack_epoch_steps=math.ceil(packs / a.batch),
                          token_epoch_steps=math.ceil(tokens / (a.seq_len * a.batch)))))


if __name__ == '__main__':
    main()
