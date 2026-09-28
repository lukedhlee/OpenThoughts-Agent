#!/usr/bin/env python3
"""Rendered relay rows (render.py / render_think_limit.py jsonl) -> the one-shard parquet the Snowball SFT chain reads.

    python relay_rows_to_parquet.py <final_rendered*.jsonl> <out dir> [--max-tokens 65536]

The chain's stages relay_relay / relay_qwen (marin lukedhlee/vista-snowball-sft, grug_datakit_chat.
SnowballPrerenderedChatFormat) copy each row's `ids` and `loss` into the cache as input_ids / assistant_masks; nothing
is re-rendered or re-tokenized, so the trainer sees exactly these ids and this per-token mask (loss[t] = 1: token t is
a trained target). Columns written: id (the episode's session id), ids (int32), loss (int8), n_tokens. Rows keep the
jsonl's order. Writes <out>/train-00000-of-00001.parquet and <out>/parquet.list (one line, the absolute shard path),
and prints a census.

Refuses (exit 1, nothing written) on any row that the chain would refuse or silently change: ids/loss length
mismatch, n_tokens disagreeing, fits false, more than --max-tokens ids (the packer would cut it), loss values other
than 0/1, a first token other than BOS 128000 or a trained BOS, no trained token, a duplicate session id.
Login-node safe: one process, Arrow's thread pools pinned to one thread.
"""
import argparse
import json
import os
import sys

os.environ.setdefault('OMP_NUM_THREADS', '1')
import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

BOS = 128000
VOCAB = 128256


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('rows')
    ap.add_argument('out_dir')
    ap.add_argument('--max-tokens', type=int, default=65536)
    ap.add_argument('--row-group', type=int, default=64)
    a = ap.parse_args()
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    ids_col, loss_col, sid_col, n_col, bad, seen = [], [], [], [], [], set()
    trained = 0
    with open(a.rows) as f:
        for k, line in enumerate(f):
            r = json.loads(line)
            ids, loss, sid = r['ids'], r['loss'], r['sid']
            why = None
            if len(ids) != len(loss) or len(ids) != r.get('n_tokens', len(ids)):
                why = 'length mismatch'
            elif r.get('fits') is False or len(ids) > a.max_tokens:
                why = f'{len(ids)} tokens > {a.max_tokens}'
            elif any(x not in (0, 1) for x in loss):
                why = 'loss not 0/1'
            elif ids[0] != BOS or loss[0] != 0:
                why = 'no untrained BOS first'
            elif not any(loss):
                why = 'no trained token'
            elif min(ids) < 0 or max(ids) >= VOCAB:
                why = 'id outside the vocabulary'
            elif sid in seen:
                why = 'duplicate sid'
            if why:
                bad.append(dict(line=k, sid=sid, why=why))
                continue
            seen.add(sid)
            sid_col.append(sid)
            ids_col.append(ids)
            loss_col.append(loss)
            n_col.append(len(ids))
            trained += sum(loss)
    if bad:
        print(json.dumps(dict(refused=len(bad), first=bad[:20]), indent=1))
        sys.exit(1)
    os.makedirs(a.out_dir, exist_ok=True)
    shard = os.path.abspath(os.path.join(a.out_dir, 'train-00000-of-00001.parquet'))
    table = pa.table({
        'id': pa.array(sid_col, pa.string()),
        'ids': pa.array(ids_col, pa.list_(pa.int32())),
        'loss': pa.array(loss_col, pa.list_(pa.int8())),
        'n_tokens': pa.array(n_col, pa.int32()),
    })
    pq.write_table(table, shard, row_group_size=a.row_group)
    with open(os.path.join(a.out_dir, 'parquet.list'), 'w') as f:
        f.write(shard + '\n')
    n = sorted(n_col)
    print(json.dumps(dict(rows=len(n), tokens=sum(n), trained_tokens=trained, max_tokens=n[-1], p50=n[len(n) // 2],
                          p90=n[int(0.9 * (len(n) - 1))], shard=shard), indent=1))


if __name__ == '__main__':
    main()
