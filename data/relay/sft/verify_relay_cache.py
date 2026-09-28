#!/usr/bin/env python3
"""Prove the SFT chain trains on exactly the rendered relay rows: every row's token ids and loss mask, as the trainer
draws them, equal the rendered ones.

    SNOWBALL_SEQ_LEN=65536 SNOWBALL_BATCH=16 SNOWBALL_DEVICES=16 JAX_PLATFORMS=cpu PYTHONPATH=<marin checkout> \
      <marin env python> verify_relay_cache.py <stage relay_relay|relay_qwen> <rows.jsonl> <parquet dir> <cache dir> \
      <09-21 tokenizer dir> [--prepare]

--prepare builds <cache dir> first with the chain's own commands (vista_snowball_chat prepare-data + write-provenance,
as jupiter_r2egym_prep.sbatch runs them); without it an existing cache (e.g. one the chain's prep job built, copied
here) is checked. Then: provenance and layout validation as preflight does, and every packed example of the trainer's
data config at 65,536 (snowball_chat_data_config(...).train_sets, the dataset the trainer's mixture draws from) is split
back into rows by segment id. For each row: the segment's ids must equal the row's ids, and the trainer's loss weight,
shifted back by one (ChatDataset weights position t by mask[t+1]), must equal the row's loss; the last position of a row
must carry no weight (nothing trained across a row boundary); padding carries no weight; every row appears exactly once.
CPU only; run it off the login nodes (it starts JAX and zephyr workers).
"""
import collections
import json
import subprocess
import sys

import jax
import numpy as np
from haliax import Axis

MOD = 'experiments.june_tpu_67b_a2b.moe.vista_snowball_chat'


def main():
    stage, rows_path, pq_dir, cache, tokdir = sys.argv[1:6]
    from experiments.june_tpu_67b_a2b.moe import vista_snowball_chat as v
    spec = v.STAGES[stage]
    if '--prepare' in sys.argv[6:]:
        run = lambda *a: subprocess.run([sys.executable, '-m', MOD, *a], check=True, capture_output=True, text=True).stdout  # noqa: E731
        run('prepare-data', '--stage', stage, '--parquet-list', f'{pq_dir}/parquet.list', '--expect-files', '1',
            '--cache-path', cache, '--tokenizer-path', tokdir)
        run('write-provenance', '--cache-path', cache, '--stage', stage, '--dataset-id', spec.dataset_id,
            '--dataset-revision', spec.dataset_revision, '--parquet-list', f'{pq_dir}/parquet.list', '--expect-files',
            '1', '--tokenizer-path', tokdir)
    rec = v.validate_cache_provenance(cache, stage, tokdir)
    tokens, examples, shards = v.validate_chat_cache_layout(cache, expect_tokens=None, expect_examples=None)
    rows = [json.loads(line) for line in open(rows_path)]
    by_ids = {tuple(r['ids']): r for r in rows}
    c = collections.Counter(rows=len(rows), distinct_rows=len(by_ids), cache_examples=examples, cache_tokens=tokens,
                            row_tokens=sum(len(r['ids']) for r in rows))
    L = v.SNOWBALL_CHAT_SEQUENCE_LENGTH
    ds = v.snowball_chat_data_config(cache_path=cache, tokenizer_path=tokdir, stage=stage).train_sets(
        Axis('position', L), key=jax.random.PRNGKey(v.SNOWBALL_CHAT_SEED), initial_batch_size=v.SNOWBALL_CHAT_BATCH_SIZE)
    d = ds[spec.component].as_sync_dataset()
    seen = collections.Counter()
    for i in range(len(d)):
        ex = d[i]
        tok, lw = np.asarray(ex.tokens), np.asarray(ex.loss_weight)
        seg = np.asarray(ex.attn_mask.segment_ids[0])
        c['packs'] += 1
        c['pad_positions_with_weight'] += int((lw[seg < 0] != 0).sum())
        j = 0
        while j < L:
            if seg[j] < 0:
                j += 1
                continue
            e = j
            while e < L and seg[e] == seg[j]:
                e += 1
            r = by_ids.get(tuple(tok[j:e].tolist()))
            if r is None:
                c['segments_matching_no_row'] += 1
            else:
                seen[r['sid']] += 1
                target = np.zeros(e - j, dtype=np.float32)
                target[1:] = lw[j:e - 1]
                c['rows_ids_equal'] += 1
                c['rows_loss_equal'] += bool(np.array_equal(target, np.asarray(r['loss'], dtype=np.float32)))
                c['row_end_weight_nonzero'] += bool(lw[e - 1] != 0)
                c['trained_tokens_rows'] += int(sum(r['loss']))
                c['trained_tokens_trainer'] += int(lw[j:e].sum())
            j = e
    c['rows_never_seen'] = sum(1 for r in rows if seen[r['sid']] == 0)
    c['rows_seen_twice'] = sum(1 for k in seen.values() if k > 1)
    ok = (c['rows_ids_equal'] == c['rows_loss_equal'] == len(rows) and not c['rows_never_seen'] and not c['rows_seen_twice']
          and not c['segments_matching_no_row'] and not c['row_end_weight_nonzero'] and not c['pad_positions_with_weight']
          and c['trained_tokens_rows'] == c['trained_tokens_trainer'] and c['cache_tokens'] == c['row_tokens'])
    print(json.dumps(dict(stage=stage, seq_len=L, format=rec['format'], verdict='EQUAL' if ok else 'MISMATCH', **c), indent=1))
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
