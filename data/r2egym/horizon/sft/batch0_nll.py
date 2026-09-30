#!/usr/bin/env python3
"""Score the SFT run's first training batch with vLLM, to place the JAX trainer's step-0 loss.

The Horizon and Jupiter SFT runs of the same arm log different step-0 losses (0.635 vs 0.628) on the same batch at the
same init with lr 0. vLLM's held-out NLL of the base is identical on both clusters (0.586), so vLLM is a common
yardstick: scoring batch 0 with it tells which trainer forward (GB200 or GH200) sits off.

  dump   marin env, CPU (login node is fine): rebuild batch 0 exactly as experiments/june_tpu_67b_a2b/moe/train.py does
         (data key = split(PRNGKey(seed=0))[0], build_train_dataset, indices 0..63), split every pack into its documents
         by segment id, and write one JSON line per document: ids, loss weights (weight[i] scores token i+1) and, when
         the document's last position carries weight, the next token it predicts.
  score  against a running vLLM server (heldout_nll.sbatch starts one and passes the same arguments; --parquet is the
         dump): prompt_logprobs per document, weighted mean NLL over the batch = the trainer's loss definition.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import sys
import time


def dump(a):
    import jax
    import numpy as np
    from levanter.schedule import BatchSchedule

    from experiments.june_tpu_67b_a2b.moe.train import build_train_dataset
    from experiments.june_tpu_67b_a2b.moe.vista_snowball_chat import snowball_chat_data_config
    from experiments.june_tpu_67b_a2b.moe.snowball_chat_recipe import (
        SNOWBALL_CHAT_BATCH_SIZE, SNOWBALL_CHAT_SEED, SNOWBALL_CHAT_SEQUENCE_LENGTH)

    cfg = snowball_chat_data_config(cache_path=a.cache, tokenizer_path=a.tokenizer, stage=a.stage)
    data_key, _ = jax.random.split(jax.random.PRNGKey(SNOWBALL_CHAT_SEED), 2)
    ds = build_train_dataset(cfg, max_seq_len=SNOWBALL_CHAT_SEQUENCE_LENGTH,
                             batch_schedule=BatchSchedule(SNOWBALL_CHAT_BATCH_SIZE), key=data_key)
    sync = ds.as_sync_dataset()
    start = a.step * SNOWBALL_CHAT_BATCH_SIZE
    exs = sync.get_batch(list(range(start, start + SNOWBALL_CHAT_BATCH_SIZE)))
    n_docs = w_total = cross = 0
    with open(a.out, "w") as f:
        for p, ex in enumerate(exs):
            tok = np.asarray(ex.tokens)
            w = np.asarray(ex.loss_weight, dtype=np.float64)
            seg = ex.attn_mask.segment_ids
            seg = np.asarray(seg[0] if isinstance(seg, (tuple, list)) else seg) if seg is not None else np.zeros_like(tok)
            bounds = [0] + [i for i in range(1, len(tok)) if seg[i] != seg[i - 1]] + [len(tok)]
            for s, e in zip(bounds[:-1], bounds[1:]):
                if seg[s] < 0:
                    assert w[s:e].sum() == 0, "weight on padding"
                    continue
                ids = tok[s:e].tolist(); ww = w[s:e].tolist()
                nxt = int(tok[e]) if (ww[-1] > 0 and e < len(tok)) else None
                cross += nxt is not None
                w_total += sum(ww); n_docs += 1
                f.write(json.dumps({"pack": p, "start": s, "ids": ids, "w": ww, "next": nxt}) + "\n")
    print(f"batch {a.step}: {len(exs)} packs, {n_docs} documents, loss weight {w_total:.0f}, "
          f"documents whose last position predicts the next document's first token: {cross}")


def score_one(url, served, doc, session):
    prompt = doc["ids"] + ([doc["next"]] if doc["next"] is not None else [])
    r = session.post(f"{url}/v1/completions", json={"model": served, "prompt": prompt, "max_tokens": 1,
                                                     "temperature": 0, "prompt_logprobs": 0}, timeout=3600)
    r.raise_for_status()
    plp = r.json()["choices"][0]["prompt_logprobs"]
    s = n = 0.0
    for i, wi in enumerate(doc["w"]):
        if wi <= 0:
            continue
        tgt = prompt[i + 1]
        lp = plp[i + 1][str(tgt)]["logprob"]
        s += -lp * wi; n += wi
    return s, n


def score(a):
    import requests
    docs = [json.loads(l) for l in open(a.parquet)]
    sess = requests.Session()
    t0 = time.time()
    with cf.ThreadPoolExecutor(a.concurrency) as ex:
        res = list(ex.map(lambda d: score_one(a.url, a.served, d, sess), docs))
    s = sum(x for x, _ in res); n = sum(y for _, y in res)
    packs = {}
    for d, (x, y) in zip(docs, res):
        ps = packs.setdefault(d["pack"], [0.0, 0.0]); ps[0] += x; ps[1] += y
    out = {"dump": a.parquet, "documents": len(docs), "weighted_tokens": n, "batch_nll": s / n,
           "per_pack_nll": {k: v[0] / v[1] for k, v in sorted(packs.items()) if v[1]}, "elapsed_s": round(time.time() - t0)}
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"BATCH0_NLL {s / n:.5f} over {n:.0f} weighted tokens, {len(docs)} documents")
    print("HELDOUT_NLL_DONE")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")
    d = sub.add_parser("dump")
    d.add_argument("--cache", required=True); d.add_argument("--tokenizer", required=True)
    d.add_argument("--stage", default="kimi_swesmith"); d.add_argument("--step", type=int, default=0)
    d.add_argument("--out", required=True)
    # score: the argument names heldout_nll.sbatch passes to its SCRIPT
    ap.add_argument("--tokenizer-dir"); ap.add_argument("--parquet"); ap.add_argument("--url")
    ap.add_argument("--served"); ap.add_argument("--out"); ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--max-len", type=int)
    a = ap.parse_args()
    dump(a) if a.cmd == "dump" else score(a)


if __name__ == "__main__":
    sys.exit(main())
