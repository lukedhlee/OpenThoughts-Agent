#!/usr/bin/env python3
"""Held-out loss of a Snowball export on the Kimi SWE-smith held-out split, through a running vLLM server.

Renders each held-out conversation with the export's own chat template (the same Marin template the SFT cache
was packed with), asks vLLM for the log-probability of every prompt token (``prompt_logprobs``), and reports
the token-weighted mean negative log-likelihood over the ASSISTANT tokens only, which is what the trainer's
loss is (user turns are masked). Also splits the assistant tokens into think spans (between the special
tokens <|start_think|> / <|end_think|>) and the rest, since the two may move differently under SFT.

    python heldout_nll.py --tokenizer-dir <export dir> --parquet heldout-00000-of-00001.parquet \
        --url http://localhost:8000 --served model --out heldout_nll.json [--limit N] [--concurrency 16]
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import statistics as st
import time

import pyarrow.parquet as pq
import requests

START, END = 128002, 128003


def score_one(url, served, ids, session):
    r = session.post(
        f"{url}/v1/completions",
        json={"model": served, "prompt": ids, "max_tokens": 1, "temperature": 0, "prompt_logprobs": 0},
        timeout=900,
    )
    r.raise_for_status()
    pl = r.json()["choices"][0]["prompt_logprobs"]  # list aligned with ids; entry 0 is None
    out = []
    for i, entry in enumerate(pl):
        if entry is None:
            out.append(None)
            continue
        # prompt_logprobs=0 returns only the actual token; keys are token ids as strings
        v = entry.get(str(ids[i])) or next(iter(entry.values()))
        out.append(float(v["logprob"]))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tokenizer-dir", required=True)
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--served", default="model")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=32768)
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer_dir)
    assert tok.convert_tokens_to_ids("<|start_think|>") == START
    # A multi-slice parquet (OTA) carries slice / row_id; the Kimi split has neither, so both are optional.
    have = set(pq.read_schema(args.parquet).names)
    cols = ["conversations", "instance_id", "result"] + [c for c in ("slice", "row_id") if c in have]
    rows = pq.read_table(args.parquet, columns=cols).to_pylist()
    if args.limit:
        rows = rows[: args.limit]

    # ids + assistant mask straight from the template's {% generation %} blocks
    seqs = []
    for r in rows:
        enc = tok.apply_chat_template(
            r["conversations"], tokenize=True, add_generation_prompt=False, return_dict=True,
            return_assistant_tokens_mask=True,
        )
        ids, mask = list(enc["input_ids"]), list(enc["assistant_masks"])
        if len(ids) > args.max_len:
            continue
        seqs.append((r.get("row_id") or r["instance_id"], r.get("slice"), r["result"], ids, mask))
    print(f"sequences {len(seqs)} (of {len(rows)} rows), assistant tokens {sum(sum(m) for _, _, _, m in seqs)}", flush=True)

    session = requests.Session()
    t0 = time.time()
    per = []
    tot_nll = tot_n = think_nll = think_n = rest_nll = rest_n = 0.0

    def work(item):
        inst, sl, res, ids, mask = item
        lp = score_one(args.url, args.served, ids, session)
        s_all = n_all = s_th = n_th = s_re = n_re = 0.0
        in_think = False
        for tid, m, l in zip(ids, mask, lp):
            if tid == START:
                in_think = True
            if m and l is not None:
                s_all += -l; n_all += 1
                if in_think:
                    s_th += -l; n_th += 1
                else:
                    s_re += -l; n_re += 1
            if tid == END:
                in_think = False
        return inst, sl, res, len(ids), s_all, n_all, s_th, n_th, s_re, n_re

    with cf.ThreadPoolExecutor(args.concurrency) as ex:
        for k, (inst, sl, res, n_ids, s_all, n_all, s_th, n_th, s_re, n_re) in enumerate(ex.map(work, seqs), 1):
            per.append({"instance_id": inst, "slice": sl, "result": res, "tokens": n_ids, "assistant_tokens": n_all,
                        "nll": s_all / max(n_all, 1), "think_nll": s_th / max(n_th, 1), "rest_nll": s_re / max(n_re, 1)})
            tot_nll += s_all; tot_n += n_all; think_nll += s_th; think_n += n_th; rest_nll += s_re; rest_n += n_re
            if k % 50 == 0:
                print(f"{k}/{len(seqs)} nll so far {tot_nll / max(tot_n, 1):.4f} ({time.time() - t0:.0f}s)", flush=True)

    summary = {
        "model": args.served, "tokenizer_dir": args.tokenizer_dir, "parquet": args.parquet,
        "sequences": len(per), "assistant_tokens": tot_n,
        "heldout_nll": tot_nll / max(tot_n, 1),
        "think_nll": think_nll / max(think_n, 1), "think_tokens": think_n,
        "rest_nll": rest_nll / max(rest_n, 1), "rest_tokens": rest_n,
        "per_sequence_nll_median": st.median(p["nll"] for p in per) if per else None,
        "pass_nll": st.mean(p["nll"] for p in per if str(p["result"]) in ("1.0", "1")) if per else None,
        "fail_nll": st.mean(p["nll"] for p in per if str(p["result"]) not in ("1.0", "1")) if per else None,
        "by_slice": {
            sl: {
                "sequences": len(g),
                "assistant_tokens": sum(p["assistant_tokens"] for p in g),
                "heldout_nll": sum(p["nll"] * p["assistant_tokens"] for p in g) / max(sum(p["assistant_tokens"] for p in g), 1),
            }
            for sl, g in sorted(
                ((sl, [p for p in per if p["slice"] == sl]) for sl in {p["slice"] for p in per if p["slice"]})
            )
        },
        "elapsed_s": round(time.time() - t0),
        "per_sequence": per,
    }
    json.dump(summary, open(args.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k != "per_sequence"}, indent=1))
    print("HELDOUT_NLL", f"{summary['heldout_nll']:.4f}", flush=True)
    print("HELDOUT_NLL_DONE")


if __name__ == "__main__":
    main()
