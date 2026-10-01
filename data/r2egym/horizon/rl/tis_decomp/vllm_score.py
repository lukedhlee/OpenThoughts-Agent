#!/usr/bin/env python3
"""tis_decomp, vLLM side: score fixed (prompt + completion) token sequences with prompt logprobs AND the experts vLLM
routed every token to, so the trainer can compare expert sets and replay them.

gate2_score.py `score` plus three deltas: the server runs with --enable-return-routed-experts and each response's
`routed_experts` (base64 .npy, [prompt_len, 26, 4] uint8, prompt = P + C here) is saved per turn; several servers can
share one rep (`--urls`, rows dealt round-robin, `--conc` requests in flight per server); one rep per call, written
to its own JSONL (`{"meta"}` line, then `{"id", "lp", "secs", "rep", "label"}` lines, the gate-2 format).

  python vllm_score.py --set S --out OUT.jsonl --label s3_dp4ep_c4 --rep 0 --conc 4 --route-dir DIR
"""
import argparse
import base64
import concurrent.futures as cf
import hashlib
import io
import json
import os
import sys
import time

HORIZON = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # .../data/r2egym/horizon
sys.path.insert(0, HORIZON)
from gate2_score import lp_of, post  # noqa: E402  (the gate-2 client, unchanged)


def score_one(url, model, row, route_dir):
    import numpy as np
    P, C = row["prompt_ids"], row["completion_ids"]
    t0 = time.time()
    r = post(url + "/v1/completions", {"model": model, "prompt": P + C, "max_tokens": 1, "temperature": 0.0,
                                       "prompt_logprobs": 0})
    ch = r["choices"][0]
    pl = ch["prompt_logprobs"]
    if len(pl) != len(P) + len(C):
        raise ValueError(f"{row['id']}: {len(pl)} prompt logprobs for {len(P) + len(C)} tokens")
    lps = [lp_of(pl[len(P) + j], tok) for j, tok in enumerate(C)]
    if any(x is None for x in lps):
        raise ValueError(f"{row['id']}: completion token missing from its prompt-logprob entry")
    shape = None
    if route_dir:
        b64 = ch.get("routed_experts")
        if b64 is None:
            raise ValueError(f"{row['id']}: no routed_experts in the response (server flag off?)")
        arr = np.load(io.BytesIO(base64.b64decode(b64)))
        shape = list(arr.shape)
        if arr.shape[0] < len(P) + len(C):
            raise ValueError(f"{row['id']}: routed_experts covers {arr.shape[0]} of {len(P) + len(C)} tokens")
        np.save(os.path.join(route_dir, row["id"] + ".npy"), arr[: len(P) + len(C)].astype(np.uint8))
    return {"id": row["id"], "lp": lps, "secs": round(time.time() - t0, 2), "route_shape": shape}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--urls", default="http://localhost:8000", help="comma list; rows dealt round-robin")
    ap.add_argument("--model", default="snowball")
    ap.add_argument("--set", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--rep", type=int, default=0)
    ap.add_argument("--conc", type=int, default=4, help="requests in flight per server")
    ap.add_argument("--route-dir", default="", help="save routed experts per turn here (empty = do not ask)")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.set)]
    if a.limit:
        rows = rows[: a.limit]
    urls = a.urls.split(",")
    if a.route_dir:
        os.makedirs(a.route_dir, exist_ok=True)
    meta = {"label": a.label, "rep": a.rep, "set": a.set, "set_sha256": hashlib.sha256(open(a.set, "rb").read()).hexdigest(),
            "n": len(rows), "conc_per_server": a.conc, "urls": urls, "host": os.uname().nodename,
            "job": os.environ.get("SLURM_JOB_ID"), "route_dir": a.route_dir}
    print(json.dumps(meta), flush=True)
    t0 = time.time()
    pools = {u: cf.ThreadPoolExecutor(a.conc) for u in urls}
    futs = [pools[urls[i % len(urls)]].submit(score_one, urls[i % len(urls)], a.model, r, a.route_dir)
            for i, r in enumerate(rows)]
    shapes = set()
    with open(a.out, "w") as f:
        f.write(json.dumps({"meta": meta}) + "\n")
        for i, fu in enumerate(futs):
            res = fu.result()
            shapes.add(tuple(res.pop("route_shape") or ())[1:])
            res["rep"], res["label"] = a.rep, a.label
            f.write(json.dumps(res) + "\n")
            if (i + 1) % 16 == 0:
                print(f"{a.label} rep {a.rep}: {i + 1}/{len(rows)} scored, {time.time() - t0:.0f}s", flush=True)
    for p in pools.values():
        p.shutdown()
    print(f"SCORE_REP_DONE {a.label} rep={a.rep} n={len(rows)} secs={time.time() - t0:.0f} route_shapes={sorted(shapes)}",
          flush=True)


if __name__ == "__main__":
    main()
