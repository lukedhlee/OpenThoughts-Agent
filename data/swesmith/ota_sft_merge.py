#!/usr/bin/env python3
"""Merge the per-slice outputs of ota_sft_convert.py into one training set, and cut the sweep subsets.

Writes, next to the per-slice files:
  ota-all-train-00000-of-00001.parquet     every slice's train rows (ONE source file -> ONE cache shard)
  ota-all-heldout-00000-of-00001.parquet   every slice's held-out rows (the full-arm scoring file)
  ota-sub-train-00000-of-00001.parquet     a stratified SUB_FRAC sample of train rows per slice (the lr sweep)
  ota-sub-heldout-00000-of-00001.parquet   SUB_HELDOUT rows per slice (the sweep's scoring file)
  parquet.list / parquet.sub.list, SHA256SUMS, census.json

    python data/swesmith/ota_sft_merge.py --root <dir with <slice>/ subdirs> --out <dir> \
        --jupiter-root /e/data1/mmlaion/lee27/snowball-sft/data/ota_sft_100k_v2
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

SLICES = ("swesmith", "superuser", "issue", "tezos")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--jupiter-root", required=True)
    ap.add_argument("--sub-frac", type=float, default=0.06)
    ap.add_argument("--sub-heldout", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    train, heldout, sub_train, sub_heldout = [], [], [], []
    per_slice = {}
    for s in SLICES:
        d = args.root / s
        t = pq.read_table(d / f"{s}-train-00000-of-00001.parquet")
        h = pq.read_table(d / f"{s}-heldout-00000-of-00001.parquet")
        census = json.loads((d / "census.json").read_text())
        per_slice[s] = census["per_slice"][s] | {"counts": census["counts"]}
        train.append(t)
        heldout.append(h)
        n_sub = round(len(t) * args.sub_frac)
        sub_train.append(t.take(sorted(rng.sample(range(len(t)), n_sub))))
        sub_heldout.append(h.take(sorted(rng.sample(range(len(h)), min(args.sub_heldout, len(h))))))
        (args.out / f"{s}-train-00000-of-00001.parquet").write_bytes((d / f"{s}-train-00000-of-00001.parquet").read_bytes())
        (args.out / f"{s}-heldout-00000-of-00001.parquet").write_bytes((d / f"{s}-heldout-00000-of-00001.parquet").read_bytes())

    written = {}
    for name, tabs in (("ota-all-train", train), ("ota-all-heldout", heldout), ("ota-sub-train", sub_train), ("ota-sub-heldout", sub_heldout)):
        tab = pa.concat_tables(tabs)
        f = args.out / f"{name}-00000-of-00001.parquet"
        pq.write_table(tab, f, compression="zstd")
        written[f.name] = (len(tab), hashlib.sha256(f.read_bytes()).hexdigest())
    for s in SLICES:
        for split in ("train", "heldout"):
            f = args.out / f"{s}-{split}-00000-of-00001.parquet"
            written[f.name] = (pq.read_metadata(f).num_rows, hashlib.sha256(f.read_bytes()).hexdigest())
    root = args.jupiter_root.rstrip("/")
    (args.out / "parquet.list").write_text(f"{root}/ota-all-train-00000-of-00001.parquet\n")
    (args.out / "parquet.sub.list").write_text(f"{root}/ota-sub-train-00000-of-00001.parquet\n")
    (args.out / "SHA256SUMS").write_text("".join(f"{h}  {n}\n" for n, (_, h) in sorted(written.items())))
    census = {
        "screen": "reference",
        "sub_frac": args.sub_frac, "sub_heldout_per_slice": args.sub_heldout, "seed": args.seed,
        "rows": {n: r for n, (r, _) in sorted(written.items())},
        "train_tokens_total": sum(v["train_tokens"] for v in per_slice.values()),
        "heldout_tokens_total": sum(v["heldout_tokens"] for v in per_slice.values()),
        "per_slice": per_slice,
        "sha256": {n: h for n, (_, h) in sorted(written.items())},
    }
    (args.out / "census.json").write_text(json.dumps(census, indent=1))
    print(json.dumps({k: v for k, v in census.items() if k != "per_slice"}, indent=1))
    for s, v in per_slice.items():
        print(s, {k: v[k] for k in ("train_rows", "heldout_rows", "train_tokens", "train_tokens_median", "compaction_rows")}, {k: c for k, c in v["counts"].items() if k.startswith("drop")})


if __name__ == "__main__":
    main()
