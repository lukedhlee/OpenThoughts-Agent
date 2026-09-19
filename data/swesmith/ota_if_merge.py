#!/usr/bin/env python3
"""Append the if-v2 slice to ota-all-train -> the ota_if_sft_v1 set (ONE source file, 1,000-row row groups).

Streams row groups so it runs on the Jupiter login node in a few hundred MB.

    python ota_if_merge.py --ota <ota_sft_100k_v2 dir> --ifv2 <ifv2-train parquet> --out <ota_if_sft_v1 dir> \
        --jupiter-root /e/data1/mmlaion/lee27/snowball-sft/data/ota_if_sft_v1
"""
import argparse, hashlib, json, shutil
from pathlib import Path
import pyarrow as pa, pyarrow.parquet as pq

ap = argparse.ArgumentParser()
ap.add_argument("--ota", required=True, type=Path); ap.add_argument("--ifv2", required=True, type=Path)
ap.add_argument("--out", required=True, type=Path); ap.add_argument("--jupiter-root", required=True)
a = ap.parse_args()
a.out.mkdir(parents=True, exist_ok=True)
src = pq.ParquetFile(a.ota / "ota-all-train-00000-of-00001.parquet")
name = "ota-if-train-00000-of-00001.parquet"
w = pq.ParquetWriter(a.out / name, src.schema_arrow, compression="zstd")
n_ota = 0
for i in range(src.num_row_groups):
    t = src.read_row_group(i); w.write_table(t, row_group_size=1000); n_ota += t.num_rows
ifv = pq.read_table(a.ifv2).select(src.schema_arrow.names).cast(src.schema_arrow)
w.write_table(ifv, row_group_size=1000); w.close()
shutil.copy(a.ota / "ota-all-heldout-00000-of-00001.parquet", a.out / "ota-all-heldout-00000-of-00001.parquet")
shutil.copy(a.ifv2, a.out / a.ifv2.name)
sums = {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(a.out.glob("*.parquet"))}
(a.out / "SHA256SUMS").write_text("".join(f"{h}  {n}\n" for n, h in sums.items()))
(a.out / "parquet.list").write_text(f"{a.jupiter_root.rstrip('/')}/{name}\n")
chk = pq.ParquetFile(a.out / name)
census = {"ota_rows": n_ota, "ifv2_rows": ifv.num_rows, "total_rows": chk.metadata.num_rows, "row_groups": chk.num_row_groups,
          "max_row_group": max(chk.metadata.row_group(i).num_rows for i in range(chk.num_row_groups)), "sha256": sums}
(a.out / "census.json").write_text(json.dumps(census, indent=1)); print(json.dumps(census, indent=1))
