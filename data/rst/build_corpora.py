#!/usr/bin/env python3
"""Assemble the three 2026-09-20 SFT corpora from the slice parquets (one merged train shard each, 1,000-row row groups).

  ota3_if_sft_v1      = OTA v2 {swesmith, superuser, tezos} train + if-v2 clean slice        (clean anchor, run 1)
  ota3_if_rst_v1      = ota3_if + rst-all   (RST GLM-5.3, >= 5 turns, one rollout per task)   (run 3)
  ota3_if_rstsucc_v1  = ota3_if + rst-succ  (the reward-1 subset of rst-all)                   (run 4)
  rst_if_sft_v1       = rst-all + the same if-v2 slice, no OTA                                 (source contrast, run 2)

IssueTasks is left out of every corpus (SWE-bench test issues inside it, ai_memory/notes/ota_swebench_contamination.md).
Held-out file per corpus = the three OTA slices' held-out rows merged (ota3-heldout), scored by heldout_nll.py as before;
the IF and RST held-out TASKS are harness probes and live with their slices. Streams row groups: fine on a login node.

    python build_corpora.py --ota <ota_sft_100k_v2> --ifv2 <ota_if_sft_v1/ifv2-train-*.parquet> --rst <rst_sft_v1> \
        --out-root /e/data1/mmlaion/lee27/snowball-sft/data
"""
import argparse, hashlib, json
from pathlib import Path
import pyarrow as pa, pyarrow.parquet as pq

OTA_SLICES = ("swesmith", "superuser", "tezos")


def write_merged(out: Path, name: str, sources: list[Path], schema: pa.Schema) -> dict:
    w = pq.ParquetWriter(out / name, schema, compression="zstd"); counts = {}
    for src in sources:
        f = pq.ParquetFile(src); n = 0
        for i in range(f.num_row_groups):
            t = f.read_row_group(i).select(schema.names).cast(schema); w.write_table(t, row_group_size=1000); n += t.num_rows
        counts[src.name] = n
    w.close()
    chk = pq.ParquetFile(out / name)
    assert max(chk.metadata.row_group(i).num_rows for i in range(chk.num_row_groups)) <= 1000
    return {"rows": chk.metadata.num_rows, "row_groups": chk.num_row_groups, "per_source": counts}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ota", required=True, type=Path); ap.add_argument("--ifv2", required=True, type=Path)
    ap.add_argument("--rst", required=True, type=Path); ap.add_argument("--out-root", required=True, type=Path)
    ap.add_argument("--only", nargs="*", default=None, help="corpus names to build (default: all)")
    a = ap.parse_args()
    schema = pq.ParquetFile(a.ota / "swesmith-train-00000-of-00001.parquet").schema_arrow
    ota_train = [a.ota / f"{s}-train-00000-of-00001.parquet" for s in OTA_SLICES]
    ota_held = [a.ota / f"{s}-heldout-00000-of-00001.parquet" for s in OTA_SLICES]
    corpora = {
        "ota3_if_sft_v1": ota_train + [a.ifv2],
        "ota3_if_rst_v1": ota_train + [a.ifv2, a.rst / "rst-all-train-00000-of-00001.parquet"],
        "ota3_if_rstsucc_v1": ota_train + [a.ifv2, a.rst / "rst-succ-train-00000-of-00001.parquet"],
        "rst_if_sft_v1": [a.ifv2, a.rst / "rst-all-train-00000-of-00001.parquet"],
    }
    for corpus, sources in corpora.items():
        if a.only and corpus not in a.only:
            continue
        out = a.out_root / corpus; out.mkdir(parents=True, exist_ok=True)
        train_name = f"{corpus.removesuffix('_v1').replace('_', '-')}-train-00000-of-00001.parquet"
        info = write_merged(out, train_name, sources, schema)
        held = write_merged(out, "ota3-heldout-00000-of-00001.parquet", ota_held, schema)
        sums = {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(out.glob("*.parquet"))}
        (out / "SHA256SUMS").write_text("".join(f"{h}  {n}\n" for n, h in sums.items()))
        (out / "parquet.list").write_text(f"{out}/{train_name}\n")
        census = {"corpus": corpus, "train": info, "heldout": held, "issue_slice_excluded": True, "ota_excluded": corpus == "rst_if_sft_v1", "sources": [str(s) for s in sources], "sha256": sums}
        (out / "census.json").write_text(json.dumps(census, indent=1))
        print(corpus, json.dumps(info), "heldout", held["rows"], flush=True)


if __name__ == "__main__":
    main()
