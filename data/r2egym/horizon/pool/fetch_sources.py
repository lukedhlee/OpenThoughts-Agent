#!/usr/bin/env python3
"""fetch_sources.py: pull only the columns the clean-pool build needs from the public HF datasets (login node, CPU only).

Writes into --out (default /scratch/11584/lukedhlee/rl_pool/src):
  coderforge_images.tsv   split, image, reward  (every row of every CoderForge-Preview `trajectories/` split; image column only
                          plus reward, read by parquet column projection over HfFileSystem)
  r2egym_v1.parquet       repo_name, docker_image, commit_hash, problem_statement  (R2E-Gym/R2E-Gym-V1)
  r2egym_subset.parquet   same columns for R2E-Gym/R2E-Gym-Subset
  swebv.parquet           repo, instance_id, base_commit, created_at  (princeton-nlp/SWE-bench_Verified)
Nemotron jsonl files are downloaded whole (hf_hub_download) and scanned by scan_nemotron.py.
"""
import argparse, os, sys
from concurrent.futures import ThreadPoolExecutor
import pyarrow as pa, pyarrow.parquet as pq
from huggingface_hub import HfFileSystem

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="/scratch/11584/lukedhlee/rl_pool/src")
ap.add_argument("--workers", type=int, default=6)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
fs = HfFileSystem()


def read_cols(path, cols):
    with fs.open(path, block_size=4 << 20) as f:
        return pq.ParquetFile(f).read(columns=cols)


def concat_dataset(repo, prefix, cols, dest):
    if os.path.exists(dest):
        print("have", dest); return
    files = sorted(p for p in fs.ls(f"datasets/{repo}/{prefix}", detail=False) if p.endswith(".parquet"))
    with ThreadPoolExecutor(a.workers) as ex:
        tabs = list(ex.map(lambda p: read_cols(p, cols), files))
    pq.write_table(pa.concat_tables(tabs), dest)
    print(repo, len(files), "files", sum(t.num_rows for t in tabs), "rows ->", dest, flush=True)


R2E_COLS = ["repo_name", "docker_image", "commit_hash", "problem_statement"]
concat_dataset("R2E-Gym/R2E-Gym-V1", "data", R2E_COLS, f"{a.out}/r2egym_v1.parquet")
concat_dataset("R2E-Gym/R2E-Gym-Subset", "data", R2E_COLS, f"{a.out}/r2egym_subset.parquet")
concat_dataset("princeton-nlp/SWE-bench_Verified", "data", ["repo", "instance_id", "base_commit", "created_at"],
               f"{a.out}/swebv.parquet")

dest = f"{a.out}/coderforge_images.tsv"
if not os.path.exists(dest):
    files = sorted(p for p in fs.ls("datasets/togethercomputer/CoderForge-Preview/trajectories", detail=False)
                   if p.endswith(".parquet"))
    print("coderforge files", len(files), flush=True)

    def one(p):
        t = read_cols(p, ["image", "reward"])
        split = os.path.basename(p).rsplit("-", 3)[0]
        return [(split, i, r) for i, r in zip(t.column("image").to_pylist(), t.column("reward").to_pylist())]

    n = 0
    with ThreadPoolExecutor(a.workers) as ex, open(dest + ".tmp", "w") as w:
        w.write("split\timage\treward\n")
        for k, rows in enumerate(ex.map(one, files)):
            for s, i, r in rows:
                w.write(f"{s}\t{i}\t{r}\n"); n += 1
            if k % 100 == 0: print(" ", k, n, flush=True)
    os.rename(dest + ".tmp", dest)
    print("coderforge rows", n, flush=True)
