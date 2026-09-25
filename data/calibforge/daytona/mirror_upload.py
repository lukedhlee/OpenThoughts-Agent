#!/usr/bin/env python3
"""Mirror every per-task layer blob of a CalibForge Daytona tree to a public HF dataset, keyed by digest.

Layout in the dataset: blobs/sha256/<first 2 hex>/<64 hex> (the raw gzip layer, byte-identical to the registry blob,
so setup.sh checks the same sha256 whichever source served it). Streams in batches: download a batch from Docker Hub
(sha256-verified), upload it as one commit, delete the local copy, while the next batch downloads. Resumable: blobs
already in the dataset are skipped. Backs off on HF 429s.

  python mirror_upload.py --tree <tree> --repo laion/calibforge-daytona-layers --work <scratch dir> [--batch-gb 6]
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import shutil
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from registry import Hub  # noqa: E402

REPO_IMAGE = "aweaiteam/calibforge"


def tree_blobs(tree: Path) -> dict[str, int]:
    blobs = {}
    for sh in tree.glob("*/setup_files/setup.sh"):
        body = sh.read_text().split('LAYERS="', 1)[1].split('"', 1)[0]
        for line in body.splitlines():
            if line.strip():
                d, s = line.split()
                blobs[d] = int(s)
    return blobs


def rel(digest: str) -> str:
    h = digest.split(":")[1]
    return f"blobs/sha256/{h[:2]}/{h}"


def fetch(hub: Hub, digest: str, dest: Path) -> None:
    hub.blob_to(REPO_IMAGE, digest, dest)
    h = hashlib.sha256()
    with dest.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    if h.hexdigest() != digest.split(":")[1]:
        dest.unlink()
        raise ValueError(f"digest mismatch {digest}")


def upload(api, repo: str, folder: Path, n: int, tries: int = 8) -> None:
    for attempt in range(tries):
        try:
            api.upload_folder(repo_id=repo, repo_type="dataset", folder_path=str(folder),
                              commit_message=f"layer blobs ({n})")
            return
        except Exception as exc:  # noqa: BLE001
            code = getattr(getattr(exc, "response", None), "status_code", None)
            wait = 60 * (attempt + 1) if code == 429 else 15 * (attempt + 1)
            print(f"upload failed ({code} {type(exc).__name__}: {str(exc)[:200]}); retry in {wait}s", flush=True)
            time.sleep(wait)
    raise RuntimeError(f"upload of {folder} failed {tries} times")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tree", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--batch-gb", type=float, default=6)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    from huggingface_hub import HfApi
    api = HfApi()
    blobs = tree_blobs(Path(a.tree))
    have = {p for p in api.list_repo_files(a.repo, repo_type="dataset") if p.startswith("blobs/")}
    todo = sorted((d for d in blobs if rel(d) not in have), key=lambda d: d)
    print(f"{len(blobs)} blobs ({sum(blobs.values()) / 1e9:.1f} GB); {len(todo)} to upload "
          f"({sum(blobs[d] for d in todo) / 1e9:.1f} GB)", flush=True)
    batches, cur, size = [], [], 0
    for d in todo:
        cur.append(d); size += blobs[d]
        if size >= a.batch_gb * 1e9:
            batches.append(cur); cur, size = [], 0
    if cur:
        batches.append(cur)
    hub = Hub()
    work = Path(a.work)
    uploader = None
    done_bytes, t0 = 0, time.time()
    lock = threading.Lock()
    for i, batch in enumerate(batches):
        folder = work / f"batch{i:04d}"
        with cf.ThreadPoolExecutor(a.workers) as ex:
            list(ex.map(lambda d: fetch(hub, d, folder / rel(d)), batch))
        if uploader:
            uploader.join()
        def job(folder=folder, batch=batch, i=i):
            nonlocal done_bytes
            upload(api, a.repo, folder, len(batch))
            shutil.rmtree(folder)
            with lock:
                done_bytes += sum(blobs[d] for d in batch)
            el = time.time() - t0
            print(f"batch {i + 1}/{len(batches)} up: {done_bytes / 1e9:.1f} GB in {el / 60:.1f} min "
                  f"({done_bytes / el / 1e6:.0f} MB/s)", flush=True)
        uploader = threading.Thread(target=job)
        uploader.start()
    if uploader:
        uploader.join()
    have = {p for p in api.list_repo_files(a.repo, repo_type="dataset") if p.startswith("blobs/")}
    missing = [d for d in blobs if rel(d) not in have]
    print(f"verify: {len(blobs) - len(missing)}/{len(blobs)} blobs present", flush=True)
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
