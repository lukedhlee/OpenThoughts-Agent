#!/usr/bin/env python3
"""Upload a harvested repo (or the shared pythons/) to the HF dataset laion/r2egym-build-artifacts, incrementally.

Layout uploaded verbatim from the work dir (see harvest_all.py):
  pythons/<uvdir>.tar.zst (+ .manifest.json)      packs/<shard>-<seq>.tar.zst + blobs.index   (content-addressed venv files)
  <repo>/<task>.{delta,tests}.tar.zst             <repo>/<task>.venv.json.zst + <repo>/<task>.manifest.json
  MANIFEST.json                                   per-repo summary (tasks, combos, bytes), rewritten on every call

Usage: harvest_upload.py --work /p/scratch/synthlaion/lee27/harvest --repo numpy [--dataset laion/r2egym-build-artifacts]
       harvest_upload.py --work ... --pythons        # just the interpreters
       harvest_upload.py --work ... --packs          # the blob packs + blobs.index (after every shard finished)
"""
import argparse
import glob
import json
import os
import time

from huggingface_hub import HfApi


def summarize(W, repo):
    mans = [json.load(open(p)) for p in glob.glob(os.path.join(W, repo, "*.manifest.json"))]
    by_py = {}
    for m in mans:
        by_py[m["python"]] = by_py.get(m["python"], 0) + 1
    return {
        "tasks": len(mans),
        "pythons": by_py,
        "venv_json_zst_bytes": sum(m["venv_json_zst"] for m in mans),
        "venv_new_blob_bytes_raw": sum(m["venv_new_bytes"] for m in mans),
        "delta_zst_bytes": sum(m["delta_slim_zst"] for m in mans),
        "tests_zst_bytes": sum(m["tests_zst"] for m in mans),
        "so_files_total": sum(m["so_files"] for m in mans),
        "tasks_with_deleted": sum(1 for m in mans if m["deleted"]),
        "so_bytes_total": sum(m["so_bytes"] for m in mans),
        "updated": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--repo", default=None)
    ap.add_argument("--pythons", action="store_true")
    ap.add_argument("--packs", action="store_true")
    ap.add_argument("--dataset", default="laion/r2egym-build-artifacts")
    a = ap.parse_args()
    api = HfApi()
    api.create_repo(a.dataset, repo_type="dataset", exist_ok=True)
    W = os.path.abspath(a.work)
    mpath = os.path.join(W, "MANIFEST.json")
    manifest = json.load(open(mpath)) if os.path.exists(mpath) else {"dataset": a.dataset, "layout": __doc__.strip().splitlines()[2:7], "repos": {}}
    if a.pythons:
        api.upload_folder(folder_path=os.path.join(W, "pythons"), path_in_repo="pythons", repo_id=a.dataset, repo_type="dataset",
                          commit_message="uv-managed interpreters")
        manifest["pythons"] = sorted(os.path.basename(p) for p in glob.glob(os.path.join(W, "pythons", "*.tar.zst")))
    if a.packs:
        api.upload_folder(folder_path=os.path.join(W, "packs"), path_in_repo="packs", repo_id=a.dataset, repo_type="dataset",
                          commit_message="blob packs")
        api.upload_file(path_or_fileobj=os.path.join(W, "blobs.index"), path_in_repo="blobs.index", repo_id=a.dataset,
                        repo_type="dataset", commit_message="blobs.index")
        packs = glob.glob(os.path.join(W, "packs", "*.tar.zst"))
        manifest["packs"] = {"n": len(packs), "zst_bytes": sum(os.path.getsize(p) for p in packs),
                             "unique_blobs": len({l.split("\t")[0] for l in open(os.path.join(W, "blobs.index"))})}
    if a.repo:
        r = a.repo
        manifest["repos"][r] = summarize(W, r)
        api.upload_folder(folder_path=os.path.join(W, r), path_in_repo=r, repo_id=a.dataset, repo_type="dataset",
                          commit_message=f"{r}: {manifest['repos'][r]['tasks']} tasks (deltas, tests, manifests)")
    json.dump(manifest, open(mpath, "w"), indent=1)
    api.upload_file(path_or_fileobj=mpath, path_in_repo="MANIFEST.json", repo_id=a.dataset, repo_type="dataset", commit_message="MANIFEST.json")
    print(json.dumps(manifest.get("repos", {}).get(a.repo, manifest.get("packs", manifest.get("pythons"))), indent=1))
    print("files in dataset:", len(api.list_repo_files(a.dataset, repo_type="dataset")))


if __name__ == "__main__":
    main()
