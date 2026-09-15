#!/usr/bin/env python3
"""Upload harvested artifacts to the HF dataset the pilot snapshot pulls from.

Canonical names in the repo (what pilot_snapshot.py / pilot_sandbox.py expect):
  <task>.venv.tar.zst      the task's /testbed/.venv (uv venv; symlinks into /root/.local/share/uv/python/<dir>)
  <task>.delta.tar.zst     the SLIM delta (in-place .so, egg-info, modified tracked files; no build/, pycache, .c)
  <task>.tests.tar.zst     /r2e_tests + /testbed/run_tests.sh + install.sh
  uvpython_<dir>.tar.zst   one per uv-managed interpreter (paths root/.local/share/uv/python/<dir>)
  manifests/<task>.deleted.list, manifests/summary.tsv, manifests/slim_sizes.txt
Usage: python upload_pilot_artifacts.py <harvest_pilot dir> laion/r2egym-build-artifacts-pilot
"""
import os
import shutil
import sys
import tempfile

from huggingface_hub import HfApi

src, repo = sys.argv[1], sys.argv[2]
stage = tempfile.mkdtemp(prefix="hfstage_")
os.makedirs(f"{stage}/manifests")
n = 0
for f in sorted(os.listdir(f"{src}/out")):
    if f.endswith(".delta.tar.zst"):
        continue  # full delta stays local
    dst = f.replace(".delta-slim.tar.zst", ".delta.tar.zst")
    os.link(f"{src}/out/{f}", f"{stage}/{dst}")
    n += 1
for f in sorted(os.listdir(f"{src}/manifests")):
    if f.endswith(".deleted.list") or f in ("summary.tsv", "slim_sizes.txt", "tarsizes.txt"):
        shutil.copy(f"{src}/manifests/{f}", f"{stage}/manifests/{f}")
        n += 1
print(f"staging {n} files from {stage}")
api = HfApi()
api.upload_folder(folder_path=stage, repo_id=repo, repo_type="dataset", commit_message="pandas pilot: 8 tasks harvested from R2E-Gym images")
print("uploaded:", len(api.list_repo_files(repo, repo_type="dataset")), "files now in", repo)
shutil.rmtree(stage)
