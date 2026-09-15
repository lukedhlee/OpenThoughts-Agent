#!/usr/bin/env python3
"""Summarise a finished harvest (harvest_all.py work dir): per-repo tasks, Pythons, unique bytes, projected image size.

Usage: harvest_summary.py --work /p/scratch/synthlaion/lee27/harvest [--json out.json]
Projection per repo image = R2E-Gym base prefix (ubuntu + apt + uv + repo clone, taken from the Docker Hub layer sizes
measured on 2026-09-14: ~0.6 GB compressed) + the interpreters that repo uses + the blobs its venvs reference (raw bytes,
deduplicated within the repo; on the image they sit uncompressed) + the slim deltas + tests + venv manifests (zst).
"""
import argparse
import collections
import glob
import json
import os
import subprocess

BASE_PREFIX_GB = 0.6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    W = a.work
    repos = sorted(d for d in os.listdir(W) if d not in ("pythons", "packs", "tmp") and os.path.isdir(os.path.join(W, d)) and glob.glob(os.path.join(W, d, "*.manifest.json")))
    pysize = {os.path.basename(p)[:-8]: os.path.getsize(p) for p in glob.glob(os.path.join(W, "pythons", "*.tar.zst"))}
    packs = glob.glob(os.path.join(W, "packs", "*.tar.zst"))
    index = {}
    for line in open(os.path.join(W, "blobs.index")):
        sha, size = line.rstrip("\n").split("\t")
        index[sha] = int(size)
    out = {"repos": {}, "pythons": {k: v for k, v in pysize.items()},
           "packs": {"n": len(packs), "zst_bytes": sum(os.path.getsize(p) for p in packs), "unique_blobs": len(index),
                     "unique_raw_bytes": sum(index.values())}}
    grand = collections.Counter()
    for r in repos:
        mans = [json.load(open(p)) for p in glob.glob(os.path.join(W, r, "*.manifest.json"))]
        # blobs referenced by this repo's venvs (deduplicated within the repo)
        shas = set()
        for p in glob.glob(os.path.join(W, r, "*.venv.json.zst")):
            vj = json.loads(subprocess.run(["zstd", "-dc", p], capture_output=True, check=True).stdout)
            shas.update(v[0] for v in vj["files"].values())
        blob_raw = sum(index.get(s, 0) for s in shas)
        pys = collections.Counter(m["python"] for m in mans)
        uvdirs = sorted({m["uvdir"] for m in mans})
        delta = sum(m["delta_slim_zst"] for m in mans)
        tests = sum(m["tests_zst"] for m in mans)
        vjson = sum(m["venv_json_zst"] for m in mans)
        pybytes = sum(pysize.get(u, 0) for u in uvdirs)
        image_gb = BASE_PREFIX_GB + (pybytes * 2.2 + blob_raw + delta + tests + vjson) / 1e9  # interpreters ~2.2x when unpacked
        out["repos"][r] = {
            "tasks": len(mans), "pythons": dict(pys), "uvdirs": uvdirs,
            "venv_files_mean": round(sum(m["venv_files"] for m in mans) / len(mans)),
            "venv_bytes_mean_MB": round(sum(m["venv_bytes"] for m in mans) / len(mans) / 1e6),
            "unique_blobs": len(shas), "unique_blob_raw_MB": round(blob_raw / 1e6),
            "dedupe_ratio": round(sum(m["venv_bytes"] for m in mans) / max(blob_raw, 1), 1),
            "delta_zst_MB": round(delta / 1e6), "delta_zst_per_task_MB": round(delta / len(mans) / 1e6, 2),
            "so_files_mean": round(sum(m["so_files"] for m in mans) / len(mans), 1),
            "tests_zst_MB": round(tests / 1e6), "venv_json_zst_MB": round(vjson / 1e6),
            "tasks_with_deleted": sum(1 for m in mans if m["deleted"]),
            "t_mean_s": round(sum(m["t_extract"] + m["t_venv"] + m["t_delta"] for m in mans) / len(mans), 1),
            "projected_image_GB": round(image_gb, 1),
        }
        grand.update({"tasks": len(mans), "delta": delta, "tests": tests, "vjson": vjson})
    out["total"] = {"tasks": grand["tasks"], "delta_zst_GB": round(grand["delta"] / 1e9, 2), "tests_zst_MB": round(grand["tests"] / 1e6),
                    "venv_json_zst_MB": round(grand["vjson"] / 1e6), "packs_zst_GB": round(out["packs"]["zst_bytes"] / 1e9, 2),
                    "blobs_raw_GB": round(out["packs"]["unique_raw_bytes"] / 1e9, 2),
                    "pythons_zst_MB": round(sum(pysize.values()) / 1e6),
                    "images_GB_sum": round(sum(v["projected_image_GB"] for v in out["repos"].values()), 1)}
    print(json.dumps(out, indent=1))
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
