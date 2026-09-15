#!/usr/bin/env python3
"""Materialise one harvested task inside a sandbox (the consumer side of harvest_all.py's layout).

Expected on the image (one per repo):
  /opt/artifacts/blobs/<sha[:2]>/<sha256>          every pack extracted (tar --zstd -xf packs/*.tar.zst -C /opt/artifacts)
  /root/.local/share/uv/python/<uvdir>/            pythons/<uvdir>.tar.zst extracted at /
  /opt/artifacts/<repo>/<task>.venv.json.zst       + <task>.delta.tar.zst + <task>.tests.tar.zst + <task>.manifest.json
  /testbed                                         git clone of the repo (any commit)

Setup for a task (what setup_files/setup.sh runs; ~1-2 s):
  git -C /testbed checkout -f <base_commit> && git -C /testbed clean -fdxq
  python3 harvest_materialize.py --artifacts /opt/artifacts --repo <repo> --task <task> [--testbed /testbed] [--verify]

It hard-links every venv file from the blob store (falls back to copy across filesystems), recreates symlinks and empty
dirs, untars the slim build delta (in-place .so, egg-info, ...) with mtimes preserved, deletes the files install.sh had
deleted, and drops /r2e_tests + run_tests.sh from the tests tarball. --verify re-hashes the venv against the manifest.
Stdlib only; runs on the sandbox's system python3 (the venv's python is a different interpreter).
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time


_DCTX = None


def _zstd_module():
    global _DCTX
    if _DCTX is None:
        try:
            import zstandard
            _DCTX = zstandard.ZstdDecompressor()
        except ImportError:
            _DCTX = False
    return _DCTX


def place_blob(blobs, sha, dst):
    """Raw blob (v1 store): hard link, copy across filesystems. Compressed blob (v2 store, <sha>.zst): decompress."""
    src = os.path.join(blobs, sha[:2], sha)
    if os.path.exists(src):
        try:
            os.link(src, dst)
        except OSError:
            shutil.copyfile(src, dst)
        return True
    srcz = src + ".zst"
    if not os.path.exists(srcz):
        return False
    d = _zstd_module()
    if d:
        with open(srcz, "rb") as fi, open(dst, "wb") as fo:
            d.copy_stream(fi, fo)
    else:
        with open(dst, "wb") as fo:
            subprocess.run(["zstd", "-dc", srcz], stdout=fo, check=True)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default="/opt/artifacts")
    ap.add_argument("--repo", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--testbed", default="/testbed")
    ap.add_argument("--venv-root", default=None,
                    help="materialise the venv under this dir instead of the testbed (the caller symlinks <testbed>/.venv to it); "
                         "keeps the venv tree out of the repo so TaskTrove v5.1's trusted-test restore never walks it")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--no-tests", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    A, tb = a.artifacts, a.testbed
    vroot = a.venv_root or tb
    base = os.path.join(A, a.repo, a.task)
    man = json.load(open(base + ".manifest.json"))
    vj = json.loads(subprocess.run(["zstd", "-dc", base + ".venv.json.zst"], capture_output=True, check=True).stdout)
    venv = os.path.join(vroot, vj["root"])
    if os.path.lexists(venv):
        shutil.rmtree(venv, ignore_errors=True)
    for d in vj["dirs"]:
        os.makedirs(os.path.join(vroot, d), exist_ok=True)
    blobs = os.path.join(A, "blobs")
    missing = []
    for rel, (sha, size, mode) in vj["files"].items():
        dst = os.path.join(vroot, rel)
        if not place_blob(blobs, sha, dst):
            missing.append(rel)
            continue
        if int(mode, 8) & 0o111 and not os.access(dst, os.X_OK):
            os.chmod(dst, int(mode, 8))
    for rel, target in vj["links"].items():
        dst = os.path.join(vroot, rel)
        if os.path.lexists(dst):
            os.remove(dst)
        os.symlink(target, dst)
    t1 = time.time()
    dj_path = os.path.join(A, a.repo, "v2", a.task + ".delta.json")
    if os.path.exists(dj_path):
        dj = json.load(open(dj_path))
        for d in dj["dirs"]:
            os.makedirs(os.path.join(tb, d), exist_ok=True)
        for rel, (sha, size, mode, mtime_ns) in dj["files"].items():
            dst = os.path.join(tb, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            if os.path.lexists(dst):
                os.remove(dst)
            if not place_blob(blobs, sha, dst):
                missing.append("delta:" + rel)
                continue
            os.chmod(dst, int(mode, 8))
            os.utime(dst, ns=(mtime_ns, mtime_ns))   # setuptools compares .so mtimes against sources: keep the harvested ones
        for rel, target in dj["links"].items():
            dst = os.path.join(tb, rel)
            if os.path.lexists(dst):
                os.remove(dst)
            os.symlink(target, dst)
    elif man["delta_slim_zst"]:
        subprocess.run(["tar", "--zstd", "-xpf", base + ".delta.tar.zst", "-C", tb], check=True)
    for rel in man.get("deleted", []):
        p = os.path.join(tb, rel)
        if os.path.lexists(p):
            os.remove(p)
    if not a.no_tests and man.get("tests_zst"):
        subprocess.run(["tar", "--zstd", "-xpf", base + ".tests.tar.zst", "-C", "/"], check=True)
    t2 = time.time()
    bad = []
    if a.verify:
        for rel, (sha, size, mode) in vj["files"].items():
            p = os.path.join(vroot, rel)
            if not os.path.isfile(p):
                bad.append(rel); continue
            h = hashlib.sha256()
            with open(p, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
            if h.hexdigest() != sha:
                bad.append(rel)
    print(json.dumps({"task": a.task, "python": vj.get("python"), "venv_files": len(vj["files"]), "links": len(vj["links"]),
                      "missing_blobs": len(missing), "verify_bad": len(bad), "t_venv": round(t1 - t0, 2),
                      "t_delta": round(t2 - t1, 2), "base_commit": man["base_commit"]}))
    if missing or bad:
        print("MISSING:", missing[:5], "BAD:", bad[:5], file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
