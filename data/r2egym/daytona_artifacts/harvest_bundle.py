#!/usr/bin/env python3
"""Per-repo bundles for the Daytona per-repo images (consumer of harvest_all.py's work dir; runs on the Jupiter login node).

For each repo it writes and uploads three files next to the per-task artifacts in the HF dataset:
  <repo>/bundle-blobs.tar.zst   only the blobs that repo's venvs reference, as blobs/<sha[:2]>/<sha256>   (extract at /opt/artifacts)
  <repo>/bundle-tasks.tar.zst   <repo>/<task>.venv.json.zst + .delta.tar.zst + .manifest.json for every task  (extract at /opt/artifacts)
  <repo>/bundle.json            task list, uv interpreter dirs, sha256 + byte sizes of both bundles (pinned in the Dockerfile)

The packs are extracted once into tmpfs (/dev/shm; ~7.7 GB raw, 111k files), per-repo selections are hard links, so nothing
is copied and nothing lands on GPFS except the bundles themselves. Resumable: a repo with bundle.json is skipped.

  python3 harvest_bundle.py --work harvest --shm /dev/shm/bundle [--repos pandas,sympy,...] [--upload] [--force]
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

ORDER = ["pandas", "sympy", "pillow", "numpy", "orange3", "datalad", "coveragepy", "pyramid", "tornado", "scrapy", "aiohttp"]
HF_REPO = "laion/r2egym-build-artifacts"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def sha256_file(p, bufsize=1 << 22):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(bufsize), b""):
            h.update(chunk)
    return h.hexdigest()


def zstd_json(p):
    return json.loads(subprocess.run(["zstd", "-dc", p], capture_output=True, check=True).stdout)


def extract_packs(work, store):
    marker = os.path.join(store, ".packs_done")
    if os.path.exists(marker):
        log("packs already extracted at", store)
        return
    os.makedirs(store, exist_ok=True)
    packs = sorted(f for f in os.listdir(os.path.join(work, "packs")) if f.endswith(".tar.zst"))
    for i, pk in enumerate(packs, 1):
        t0 = time.time()
        subprocess.run(["tar", "-I", "zstd -T8 -d", "-xf", os.path.join(work, "packs", pk), "-C", store], check=True)
        log(f"extracted pack {i}/{len(packs)} {pk} in {time.time() - t0:.0f}s")
    n = sum(len(fs) for _, _, fs in os.walk(os.path.join(store, "blobs")))
    open(marker, "w").write(str(n))
    log("blob store ready:", n, "blobs")


def bundle_repo(work, store, shm, repo, upload, threads, group=None, subset=None):
    """One bundle for `repo`, or for the task subset `subset` written under <repo>/<group>/ (images capped at 5 GB
    on Daytona: pandas and orange3 must be split into contiguous id-sorted groups)."""
    rdir = os.path.join(work, repo)
    odir = os.path.join(rdir, group) if group else rdir
    os.makedirs(odir, exist_ok=True)
    out_json = os.path.join(odir, "bundle.json")
    mans = sorted(f for f in os.listdir(rdir) if f.endswith(".manifest.json"))
    tasks = [m[: -len(".manifest.json")] for m in mans]
    if subset is not None:
        tasks = [t for t in tasks if t in subset]
    if not tasks:
        log(repo, group, "no tasks, skipping")
        return
    tag = f"{repo}/{group}" if group else repo
    rel = f"{repo}/{group}" if group else repo
    t0 = time.time()
    shas, uvdirs, missing_files = set(), set(), []
    for t in tasks:
        man = json.load(open(os.path.join(rdir, t + ".manifest.json")))
        uvdirs.add(man["uvdir"])
        vj = zstd_json(os.path.join(rdir, t + ".venv.json.zst"))
        for rel, (sha, size, mode) in vj["files"].items():
            shas.add(sha)
        for name in (t + ".venv.json.zst", t + ".delta.tar.zst", t + ".manifest.json"):
            if not os.path.exists(os.path.join(rdir, name)):
                missing_files.append(name)
    assert not missing_files, f"{repo}: missing per-task files: {missing_files[:5]}"
    log(f"{tag}: {len(tasks)} tasks, {len(shas)} unique blobs, pythons {sorted(uvdirs)} (scan {time.time() - t0:.0f}s)")
    # staging via hard links
    sel = os.path.join(shm, "sel", tag.replace("/", "_"))
    shutil.rmtree(sel, ignore_errors=True)
    raw = 0
    missing = 0
    for sha in shas:
        src = os.path.join(store, "blobs", sha[:2], sha)
        dst_dir = os.path.join(sel, "blobs", sha[:2])
        os.makedirs(dst_dir, exist_ok=True)
        try:
            os.link(src, os.path.join(dst_dir, sha))
            raw += os.path.getsize(src)
        except FileNotFoundError:
            missing += 1
    assert missing == 0, f"{repo}: {missing} blobs missing from the store"
    blobs_tar = os.path.join(odir, "bundle-blobs.tar.zst")
    t1 = time.time()
    subprocess.run(["tar", "-I", f"zstd -T{threads} -3", "-cf", blobs_tar, "-C", sel, "blobs"], check=True)
    log(f"{tag}: bundle-blobs {os.path.getsize(blobs_tar) / 1e6:.0f} MB zst from {raw / 1e6:.0f} MB raw ({time.time() - t1:.0f}s)")
    shutil.rmtree(sel, ignore_errors=True)
    # tasks bundle (deltas are already zst; level 1 keeps it cheap)
    lst = os.path.join(shm, f"{tag.replace('/', '_')}.tasks.list")
    with open(lst, "w") as f:
        for t in tasks:
            for name in (t + ".venv.json.zst", t + ".delta.tar.zst", t + ".manifest.json"):
                f.write(f"{repo}/{name}\n")
    tasks_tar = os.path.join(odir, "bundle-tasks.tar.zst")
    t2 = time.time()
    subprocess.run(["tar", "-I", f"zstd -T{threads} -1", "-cf", tasks_tar, "-C", work, "-T", lst], check=True)
    log(f"{tag}: bundle-tasks {os.path.getsize(tasks_tar) / 1e6:.0f} MB ({time.time() - t2:.0f}s)")
    info = {
        "repo": repo, "group": group, "n_tasks": len(tasks), "tasks": tasks, "pythons": sorted(uvdirs),
        "blobs": {"file": f"{rel}/bundle-blobs.tar.zst", "sha256": sha256_file(blobs_tar), "bytes": os.path.getsize(blobs_tar),
                  "n_blobs": len(shas), "raw_bytes": raw},
        "tasks_bundle": {"file": f"{rel}/bundle-tasks.tar.zst", "sha256": sha256_file(tasks_tar), "bytes": os.path.getsize(tasks_tar)},
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    json.dump(info, open(out_json, "w"), indent=1)
    log(f"{tag}: bundle.json written; total {time.time() - t0:.0f}s")
    if upload:
        from huggingface_hub import HfApi
        api = HfApi()
        for local, remote in ((blobs_tar, f"{rel}/bundle-blobs.tar.zst"), (tasks_tar, f"{rel}/bundle-tasks.tar.zst"),
                              (out_json, f"{rel}/bundle.json")):
            t3 = time.time()
            for attempt in range(4):
                try:
                    api.upload_file(path_or_fileobj=local, path_in_repo=remote, repo_id=HF_REPO, repo_type="dataset",
                                    commit_message=f"bundle {remote}")
                    break
                except Exception as e:  # noqa: BLE001
                    log(f"{tag}: upload {remote} attempt {attempt + 1} failed: {type(e).__name__}: {str(e)[:120]}")
                    time.sleep(30 * (attempt + 1))
            else:
                raise SystemExit(f"{tag}: upload failed for {remote}")
            log(f"{tag}: uploaded {remote} ({os.path.getsize(local) / 1e6:.0f} MB, {time.time() - t3:.0f}s)")


# ---------------------------------------------------------------------------------------------------------------------
# Format v2: Daytona caps a snapshot at 5 GB. Raw venv blobs (pandas 3.7 GB, orange3 4.3 GB) plus per-task delta tarballs
# (pandas 2.7 GB) do not fit, and splitting by task does not help because the venvs differ per commit but overlap heavily.
# v2 stores every blob zstd-compressed (~0.29x) and content-addresses the build deltas too (pandas .so files dedupe 4.7x),
# so the materialiser decompresses instead of hard-linking (~1 s extra). Written under <repo>/v2/.
# ---------------------------------------------------------------------------------------------------------------------
def _zstd_compress_many(pairs, threads):
    """pairs = [(src_path, dst_path)]; compress each src into dst (.zst). zstandard module if available, else the CLI."""
    try:
        import threading
        import zstandard
        from concurrent.futures import ThreadPoolExecutor
        # ZstdCompressor instances are NOT thread safe (sharing one across threads segfaulted twice): one per thread
        tls = threading.local()
        def one(p):
            src, dst = p
            cctx = getattr(tls, "cctx", None)
            if cctx is None:
                cctx = tls.cctx = zstandard.ZstdCompressor(level=3)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            with open(src, "rb") as fi, open(dst, "wb") as fo:
                cctx.copy_stream(fi, fo)
        with ThreadPoolExecutor(threads) as ex:
            for _ in ex.map(one, pairs, chunksize=64):
                pass
    except ImportError:
        lst = "\n".join(f"{src}\t{dst}" for src, dst in pairs)
        for src, dst in pairs:
            os.makedirs(os.path.dirname(dst), exist_ok=True)
        subprocess.run(["bash", "-c", f"xargs -P{threads} -n2 -d '\n' sh -c 'zstd -3 -q -f -o \"$1\" \"$0\"' <<'EOL'\n" +
                        "\n".join(f"{src}\n{dst}" for src, dst in pairs) + "\nEOL"], check=True)


def bundle_repo_v2(work, store, shm, repo, upload, threads):
    rdir = os.path.join(work, repo)
    odir = os.path.join(rdir, "v2")
    os.makedirs(odir, exist_ok=True)
    rel = f"{repo}/v2"
    tasks = sorted(f[: -len(".manifest.json")] for f in os.listdir(rdir) if f.endswith(".manifest.json"))
    t0 = time.time()
    shas, uvdirs = set(), set()
    for t in tasks:
        man = json.load(open(os.path.join(rdir, t + ".manifest.json")))
        uvdirs.add(man["uvdir"])
        vj = zstd_json(os.path.join(rdir, t + ".venv.json.zst"))
        shas.update(sha for sha, _, _ in vj["files"].values())
    log(f"{rel}: {len(tasks)} tasks, {len(shas)} unique venv blobs (scan {time.time() - t0:.0f}s)")
    # deltas -> content-addressed files + per-task delta.json (mtimes kept)
    sel = os.path.join(shm, "sel", f"{repo}_v2")
    shutil.rmtree(sel, ignore_errors=True)
    dstore = os.path.join(shm, "dstore", repo)          # raw unique delta files, by sha
    shutil.rmtree(dstore, ignore_errors=True)
    os.makedirs(dstore, exist_ok=True)
    tmp = os.path.join(shm, "dtmp", repo)
    delta_shas = set()
    delta_raw = 0
    n_delta_files = 0
    for t in tasks:
        man = json.load(open(os.path.join(rdir, t + ".manifest.json")))
        dj = {"files": {}, "links": {}, "dirs": []}
        if man.get("delta_slim_zst"):
            shutil.rmtree(tmp, ignore_errors=True)
            os.makedirs(tmp)
            subprocess.run(["tar", "--zstd", "-xpf", os.path.join(rdir, t + ".delta.tar.zst"), "-C", tmp], check=True)
            for root, dirs, files in os.walk(tmp):
                r0 = os.path.relpath(root, tmp)
                if r0 != ".":
                    dj["dirs"].append(r0)
                for name in files:
                    p = os.path.join(root, name)
                    relp = os.path.normpath(os.path.join(r0, name))
                    if os.path.islink(p):
                        dj["links"][relp] = os.readlink(p)
                        continue
                    st = os.lstat(p)
                    h = hashlib.sha256()
                    with open(p, "rb") as f:
                        for chunk in iter(lambda: f.read(1 << 20), b""):
                            h.update(chunk)
                    sha = h.hexdigest()
                    dj["files"][relp] = [sha, st.st_size, oct(st.st_mode & 0o777), st.st_mtime_ns]
                    n_delta_files += 1
                    if sha not in delta_shas:
                        delta_shas.add(sha)
                        delta_raw += st.st_size
                        os.link(p, os.path.join(dstore, sha))
        json.dump(dj, open(os.path.join(odir, t + ".delta.json"), "w"))
    shutil.rmtree(tmp, ignore_errors=True)
    log(f"{rel}: deltas {n_delta_files} files -> {len(delta_shas)} unique ({delta_raw / 1e9:.2f} GB raw)")
    # compress venv blobs + delta blobs into the staging store
    pairs = []
    for sha in shas:
        pairs.append((os.path.join(store, "blobs", sha[:2], sha), os.path.join(sel, "blobs", sha[:2], sha + ".zst")))
    for sha in delta_shas:
        if sha not in shas:
            pairs.append((os.path.join(dstore, sha), os.path.join(sel, "blobs", sha[:2], sha + ".zst")))
    t1 = time.time()
    _zstd_compress_many(pairs, threads)
    zbytes = sum(os.path.getsize(d) for _, d in pairs)
    log(f"{rel}: compressed {len(pairs)} blobs -> {zbytes / 1e9:.2f} GB zst ({time.time() - t1:.0f}s)")
    blobs_tar = os.path.join(odir, "bundle-blobs.tar.zst")
    subprocess.run(["tar", "-I", f"zstd -T{threads} -1", "-cf", blobs_tar, "-C", sel, "blobs"], check=True)
    shutil.rmtree(sel, ignore_errors=True)
    shutil.rmtree(dstore, ignore_errors=True)
    lst = os.path.join(shm, f"{repo}_v2.tasks.list")
    with open(lst, "w") as f:
        for t in tasks:
            f.write(f"{repo}/{t}.venv.json.zst\n{repo}/{t}.manifest.json\n{repo}/v2/{t}.delta.json\n")
    tasks_tar = os.path.join(odir, "bundle-tasks.tar.zst")
    subprocess.run(["tar", "-I", f"zstd -T{threads} -1", "-cf", tasks_tar, "-C", work, "-T", lst], check=True)
    info = {
        "repo": repo, "format": "v2", "n_tasks": len(tasks), "tasks": tasks, "pythons": sorted(uvdirs),
        "blobs": {"file": f"{rel}/bundle-blobs.tar.zst", "sha256": sha256_file(blobs_tar), "bytes": os.path.getsize(blobs_tar),
                  "n_blobs": len(pairs), "raw_bytes": sum(os.path.getsize(s) for s, _ in pairs if os.path.exists(s)), "zst_bytes": zbytes},
        "tasks_bundle": {"file": f"{rel}/bundle-tasks.tar.zst", "sha256": sha256_file(tasks_tar), "bytes": os.path.getsize(tasks_tar)},
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    out_json = os.path.join(odir, "bundle.json")
    json.dump(info, open(out_json, "w"), indent=1)
    log(f"{rel}: bundle.json written (blobs {os.path.getsize(blobs_tar) / 1e6:.0f} MB, tasks {os.path.getsize(tasks_tar) / 1e6:.0f} MB); total {time.time() - t0:.0f}s")
    if upload:
        from huggingface_hub import HfApi
        api = HfApi()
        for local, remote in ((blobs_tar, f"{rel}/bundle-blobs.tar.zst"), (tasks_tar, f"{rel}/bundle-tasks.tar.zst"),
                              (out_json, f"{rel}/bundle.json")):
            t3 = time.time()
            for attempt in range(4):
                try:
                    api.upload_file(path_or_fileobj=local, path_in_repo=remote, repo_id=HF_REPO, repo_type="dataset",
                                    commit_message=f"bundle {remote}")
                    break
                except Exception as e:  # noqa: BLE001
                    log(f"{rel}: upload {remote} attempt {attempt + 1} failed: {type(e).__name__}: {str(e)[:120]}")
                    time.sleep(30 * (attempt + 1))
            else:
                raise SystemExit(f"{rel}: upload failed for {remote}")
            log(f"{rel}: uploaded {remote} ({os.path.getsize(local) / 1e6:.0f} MB, {time.time() - t3:.0f}s)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="harvest")
    ap.add_argument("--shm", default="/dev/shm/bundle")
    ap.add_argument("--repos", default=",".join(ORDER))
    ap.add_argument("--upload", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--format", default="v1", choices=["v1", "v2"], help="v2 = compressed content-addressed store incl. deltas (5 GB cap)")
    ap.add_argument("--split", default="", help="repo=k[,repo=k]: write k contiguous id-sorted groups under <repo>/g<i>/ "
                                                "instead of one bundle (Daytona snapshots are capped at 5 GB)")
    a = ap.parse_args()
    split = {kv.split("=")[0]: int(kv.split("=")[1]) for kv in a.split.split(",") if kv}
    work = os.path.abspath(a.work)
    store = os.path.join(a.shm, "store")
    os.makedirs(a.shm, exist_ok=True)
    extract_packs(work, store)
    for repo in [r for r in a.repos.split(",") if r]:
        if a.format == "v2":
            if not a.force and os.path.exists(os.path.join(work, repo, "v2", "bundle.json")):
                log(repo, "v2/bundle.json exists, skipping (use --force)")
                continue
            bundle_repo_v2(work, store, a.shm, repo, a.upload, a.threads)
            continue
        if repo in split:
            k = split[repo]
            tasks = sorted(f[: -len(".manifest.json")] for f in os.listdir(os.path.join(work, repo)) if f.endswith(".manifest.json"))
            per = -(-len(tasks) // k)
            for i in range(k):
                grp, subset = f"g{i}", set(tasks[i * per:(i + 1) * per])
                if not a.force and os.path.exists(os.path.join(work, repo, grp, "bundle.json")):
                    log(repo, grp, "bundle.json exists, skipping (use --force)")
                    continue
                bundle_repo(work, store, a.shm, repo, a.upload, a.threads, group=grp, subset=subset)
            continue
        if not a.force and os.path.exists(os.path.join(work, repo, "bundle.json")):
            log(repo, "bundle.json exists, skipping (use --force)")
            continue
        bundle_repo(work, store, a.shm, repo, a.upload, a.threads)
    log("ALL DONE")


if __name__ == "__main__":
    main()
