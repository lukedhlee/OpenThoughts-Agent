#!/usr/bin/env python3
"""Full artifact harvest of TaskTrove r2egym tasks from the JSC SIF cache (no image execution).

Generalises the pandas pilot (harvest_from_sif.sh + slim_and_uvpython.sh) to every repo and adds a content-addressed
layout so the per-repo Daytona image carries each thing once:

  pythons/<uvdir>.tar.zst                 one per uv-managed interpreter (root/.local/share/uv/python/<uvdir>)
  packs/<shard>-<seq>.tar.zst             content-addressed blob store: blobs/<sha[:2]>/<sha256> = every distinct venv file,
                                          once across all tasks/repos (extract every pack into one /opt/artifacts tree)
  blobs.index                             append-only "sha256<TAB>size" of blobs already packed (shards share it)
  <repo>/<task>.venv.json.zst             this task's /testbed/.venv as {files: {rel: [sha256, size, mode]}, links: {rel: target},
                                          dirs: [...]}; materialise by hard-linking blobs (≈20k links, <1 s)
  <repo>/<task>.delta.tar.zst             slim build delta: everything install.sh wrote in /testbed outside .venv, minus
                                          build/, __pycache__, *.pyc, generated *.c/*.cpp  (in-place .so, egg-info, ...)
  <repo>/<task>.tests.tar.zst             /r2e_tests + /testbed/run_tests.sh + install.sh
  <repo>/<task>.manifest.json             repo, task, image, base commit, python, dists, deleted files, sizes, sha256s

Runs one shard = a list of repos, sequentially, one SIF at a time (login-node safe: unsquashfs -p 2, zstd -T2, nice).
Resumable: a task with an existing manifest is skipped.

Usage: harvest_all.py --tsv harvest_all.tsv --work /p/scratch/synthlaion/lee27/harvest --repos sympy [--limit N]
  tsv columns: task<TAB>image<TAB>sifhash<TAB>repo<TAB>commit   (from the Mac-side builder in this session)
"""
import argparse
import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
import time

SIF_CACHE = "/p/scratch/synthlaion/lee27/r2egym_sif"
SLIM_EXCLUDE_PREFIX = ("build/",)
SLIM_EXCLUDE_SUFFIX = (".pyc", ".c", ".cpp")
GIT = ["git", "-c", "safe.directory=*"]
KNOWN = set()          # sha256 of blobs already stored (loaded from blobs.index at start, grown as we go)
PENDING = ""           # <work>/tmp/pending_<shard>/ : blobs not yet packed
PENDING_BYTES = [0]
PACK_BYTES = 300 * 1024 * 1024
SHARD = ""
PACK_SEQ = [0]
ENV = dict(os.environ, GIT_CONFIG_GLOBAL="/dev/null", GIT_CONFIG_NOSYSTEM="1", ZSTD_NBTHREADS="2")


def sh(cmd, cwd=None, check=True, capture=True):
    r = subprocess.run(cmd, cwd=cwd, env=ENV, text=True, capture_output=capture)
    if check and r.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)} -> {r.returncode}: {(r.stderr or '')[-400:]}")
    return r.stdout if capture else ""


def glob_pending(pending):
    out = []
    for dp, _, fns in os.walk(pending):
        out += [f for f in fns if not f.endswith(".tmp")]
    return out


def sha256_file(path, buf=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(buf)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sif_offset(sif):
    out = sh(["apptainer", "sif", "list", sif])
    for line in out.splitlines():
        if "|FS" in line.replace(" ", "") or "Squashfs" in line:
            cols = [c.strip() for c in line.split("|")]
            for c in cols:
                if "-" in c and c.split("-")[0].isdigit():
                    return int(c.split("-")[0])
    raise RuntimeError(f"no squashfs partition in {sif}")


def walk_manifest(root):
    """{relpath: [sha256, size, mode]} for files, ["link", target] for symlinks. relpath relative to root's parent."""
    man = {}
    base = os.path.dirname(root.rstrip("/"))
    for dp, dns, fns in os.walk(root):
        if "__pycache__" in dp.split(os.sep):
            continue  # bytecode differs by build timestamp only; not stored (regenerated on first import)
        for n in fns + [d for d in dns if os.path.islink(os.path.join(dp, d))]:
            p = os.path.join(dp, n)
            rel = os.path.relpath(p, base)
            st = os.lstat(p)
            if stat.S_ISLNK(st.st_mode):
                man[rel] = ["link", os.readlink(p)]
            elif stat.S_ISREG(st.st_mode):
                man[rel] = [sha256_file(p), st.st_size, oct(st.st_mode & 0o777)]
    return man


def tar_zst(out, cwd, members, list_file=None):
    tmp = out + ".tmp"
    cmd = ["tar", "--zstd", "--numeric-owner", "-cf", tmp]
    if list_file:
        cmd += ["-T", list_file]
    else:
        cmd += members
    sh(cmd, cwd=cwd)
    os.replace(tmp, out)
    return os.path.getsize(out)


def flush_pack(W):
    """tar.zst the pending blobs into packs/<shard>-<seq>.tar.zst and drop the raw copies (keeps inodes low)."""
    if not os.path.isdir(PENDING) or not os.listdir(PENDING):
        return
    os.makedirs(os.path.join(W, "packs"), exist_ok=True)
    while True:
        out = os.path.join(W, "packs", f"{SHARD}-{PACK_SEQ[0]:04d}.tar.zst")
        if not os.path.exists(out):
            break
        PACK_SEQ[0] += 1
    tmp = out + ".tmp"
    sh(["tar", "--zstd", "--numeric-owner", "-cf", tmp, "--transform", "s#^\\.#blobs#", "."], cwd=PENDING)
    os.replace(tmp, out)
    shutil.rmtree(PENDING, ignore_errors=True)
    os.makedirs(PENDING, exist_ok=True)
    PENDING_BYTES[0] = 0
    PACK_SEQ[0] += 1
    return out


def harvest_task(task, image, h, repo, W, log):
    t0 = time.time()
    sifs = [f for f in os.listdir(SIF_CACHE) if f.endswith(f"-{h}.sif")]
    if not sifs:
        raise RuntimeError(f"no SIF for hash {h}")
    sif = os.path.realpath(os.path.join(SIF_CACHE, sifs[0]))
    x = os.path.join(W, "tmp", f"x_{task}")
    shutil.rmtree(x, ignore_errors=True)
    off = sif_offset(sif)
    r = subprocess.run(["nice", "-n", "10", "unsquashfs", "-q", "-n", "-no-xattrs", "-p", "2", "-o", str(off), "-d", x, sif,
                        "testbed", "root/.local/share/uv/python", "r2e_tests"], env=ENV, text=True, capture_output=True)
    t1 = time.time()
    tb = os.path.join(x, "testbed")
    venv = os.path.join(tb, ".venv")
    if not os.path.isfile(os.path.join(venv, "pyvenv.cfg")):
        raise RuntimeError(f"no .venv/pyvenv.cfg (unsquashfs rc={r.returncode}: {r.stderr[-300:]})")
    cfg = dict(l.split("=", 1) for l in open(os.path.join(venv, "pyvenv.cfg")) if "=" in l)
    cfg = {k.strip(): v.strip() for k, v in cfg.items()}
    home = cfg.get("home", "")  # /root/.local/share/uv/python/<uvdir>/bin
    uvdir = os.path.basename(os.path.dirname(home)) if "/uv/python/" in home else ""
    pyver = cfg.get("version", cfg.get("version_info", ""))
    head = sh(GIT + ["rev-parse", "HEAD"], cwd=tb).strip()

    os.makedirs(os.path.join(W, repo), exist_ok=True)
    # 1. uv python (once per interpreter, shared across repos; atomic rename so parallel shards are safe)
    pdir = os.path.join(W, "pythons")
    os.makedirs(pdir, exist_ok=True)
    pytar = os.path.join(pdir, f"{uvdir}.tar.zst")
    if uvdir and not os.path.exists(pytar):
        rel = os.path.join("root/.local/share/uv/python", uvdir)
        if os.path.isdir(os.path.join(x, rel)):
            tar_zst(pytar, x, [rel])
            json.dump(walk_manifest(os.path.join(x, rel)), open(pytar + ".manifest.json", "w"))
            log(f"  new python {uvdir} {os.path.getsize(pytar)//1e6:.0f} MB")

    # 2. venv -> content-addressed blobs (deduplicated across every task and repo) + per-task venv manifest
    vman = walk_manifest(venv)  # paths like .venv/...
    dirs = sorted(os.path.relpath(dp, tb) for dp, _, _ in os.walk(venv))
    new_blobs = new_bytes = 0
    for rel, v in vman.items():
        if v[0] == "link" or v[0] in KNOWN:
            continue
        dst = os.path.join(PENDING, v[0][:2], v[0])
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(os.path.join(tb, rel), dst + ".tmp")
        os.chmod(dst + ".tmp", int(v[2], 8))
        os.replace(dst + ".tmp", dst)
        KNOWN.add(v[0]); new_blobs += 1; new_bytes += v[1]
    with open(os.path.join(W, "blobs.index"), "a") as f:
        f.write("".join(f"{v[0]}\t{v[1]}\n" for v in vman.values() if v[0] != "link"))
    PENDING_BYTES[0] += new_bytes
    if PENDING_BYTES[0] > PACK_BYTES:
        flush_pack(W)
    vj = {"root": ".venv", "uvdir": uvdir, "python": pyver,
          "files": {k: v for k, v in vman.items() if v[0] != "link"}, "links": {k: v[1] for k, v in vman.items() if v[0] == "link"},
          "dirs": dirs}
    vpath = os.path.join(W, repo, f"{task}.venv.json.zst")
    p_json = os.path.join(W, "tmp", f"{task}.venv.json"); json.dump(vj, open(p_json, "w"))
    sh(["zstd", "-q", "-f", "-T2", "-o", vpath, p_json]); os.remove(p_json)
    sps = [d for d in os.listdir(os.path.join(venv, "lib")) if d.startswith("python")] if os.path.isdir(os.path.join(venv, "lib")) else []
    sp = os.path.join(venv, "lib", sps[0], "site-packages") if sps else venv
    dists = sorted(d for d in os.listdir(sp) if d.endswith((".dist-info", ".egg-info", ".egg-link")))
    t2 = time.time()

    # 3. slim build delta outside .venv: untracked+ignored (git ls-files --others) + modified tracked, minus noise
    others = sh(GIT + ["ls-files", "--others", "-z"], cwd=tb).split("\0")
    modified = sh(GIT + ["diff", "--name-only", "HEAD"], cwd=tb).splitlines()
    deleted = sh(GIT + ["ls-files", "--deleted"], cwd=tb).splitlines()
    full = sorted({p for p in others + modified if p and not p.startswith(".venv/")})
    slim = [p for p in full if not p.startswith(SLIM_EXCLUDE_PREFIX) and "__pycache__/" not in p and not p.endswith(SLIM_EXCLUDE_SUFFIX)]
    slim = [p for p in slim if os.path.lexists(os.path.join(tb, p))]
    so = [p for p in slim if ".so" in os.path.basename(p)]
    sizes = {p: os.lstat(os.path.join(tb, p)).st_size for p in slim}
    os.makedirs(os.path.join(W, repo), exist_ok=True)
    lst = os.path.join(W, "tmp", f"{task}.delta.list")
    open(lst, "w").write("\n".join(slim) + "\n")
    dtar = os.path.join(W, repo, f"{task}.delta.tar.zst")
    delta_zst = tar_zst(dtar, tb, [], list_file=lst) if slim else 0
    os.remove(lst)
    # 4. tests
    members = [m for m in ("r2e_tests", "testbed/run_tests.sh", "testbed/install.sh") if os.path.exists(os.path.join(x, m))]
    ttar = os.path.join(W, repo, f"{task}.tests.tar.zst")
    tests_zst = tar_zst(ttar, x, members) if members else 0
    t3 = time.time()
    man = {
        "task": task, "repo": repo, "image": image, "sif": os.path.basename(sif), "base_commit": head,
        "uvdir": uvdir, "python": pyver, "n_dists": len(dists), "dists": dists,
        "venv_files": len(vman), "venv_bytes": sum(v[1] for v in vman.values() if v[0] != "link"),
        "venv_new_blobs": new_blobs, "venv_new_bytes": new_bytes, "venv_json_zst": os.path.getsize(vpath),
        "delta_full_files": len(full), "delta_slim_files": len(slim), "delta_slim_bytes": sum(sizes.values()),
        "delta_slim_zst": delta_zst, "so_files": len(so), "so_bytes": sum(sizes[p] for p in so), "so_list": so[:50],
        "deleted": deleted, "tests_zst": tests_zst,
        "sha256": {os.path.basename(p): sha256_file(p) for p in (dtar, ttar) if os.path.exists(p) and os.path.getsize(p)},
        "t_extract": round(t1 - t0, 1), "t_venv": round(t2 - t1, 1), "t_delta": round(t3 - t2, 1),
    }
    json.dump(man, open(os.path.join(W, repo, f"{task}.manifest.json"), "w"))
    shutil.rmtree(x, ignore_errors=True)
    return man


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--repos", required=True, help="comma list; processed in order")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", default=None, help="name used for pack files (default: first repo)")
    ap.add_argument("--reverse", action="store_true", help="walk the task list backwards (a second worker on the same repo)")
    a = ap.parse_args()
    W = os.path.abspath(a.work)
    os.makedirs(os.path.join(W, "tmp"), exist_ok=True)
    repos = a.repos.split(",")
    global PENDING, SHARD
    SHARD = a.shard or repos[0]
    PENDING = os.path.join(W, "tmp", f"pending_{SHARD}")
    os.makedirs(PENDING, exist_ok=True)
    if os.path.exists(os.path.join(W, "blobs.index")):
        for line in open(os.path.join(W, "blobs.index")):
            KNOWN.add(line.split("\t", 1)[0])
    # blobs left in a pending dir from a crashed run are not in any pack: forget them so they get re-copied
    for f in glob_pending(PENDING):
        KNOWN.discard(f)
    rows = [l.rstrip("\n").split("\t") for l in open(a.tsv) if l.strip()]
    rows = [r for r in rows if r[3] in repos]
    rows.sort(key=lambda r: (repos.index(r[3]), r[0]))
    if a.reverse:
        rows.reverse()
    if a.limit:
        rows = rows[: a.limit]
    logf = open(os.path.join(W, f"harvest_{'_'.join(repos)}.log"), "a")

    def log(msg):
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        logf.write(line + "\n"); logf.flush()

    log(f"START shard repos={repos} tasks={len(rows)}")
    done = fail = 0
    for i, (task, image, h, repo, commit) in enumerate(rows, 1):
        if os.path.exists(os.path.join(W, repo, f"{task}.manifest.json")):
            done += 1
            continue
        try:
            m = harvest_task(task, image, h, repo, W, log)
            done += 1
            log(f"[{i}/{len(rows)}] {repo} {task} py={m['python']} venv={m['venv_files']}f/+{m['venv_new_blobs']}new/{m['venv_new_bytes']//1024}KB "
                f"delta={m['delta_slim_files']}f/{m['delta_slim_zst']//1024}KB so={m['so_files']} del={len(m['deleted'])} "
                f"t={m['t_extract']}/{m['t_venv']}/{m['t_delta']}s")
        except Exception as e:  # noqa: BLE001
            fail += 1
            log(f"[{i}/{len(rows)}] FAIL {repo} {task}: {type(e).__name__}: {str(e)[:300]}")
            with open(os.path.join(W, "failures.tsv"), "a") as f:
                f.write(f"{repo}\t{task}\t{h}\t{type(e).__name__}\t{str(e)[:300]}\n")
            shutil.rmtree(os.path.join(W, "tmp", f"x_{task}"), ignore_errors=True)
        if i % 10 == 0:
            load = os.getloadavg()[0]
            npid = int(sh(["bash", "-c", "ps -u $USER | wc -l"]).strip())
            if load > 55 or npid > 2500:
                log(f"PAUSE load={load:.1f} pids={npid}")
                while os.getloadavg()[0] > 48 or int(sh(["bash", "-c", "ps -u $USER | wc -l"]).strip()) > 2500:
                    time.sleep(60)
                log("RESUME")
    out = flush_pack(W)
    log(f"SHARD_DONE done={done} fail={fail} last_pack={out}")


if __name__ == "__main__":
    main()
