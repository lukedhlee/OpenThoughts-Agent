#!/usr/bin/env python3
"""Fidelity spot check: does a CalibForge task's Daytona sandbox, after setup.sh, hold the task image's filesystem?

For each task: start a sandbox from the task's base snapshot (Daytona SDK, labelled cf-fidelity, always deleted),
record which packages harbor's agent tooling added or upgraded, run the task's setup_files/setup.sh the way harbor
does (root, cwd /app, the task's env), then list every path with type, mode, owner, size, link target and sha256.
The expected view is the original image built from its registry layers on this machine (whiteouts applied). Every
path of the image must be present with identical content and metadata; differences are sorted into
  tooling   paths of packages the agent tooling installed or upgraded (tmux, asciinema, python3 on ubuntu, ...)
  pkgmeta   dpkg/apt/debconf/ldconfig bookkeeping those packages touch
  runtime   /etc/hosts, /etc/hostname, /etc/resolv.conf
  other     anything else (should be empty)

  python fidelity_check.py --tree <tree> --tasks a,b,c --out <dir> [--key-file ~/.config/otagent/daytona_eval.env]
"""
from __future__ import annotations

import argparse
import asyncio
import collections
import hashlib
import json
import sys
import tarfile
import time
import tomllib
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from build_snapshots import DEFAULT_KEY_FILE, load_secret  # noqa: E402
from registry import Hub  # noqa: E402

RUNTIME = {"/etc/hostname", "/etc/hosts", "/etc/resolv.conf", "/.dockerenv", "/etc/mtab", "/tmp/cf_pre.sh", "/tmp/cf_list.sh"}
# what the Daytona runtime puts into every sandbox (its daemon, host kernel headers/modules, container-runtime dirs)
RUNTIME_PREFIX = ("/usr/local/bin/daytona", "/usr/local/lib/daytona-computer-use", "/root/.daytona", "/tmp/daytona",
                  "/usr/lib/modules", "/usr/src/linux-", "/var/lib/rancher", "/var/lib/containerd",
                  "/var/lib/kubelet", "/var/lib/k0s", "/var/lib/buildkit", "/var/lib/docker")
# written by the agent tooling's maintainer scripts rather than shipped in its packages (so not in dpkg's .list files)
TOOLING_GENERATED = ("/etc/ssl/certs", "/usr/share/ca-certificates", "/etc/python3", "/usr/share/python3",
                     "/var/lib/python", "/usr/local/lib/python3", "/etc/shells", "/var/lib/shells.state",
                     "/etc/ld.so.cache", "/etc/localtime", "/etc/timezone")
USRMERGE = ("/bin/", "/sbin/", "/lib/", "/lib64/", "/lib32/", "/libx32/")
PKGMETA = ("/var/lib/dpkg/", "/var/cache/debconf/", "/var/log/", "/var/cache/ldconfig/", "/etc/ld.so.cache",
           "/var/lib/apt/", "/var/cache/apt/", "/var/lib/systemd/deb-systemd", "/etc/systemd/system/")
SKIP_PREFIX = ("/proc/", "/sys/", "/dev/", "/setup_files", "/logs", "/tmp/cf_fid")
LIST_SH = r"""set -e
mkdir -p /tmp/cf_fid
find / -xdev \( -path /proc -o -path /sys -o -path /dev -o -path /tmp/cf_fid -o -path /setup_files -o -path /logs \) -prune \
  -o -printf '%y\t%m\t%U\t%G\t%s\t%p\t%l\n' > /tmp/cf_fid/list.tsv
find / -xdev \( -path /proc -o -path /sys -o -path /dev -o -path /tmp/cf_fid -o -path /setup_files -o -path /logs \) -prune \
  -o -type f -print0 | xargs -0 -r sha256sum > /tmp/cf_fid/sums.txt 2>/dev/null || true
dpkg-query -W -f='${Package}\t${Version}\n' > /tmp/cf_fid/dpkg.tsv 2>/dev/null || true
env > /tmp/cf_fid/env.txt
for f in /var/lib/dpkg/info/*.list; do b=${f##*/}; b=${b%.list}; b=${b%%:*}; sed "s|^|$b\t|" "$f"; done > /tmp/cf_fid/owned.tsv
cd /tmp/cf_fid && tar czf /tmp/cf_fid/out.tgz list.tsv sums.txt dpkg.tsv env.txt owned.tsv
"""
PRE_SH = r"""dpkg-query -W -f='${Package}\t${Version}\t${binary:Package}\n'
"""


def norm(name: str) -> str:
    name = name[2:] if name.startswith("./") else name
    return "/" + name.rstrip("/")


def image_view(hub: Hub, ref: str, cache: Path) -> tuple[dict, dict]:
    """path -> (type, mode, uid, gid, size, sha256|linkname) of the image, plus its dpkg status versions."""
    repo = ref.split("@")[0]
    repo = repo if "/" in repo else f"library/{repo}"
    m = hub.manifest(f"{repo}@{ref.split('@')[1]}", cache)
    view: dict[str, tuple] = {}
    status = b""
    for layer in m["layers"]:
        blob = hub.blob_to(repo, layer["digest"], cache / "blobs" / layer["digest"].split(":")[1])
        entries, wh, opq = {}, [], []
        with tarfile.open(blob, "r:gz") as tf:
            for mem in tf:
                p = norm(mem.name)
                base = p.rsplit("/", 1)[-1]
                if base == ".wh..wh..opq":
                    opq.append(p.rsplit("/", 1)[0] or "/")
                    continue
                if base.startswith(".wh."):
                    wh.append(p.rsplit("/", 1)[0] + "/" + base[4:])
                    continue
                meta = (mem.mode & 0o7777, mem.uid, mem.gid)
                if mem.isfile():
                    data = tf.extractfile(mem).read()
                    if p == "/var/lib/dpkg/status":
                        status = data
                    entries[p] = ("f", *meta, mem.size, hashlib.sha256(data).hexdigest())
                elif mem.islnk():
                    tgt = norm(mem.linkname)
                    src = entries.get(tgt) or view.get(tgt)
                    entries[p] = ("f", *meta, src[4] if src else None, src[5] if src else None)
                elif mem.issym():
                    entries[p] = ("l", *meta, None, mem.linkname)
                elif mem.isdir():
                    entries[p] = ("d", *meta, None, None)
                else:
                    entries[p] = ("o", *meta, None, None)
        for d in opq:
            for k in [k for k in view if k.startswith(d.rstrip("/") + "/")]:
                del view[k]
        for w in wh:
            for k in [k for k in view if k == w or k.startswith(w + "/")]:
                del view[k]
        view.update(entries)
    versions = {}
    for stanza in status.decode(errors="replace").split("\n\n"):
        f = dict(line.split(": ", 1) for line in stanza.splitlines() if ": " in line and not line.startswith(" "))
        if "Package" in f and "installed" in f.get("Status", ""):
            versions[f["Package"]] = f.get("Version")
    return view, versions


def parse_actual(files: dict[str, bytes]) -> dict:
    sums = {}
    for line in files["sums.txt"].decode(errors="replace").splitlines():
        h, _, p = line.partition("  ")
        sums[p] = h
    view = {}
    for line in files["list.tsv"].decode(errors="replace").splitlines():
        parts = line.split("\t")
        if len(parts) < 7:
            continue
        y, mode, uid, gid, size, p, link = parts[:7]
        p = "/" if p == "/" else p.rstrip("/")
        t = {"f": "f", "d": "d", "l": "l"}.get(y, "o")
        view[p] = (t, int(mode, 8), int(uid), int(gid), int(size) if t == "f" else None,
                   sums.get(p) if t == "f" else (link if t == "l" else None))
    return view


def classify(p: str, tooling: set[str], entry: tuple | None = None) -> str:
    if p in RUNTIME or p.startswith(RUNTIME_PREFIX):
        return "runtime"
    if p in tooling or p.startswith(TOOLING_GENERATED) or "/__pycache__" in p:
        return "tooling"
    if entry and entry[0] == "l" and entry[5] and not entry[5].startswith("/"):
        # ldconfig's soname links next to a tooling library
        if f"{p.rsplit('/', 1)[0]}/{entry[5]}" in tooling:
            return "tooling"
    if p.startswith(PKGMETA) or p in {"/var/lib/dpkg", "/var/cache/debconf", "/var/log"}:
        return "pkgmeta"
    return "other"


async def sandbox_side(d, snapshot: str, task_dir: Path, env: dict, out: Path) -> dict:
    from daytona import CreateSandboxFromSnapshotParams
    t0 = time.monotonic()
    sb = await d.create(CreateSandboxFromSnapshotParams(snapshot=snapshot, labels={"purpose": "cf-fidelity"},
                                                        auto_delete_interval=60), timeout=600)
    t_start = time.monotonic() - t0
    try:
        await sb.fs.upload_file(PRE_SH.encode(), "/tmp/cf_pre.sh")
        await sb.fs.upload_file(LIST_SH.encode(), "/tmp/cf_list.sh")
        pre = await sb.process.exec("bash /tmp/cf_pre.sh", timeout=120)
        pre_rows = [l.split("\t") for l in pre.result.splitlines() if l.count("\t") == 2]
        lists = {}
        for pkg, _ver, binpkg in pre_rows:
            lists[pkg] = binpkg
        await sb.fs.upload_file((task_dir / "setup_files" / "setup.sh").read_bytes(), "/setup_files/setup.sh")
        t1 = time.monotonic()
        r = await sb.process.exec("bash /setup_files/setup.sh", cwd="/app", env=env or None, timeout=1800)
        setup_s = time.monotonic() - t1
        (out / "setup.log").write_text(r.result or "")
        if r.exit_code != 0:
            raise RuntimeError(f"setup.sh exit {r.exit_code}: {(r.result or '')[-800:]}")
        r = await sb.process.exec("bash /tmp/cf_list.sh", timeout=1200)
        if r.exit_code != 0:
            raise RuntimeError(f"listing failed: {(r.result or '')[-500:]}")
        (out / "actual.tgz").write_bytes(await sb.fs.download_file("/tmp/cf_fid/out.tgz"))
        return {"pre": {p: v for p, v, _ in pre_rows}, "start_s": round(t_start, 1), "setup_s": round(setup_s, 1)}
    finally:
        await d.delete(sb)


async def check(task: str, tree: Path, base_label: str, snapshot: str, base_image: str, d, hub, cache, outdir) -> dict:
    out = outdir / task
    out.mkdir(parents=True, exist_ok=True)
    tdir = tree / task
    cfg = tomllib.loads((tdir / "task.toml").read_text())
    env = cfg["environment"].get("env", {})
    image = cfg["metadata"]["calibforge_image"]
    side = await sandbox_side(d, snapshot, tdir, env, out)
    with tarfile.open(out / "actual.tgz") as tf:
        files = {m.name: tf.extractfile(m).read() for m in tf if m.isfile()}
    actual = parse_actual(files)
    expected, exp_versions = await asyncio.to_thread(image_view, hub, image, cache)
    _, base_versions = await asyncio.to_thread(image_view, hub, base_image, cache)
    # tooling packages: installed in the snapshot but absent from the base, or at a different version
    tooling_pkgs = {p for p, v in side["pre"].items() if base_versions.get(p) != v}
    tooling = set()
    for line in files["owned.tsv"].decode(errors="replace").splitlines():
        pkg, _, path = line.partition("\t")
        if pkg in tooling_pkgs:
            path = path.rstrip("/") or "/"
            tooling.add(path)
            if path.startswith(USRMERGE):
                tooling.add("/usr" + path)
    diffs = collections.defaultdict(list)
    missing = []
    for p, e in expected.items():
        if p.startswith(SKIP_PREFIX) or p in ("/proc", "/sys", "/dev", "/logs"):
            continue
        a = actual.get(p)
        if a is None:
            (diffs[classify(p, tooling)] if classify(p, tooling) != "other" else missing).append(p)
            continue
        if e[0] != a[0] or (e[0] in "fl" and e[5] != a[5]) or e[1:4] != a[1:4]:
            what = "content" if e[0] == a[0] and e[5] != a[5] else ("type" if e[0] != a[0] else "meta")
            diffs[classify(p, tooling)].append(f"{what}:{p}")
    extra = collections.Counter()
    extra_other = []
    for p in actual:
        if p not in expected and not p.startswith(SKIP_PREFIX) and p not in ("/", "/proc", "/sys", "/dev"):
            c = classify(p, tooling, actual[p])
            extra[c] += 1
            if c == "other":
                extra_other.append(p)
    act_versions = dict(l.split("\t", 1) for l in files["dpkg.tsv"].decode().splitlines() if "\t" in l)
    pkg_missing = sorted(p for p, v in exp_versions.items() if act_versions.get(p) != v and p not in tooling_pkgs)
    res = {
        "task": task, "base": base_label, "image": image, "start_s": side["start_s"], "setup_s": side["setup_s"],
        "expected_paths": len(expected), "actual_paths": len(actual),
        "missing": len(missing), "missing_sample": missing[:15],
        "differ": {k: len(v) for k, v in diffs.items()},
        "differ_other_sample": diffs.get("other", [])[:15],
        "extra": dict(extra), "extra_other_sample": sorted(extra_other)[:25],
        "image_packages": len(exp_versions), "image_packages_not_matching": pkg_missing[:20],
        "tooling_packages": sorted(tooling_pkgs)[:60],
        "env_path": next((l for l in files["env.txt"].decode().splitlines() if l.startswith("PATH=")), None),
    }
    (out / "result.json").write_text(json.dumps(res, indent=1))
    return res


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tree", required=True)
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache", default=str(Path("~/.local/share/otagent/calibforge-work").expanduser()))
    ap.add_argument("--key-file", default=DEFAULT_KEY_FILE)
    ap.add_argument("--parallel", type=int, default=3)
    a = ap.parse_args()
    tree, outdir, cache = Path(a.tree), Path(a.out), Path(a.cache)
    pool = json.loads((tree / "pool.json").read_text())
    hub = Hub()
    from daytona import AsyncDaytona, DaytonaConfig
    sem = asyncio.Semaphore(a.parallel)
    results = []
    async with AsyncDaytona(DaytonaConfig(api_key=load_secret("DAYTONA_API_KEY", a.key_file))) as d:
        async def one(task):
            async with sem:
                label = tomllib.loads((tree / task / "task.toml").read_text())["metadata"]["calibforge_daytona_base"]
                try:
                    r = await check(task, tree, label, pool[label]["name"], pool[label]["base_image"], d, hub, cache, outdir)
                except Exception as exc:  # noqa: BLE001
                    r = {"task": task, "base": label, "error": f"{type(exc).__name__}: {exc}"[:1500]}
                print(json.dumps({k: r.get(k) for k in ("task", "base", "setup_s", "missing", "differ", "extra", "error")}),
                      flush=True)
                results.append(r)
        await asyncio.gather(*(one(t) for t in a.tasks.split(",") if t))
    (outdir / "summary.json").write_text(json.dumps(results, indent=1))
    return 0 if all(not r.get("error") and r["missing"] == 0 and not r["differ"].get("other") for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
