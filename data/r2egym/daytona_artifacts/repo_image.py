#!/usr/bin/env python3
"""One Daytona image per R2E-Gym repo, carrying every task commit's harvested build artifacts (bundle layout).

The image is R2E-Gym's own per-repo Dockerfile prefix (ubuntu:22.04, that repo's apt packages, uv, a full clone of the
repo at /testbed) plus, pulled from the public HF dataset at build time and pinned by sha256 in the Dockerfile itself:
  pythons/<uvdir>.tar.zst              the uv-managed interpreters that repo's venvs use, untarred at /
  tools/harvest_materialize.py         the per-task materialiser (stdlib; runs on the system python3)
  <repo>/bundle-blobs.tar.zst          that repo's content-addressed venv files -> /opt/artifacts/blobs/<sha[:2]>/<sha>
  <repo>/bundle-tasks.tar.zst          <repo>/<task>.venv.json.zst + .delta.tar.zst + .manifest.json -> /opt/artifacts/<repo>/
Nothing task-specific is materialised at build time; the per-task `setup_files/setup.sh` written by tt_daytona_tree_v3.py
checks out the base commit, hard-links the venv from the blob store into /opt/venvs/<task> (symlinked as /testbed/.venv),
untars the commit's build delta and applies install.sh's deletions, in about a second and with no network.

The Dockerfile is deterministic for a given (repo, bundle): harbor hashes the environment dir, so the same bundle always
maps to the same `harbor__<hash>__snapshot`, and a purged snapshot rebuilds from this file alone (bundle sha256s pinned).
Layout produced by harvest_all.py + harvest_bundle.py (see the 2026-09-14 harvest note).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
R2E_DOCKERFILES = HERE / "r2e_dockerfiles"
TOOL = HERE / "harvest_materialize.py"
# content-addressed remote name: an updated materialiser never changes the Dockerfile (and snapshot hash) of images already built
TOOL_REMOTE = "tools/harvest_materialize.py"   # the first (v1) upload, referenced by the images built on 2026-09-15 01:00 UTC
# The materialiser each built image pins (remote name, sha256), so regenerating the tree after the local materialiser changes
# reproduces those images' Dockerfiles and snapshot hashes; a repo not listed gets the current copy.
_V1 = ("tools/harvest_materialize.py", "93da240fe0bd08756f2e6e57f1ea39611bf22b98e9cd2e5fd1032cff166a8404")
_V2 = ("tools/harvest_materialize-b27433cf48e8.py", "b27433cf48e81c55fa11e1dfc0d2dc5f10975ec03746cd70d54c2b2d954ee15c")
BUILT_TOOL_PINS = {r: _V1 for r in ("sympy", "pillow", "pyramid", "tornado", "numpy", "scrapy", "datalad", "aiohttp",
                                    "coveragepy")} | {"pandas": _V2, "orange3": _V2}

GITHUB = {
    "sympy": "sympy/sympy", "tornado": "tornadoweb/tornado", "scrapy": "scrapy/scrapy", "pyramid": "Pylons/pyramid",
    "datalad": "datalad/datalad", "coveragepy": "nedbat/coveragepy", "aiohttp": "aio-libs/aiohttp", "moto": "getmoto/moto",
    "orange3": "biolab/orange3", "pandas": "pandas-dev/pandas", "numpy": "numpy/numpy", "pillow": "python-pillow/Pillow",
}
MODULE = {"coveragepy": "coverage", "orange3": "Orange", "pillow": "PIL"}
# R2E-Gym's own workdir where it differs from TaskTrove's /testbed (the harvested venv's editable install points there).
R2E_WORKDIR = {"sympy": "/sympy"}
# Compressed size of the prefix layers (ubuntu + apt + uv + full clone), from the Docker Hub layer sizes measured
# on 2026-09-14: pandas 620 MB, numpy 490, pillow 500, sympy 420; used only for the SNAPSHOTS.md estimate.
BASE_MB = {"pandas": 620, "numpy": 490, "pillow": 500, "sympy": 420, "orange3": 700, "aiohttp": 550, "datalad": 550,
           "scrapy": 450, "tornado": 400, "pyramid": 400, "coveragepy": 400}


@dataclass
class Artifact:
    task: str
    repo: str
    base: str                 # commit checked out in /testbed before the agent starts
    python: str = ""          # e.g. 3.8.20
    uvdir: str = ""           # uv-managed interpreter dir; "" = the image's system /usr/bin/python3.10 (ubuntu 22.04)
    deleted: list[str] = field(default_factory=list)   # tracked files install.sh removed (e.g. pyproject.toml)
    has_delta: bool = True
    workdir: str = "/testbed"
    pytest: str = ""          # the harvested venv's pytest version, e.g. 4.6.11


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class ArtifactIndex:
    """Task -> Artifact plus the per-repo bundle records, for the bundle-layout artifact dataset."""

    def __init__(self, hf_repo: str, repo_of: dict[str, str] | None = None, repos: list[str] | None = None):
        from huggingface_hub import HfApi, hf_hub_download, snapshot_download
        self.hf_repo = hf_repo
        api = HfApi()
        info = api.dataset_info(hf_repo, files_metadata=True)
        self.sizes = {s.rfilename: (s.size or 0) for s in info.siblings}
        files = set(self.sizes)
        self.bundles: dict[str, dict] = {}
        for f in sorted(files):
            if f.endswith("/bundle.json"):
                parts = f.split("/")
                repo = parts[0]
                if repos and repo not in repos:
                    continue
                if len(parts) == 3 and parts[1] == "v2":          # <repo>/v2/bundle.json wins over <repo>/bundle.json
                    self.bundles[repo] = json.load(open(hf_hub_download(hf_repo, f, repo_type="dataset", force_download=True)))
                elif len(parts) == 2 and repo not in self.bundles:
                    self.bundles[repo] = json.load(open(hf_hub_download(hf_repo, f, repo_type="dataset", force_download=True)))
        assert self.bundles, f"no <repo>/bundle.json in {hf_repo} (run harvest_bundle.py)"
        local = snapshot_download(hf_repo, repo_type="dataset",
                                  allow_patterns=[f"{r}/*.manifest.json" for r in self.bundles])
        self.tasks: dict[str, Artifact] = {}
        for repo, b in self.bundles.items():
            for task in b["tasks"]:
                d = json.load(open(Path(local) / repo / f"{task}.manifest.json"))
                pytest = [x[len("pytest-"): -len(".dist-info")] for x in d.get("dists") or [] if x.startswith("pytest-") and x.endswith(".dist-info")]
                assert d["repo"] == repo and (not repo_of or repo_of.get(task, repo) == repo), f"{task}: repo mismatch"
                self.tasks[task] = Artifact(task=task, repo=repo, base=d["base_commit"], python=d.get("python", ""),
                                            uvdir=d.get("uvdir") or "", deleted=list(d.get("deleted") or []),
                                            has_delta=bool(d.get("delta_slim_zst")), workdir=R2E_WORKDIR.get(repo, "/testbed"),
                                            pytest=pytest[0] if pytest else "")
        self.tool_sha = sha256_file(TOOL)
        self.tool_remote = f"tools/harvest_materialize-{self.tool_sha[:12]}.py"

    def size_mb(self, path: str) -> float:
        return self.sizes.get(path, 0) / 1e6

    def ensure_tool_uploaded(self) -> None:
        """The Dockerfile pins the materialiser by sha256 under a content-addressed name; upload this copy if absent."""
        from huggingface_hub import HfApi
        if self.tool_remote not in self.sizes:
            HfApi().upload_file(path_or_fileobj=str(TOOL), path_in_repo=self.tool_remote, repo_id=self.hf_repo,
                                repo_type="dataset", commit_message=f"tools: harvest_materialize.py {self.tool_sha[:12]}")
            self.sizes[self.tool_remote] = TOOL.stat().st_size


def r2e_prefix(repo: str) -> list[str]:
    """R2E-Gym's Dockerfile.<repo> up to and including the clone, normalised: no build ARG, clone into /testbed."""
    src = (R2E_DOCKERFILES / f"Dockerfile.{repo}").read_text().splitlines()
    out: list[str] = []
    for line in src:
        s = line.strip()
        if not s or s.startswith("#") or s.startswith("ARG ") or s.startswith("CMD "):
            continue
        if s.startswith("ENV DEBIAN_FRONTEND"):
            out.append("ENV DEBIAN_FRONTEND=noninteractive")
            continue
        if "git clone" in s:
            out.append(f"RUN git clone https://github.com/{GITHUB[repo]}.git /testbed")
            break
        out.append(s)
    else:
        raise ValueError(f"Dockerfile.{repo}: no git clone line")
    return out


def dockerfile_for_repo(repo: str, arts: list[Artifact], index: ArtifactIndex) -> str:
    """Deterministic per-repo image from the repo's bundle; same bundle -> byte-identical output."""
    assert arts and all(a.repo == repo for a in arts)
    b = index.bundles[repo]
    missing = sorted({a.task for a in arts} - set(b["tasks"]))
    assert not missing, f"{repo}: {len(missing)} tasks not in the bundle: {missing[:5]}"
    base = f"https://huggingface.co/datasets/{index.hf_repo}/resolve/main"
    pythons = sorted(u for u in b["pythons"] if u)
    workdir = arts[0].workdir
    tool_remote, tool_sha = BUILT_TOOL_PINS.get(repo, (index.tool_remote, index.tool_sha))
    lines = r2e_prefix(repo)
    v2 = b.get("format") == "v2"
    lines += [
        # zstd for the bundles, python3 (ubuntu 22.04 = 3.10.12, the interpreter behind the venvs with no uv dir);
        # v2 bundles keep every blob zstd-compressed, so the materialiser needs the zstandard module to inflate a venv
        # in-process (~1 s); tmux/asciinema are added by harbor's bake_agent_tooling, apt lists dropped to keep the layer small
        ("RUN apt-get update -y && apt-get install -y zstd python3 python3-pip && rm -rf /var/lib/apt/lists/* && "
         "pip3 install --no-cache-dir zstandard==0.23.0") if v2 else
        "RUN apt-get update -y && apt-get install -y zstd python3 && rm -rf /var/lib/apt/lists/*",
        "WORKDIR /testbed",
        # submodules (numpy, aiohttp): populate .git/modules at build so the per-task `git submodule update` works offline
        "RUN git submodule update --init --recursive || true",
    ]
    if workdir != "/testbed":
        lines.append(f"RUN ln -s /testbed {workdir}")
    for u in pythons:
        lines.append(f"RUN cd / && curl -fsSL --retry 5 {base}/pythons/{u}.tar.zst | tar --zstd -xpf -")
    lines.append(
        f"RUN mkdir -p /opt/artifacts/tools /opt/venvs && curl -fsSL --retry 5 -o /opt/artifacts/tools/harvest_materialize.py "
        f"{base}/{tool_remote} && echo '{tool_sha}  /opt/artifacts/tools/harvest_materialize.py' | sha256sum -c"
    )
    for key in ("blobs", "tasks_bundle"):
        rec = b[key]
        name = Path(rec["file"]).name
        lines.append(
            f"RUN cd /opt/artifacts && curl -fsSL --retry 5 -o {name} {base}/{rec['file']} && "
            f"echo '{rec['sha256']}  {name}' | sha256sum -c && tar --zstd -xf {name} && rm {name}"
        )
    lines += [
        f"ENV VIRTUAL_ENV={workdir}/.venv",
        f'ENV PATH="{workdir}/.venv/bin:$PATH"',
        "RUN mkdir -p /logs/verifier",
    ]
    return "\n".join(lines) + "\n"


def estimate_image_mb(repo: str, arts: list[Artifact], index: ArtifactIndex) -> dict:
    b = index.bundles[repo]
    py = sum(index.size_mb(f"pythons/{u}.tar.zst") for u in b["pythons"] if u)
    blobs_raw = (b["blobs"].get("zst_bytes") if b.get("format") == "v2" else b["blobs"]["raw_bytes"]) / 1e6
    tasks_zst = b["tasks_bundle"]["bytes"] / 1e6
    return {"tasks": len(arts), "bundle_tasks": b["n_tasks"], "pythons": len([u for u in b["pythons"] if u]),
            "python_mb": round(py * 3), "blobs_raw_mb": round(blobs_raw), "tasks_bundle_mb": round(tasks_zst),
            "base_mb": BASE_MB.get(repo, 500), "total_mb": round(py * 3 + blobs_raw + tasks_zst + BASE_MB.get(repo, 500))}


SETUP_SH = r"""#!/bin/bash
# Per-task state on top of the per-repo artifact image (tt_daytona_tree_v3.py). No network, no build:
# check out the base commit, materialise the harvested venv (hard links from the blob store, outside the repo, symlinked
# as $W/.venv) and this commit's build outputs (mtimes preserved, so setuptools never recompiles), remove what install.sh
# had removed, make `python`/`pytest` resolve to the venv from any PATH.
set -euo pipefail
T={task}; R={repo}; BASE={base}; A=/opt/artifacts; W={workdir}
cd /testbed
rm -rf /testbed/.venv "$W/.venv" /opt/venvs/"$T" 2>/dev/null || true
git -c safe.directory='*' checkout -q -f "$BASE"
git -c safe.directory='*' clean -fdxq
# git:// submodule URLs (pyramid's docs/_themes before 2015-04) can never clone: GitHub dropped the git protocol, and a
# connect to port 9418 times out after ~136 s, twice. Refuse the protocol so the doomed clone fails at once (same end state)
git -c safe.directory='*' -c protocol.git.allow=never submodule update --init --recursive -q 2>/dev/null || true
# the venv lives outside the repo (a symlink from $W/.venv): the verifier's trusted-test restore walks every
# untracked path under /testbed, and a real .venv tree would be walked and partly deleted (numpy/testing, */tests/*)
/usr/bin/python3 "$A/tools/harvest_materialize.py" --artifacts "$A" --repo "$R" --task "$T" --testbed /testbed \
    --venv-root /opt/venvs/"$T" --no-tests
ln -sfn /opt/venvs/"$T"/.venv "$W/.venv"
ln -sfn "$W/.venv/bin/python" /usr/local/bin/python
ln -sfn "$W/.venv/bin/python" /usr/local/bin/python3
[ -x "$W/.venv/bin/pytest" ] && ln -sfn "$W/.venv/bin/pytest" /usr/local/bin/pytest || true
"$W/.venv/bin/python" -c "import {module}"
echo "setup ok: $(git rev-parse --short HEAD) python=$(readlink -f "$W/.venv/bin/python") which=$(command -v python)"
"""


def setup_sh(a: Artifact) -> str:
    return SETUP_SH.format(task=a.task, repo=a.repo, base=a.base, workdir=a.workdir, module=MODULE.get(a.repo, a.repo))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="print the per-repo Dockerfile and size estimate for an artifact dataset")
    ap.add_argument("--hf-repo", default="laion/r2egym-build-artifacts")
    ap.add_argument("--repo", required=True)
    ap.add_argument("--tasks", default=None, help="comma list; default = every task of the repo in the bundle")
    a = ap.parse_args()
    idx = ArtifactIndex(a.hf_repo, repos=[a.repo])
    arts = [t for t in idx.tasks.values() if t.repo == a.repo and (not a.tasks or t.task in a.tasks.split(","))]
    print(dockerfile_for_repo(a.repo, arts, idx))
    print("# estimate:", json.dumps(estimate_image_mb(a.repo, arts, idx)))
