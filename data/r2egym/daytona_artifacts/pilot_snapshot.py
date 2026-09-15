#!/usr/bin/env python3
"""Build ONE Daytona snapshot that carries R2E-Gym's harvested per-commit build artifacts for a repo.

The image = R2E-Gym's own per-repo Dockerfile prefix (ubuntu:22.04 + apt deps + uv + `git clone <repo> /testbed`)
+ the uv-managed Pythons untarred into /root/.local/share/uv/python + every per-task tarball (venv, delta, tests)
kept as-is under /opt/artifacts + harbor's agent tooling layer (tmux, asciinema). Nothing is untarred into /testbed
at build time: the rollout setup (`pilot_sandbox.py`) does `git checkout <base> && untar venv && untar delta`.

Usage (harbor venv, Mac):
  /Users/lukedhlee/harbor/.venv/bin/python pilot_snapshot.py --hf-repo laion/r2egym-build-artifacts-pilot \
      --name harbor__pilot-pandas-artifacts__snapshot --github-repo pandas-dev/pandas [--api-key-env DAYTONA_API_KEY]
"""
import argparse
import asyncio
import os
import sys
import time

from huggingface_hub import HfApi

sys.path.insert(0, "/Users/lukedhlee/harbor/src")


def load_secret(name: str) -> str:
    if os.environ.get(name):
        return os.environ[name]
    for line in open(os.path.expanduser("~/.config/otagent/secrets.env")):
        line = line.strip()
        if line.startswith((f"{name}=", f"export {name}=")):
            return line.split("=", 1)[1].split("#", 1)[0].strip().strip("\"'").split()[0]
    raise SystemExit(f"{name} not found")


def build_dockerfile(hf_repo: str, github_repo: str, files: list[str]) -> str:
    base = f"https://huggingface.co/datasets/{hf_repo}/resolve/main"
    uvpy = [f for f in files if f.startswith("uvpython_") and f.endswith(".tar.zst")]
    tars = [f for f in files if f.endswith(".tar.zst") and not f.startswith("uvpython_")]
    manifests = [f for f in files if f.startswith("manifests/")]
    lines = [
        "FROM ubuntu:22.04",
        "ENV DEBIAN_FRONTEND=noninteractive",
        'RUN echo "tzdata tzdata/Areas select America" | debconf-set-selections && echo "tzdata tzdata/Zones/America select Los_Angeles" | debconf-set-selections',
        "RUN apt-get update -y && apt-get install -y git curl wget build-essential ca-certificates python3-dev zstd && rm -rf /var/lib/apt/lists/*",
        "RUN curl -LsSf https://astral.sh/uv/install.sh | sh",
        'ENV PATH="/root/.cargo/bin:/root/.local/bin:${PATH}"',
        f"RUN git clone https://github.com/{github_repo}.git /testbed",
        "WORKDIR /testbed",
        # shared uv-managed interpreters, untarred once (paths inside the tarball are root/.local/share/uv/python/<dir>)
        "RUN mkdir -p /opt/artifacts && cd / && "
        + " && ".join(f"curl -fsSL {base}/{f} | tar --zstd -xpf -" for f in uvpy),
    ]
    # per-task tarballs stay tarballs; one RUN per ~6 files keeps layers reasonable
    for i in range(0, len(tars), 6):
        chunk = tars[i : i + 6]
        lines.append("RUN cd /opt/artifacts && " + " && ".join(f"curl -fsSLO {base}/{f}" for f in chunk))
    if manifests:
        lines.append(
            "RUN mkdir -p /opt/artifacts/manifests && cd /opt/artifacts/manifests && "
            + " && ".join(f"curl -fsSLO {base}/{f}" for f in manifests)
        )
    lines += [
        "ENV VIRTUAL_ENV=/testbed/.venv",
        'ENV PATH="/testbed/.venv/bin:$PATH"',
        "RUN mkdir -p /logs/verifier /r2e_tests",
    ]
    return "\n".join(lines) + "\n"


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-repo", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--github-repo", required=True)
    ap.add_argument("--api-key-env", default="DAYTONA_API_KEY")
    ap.add_argument("--dockerfile-out", default=None)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    files = HfApi().list_repo_files(a.hf_repo, repo_type="dataset")
    df = build_dockerfile(a.hf_repo, a.github_repo, files)
    out = a.dockerfile_out or f"/tmp/Dockerfile.{a.name}"
    open(out, "w").write(df)
    print(df)
    if a.dry_run:
        return 0

    from daytona import AsyncDaytona, CreateSnapshotParams, DaytonaConfig, Image, Resources
    from harbor.environments.daytona.snapshots import bake_agent_tooling

    key = load_secret(a.api_key_env)
    t0 = time.time()
    async with AsyncDaytona(DaytonaConfig(api_key=key)) as d:
        try:
            existing = await d.snapshot.get(a.name)
            print(f"snapshot {a.name} exists: state={existing.state} size={existing.size}")
            if str(existing.state).lower().endswith("active"):
                return 0
            await d.snapshot.delete(existing)
            print("deleted non-active existing snapshot")
        except Exception as e:  # noqa: BLE001
            print("no existing snapshot:", type(e).__name__)
        snap = await d.snapshot.create(
            CreateSnapshotParams(
                name=a.name,
                image=bake_agent_tooling(Image.from_dockerfile(out)),
                resources=Resources(cpu=2, memory=4, disk=10),
            ),
            on_logs=lambda chunk: print(chunk, end=""),
            timeout=0,
        )
        print(f"\nsnapshot created: {snap.name} state={snap.state} size={snap.size} GB in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
