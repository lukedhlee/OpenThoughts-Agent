#!/usr/bin/env python3
"""Build the per-repo Daytona snapshots of an artifact-shape tree (tt_daytona_tree_v3.py output) by hand, with streamed logs.

harbor's auto_snapshot would build each image on first use, but a Dockerfile error is only visible in the streamed build
log and eleven concurrent first-use builds are more than the builder likes; so build them here, N at a time, under the
exact names harbor will look up (`harbor__<env-dir hash>__snapshot`, computed with harbor's own function on one task of
each repo). Existing ACTIVE snapshots are skipped; ERROR/BUILD_FAILED ones are deleted and rebuilt.

  PYTHONPATH=/Users/lukedhlee/harbor-wt/snowball-r2egym/src /Users/lukedhlee/harbor/.venv/bin/python build_snapshots.py \
      --tree <tree> [--repos pandas,sympy] [--parallel 4] [--logs <dir>] [--disk 10]
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path


def load_secret(name: str) -> str:
    v = os.environ.get(name, "")
    if not v:
        for line in open(os.path.expanduser("~/.config/otagent/secrets.env")):
            line = line.strip()
            if line.startswith((f"{name}=", f"export {name}=")):
                v = line.split("=", 1)[1]
                break
    v = v.split("#", 1)[0].strip().strip("\"'").split()[0] if v.strip() else ""
    if not v:
        raise SystemExit(f"{name} not found")
    return v


def repo_dockerfiles(tree: Path, repos: list[str] | None) -> dict[str, tuple[Path, str]]:
    """repo -> (Dockerfile path, snapshot name), the name from harbor's hash of one task's environment dir."""
    from harbor.utils.container_cache import environment_dir_hash_truncated
    tasks = [l.strip() for l in open(tree / "TASKS.txt") if l.strip()]
    out: dict[str, tuple[Path, str]] = {}
    for df in sorted(tree.glob("Dockerfile.*")):
        repo = df.name.split(".", 1)[1]
        if repos and repo not in repos:
            continue
        text = df.read_text()
        # one task of this repo: its environment/Dockerfile must equal Dockerfile.<repo>
        for t in tasks:
            env = tree / t / "environment"
            if (env / "Dockerfile").exists() and (env / "Dockerfile").read_text() == text:
                out[repo] = (df, f"harbor__{environment_dir_hash_truncated(env, truncate=12)}__snapshot")
                break
        else:
            raise SystemExit(f"{repo}: no task in TASKS.txt carries Dockerfile.{repo}")
    return out


async def build_one(d, repo: str, df: Path, name: str, logs: Path, disk: int, sem: asyncio.Semaphore) -> dict:
    from daytona import CreateSnapshotParams, Image, Resources
    from harbor.environments.daytona.snapshots import bake_agent_tooling
    async with sem:
        t0 = time.time()
        log = open(logs / f"{repo}.build.log", "a")
        try:
            existing = await d.snapshot.get(name)
            st = str(existing.state).lower()
            if st.endswith("active"):
                print(f"[{repo}] {name} already ACTIVE ({existing.size} GB), skipping", flush=True)
                return {"repo": repo, "name": name, "state": "active", "size": existing.size, "seconds": 0, "skipped": True}
            print(f"[{repo}] {name} exists in state {st}: deleting before rebuild", flush=True)
            await d.snapshot.delete(existing)
            await asyncio.sleep(5)
        except Exception as e:  # noqa: BLE001
            if "not found" not in str(e).lower() and "404" not in str(e):
                print(f"[{repo}] get({name}) -> {type(e).__name__}: {str(e)[:120]}", flush=True)
        print(f"[{repo}] building {name} from {df.name} (disk {disk} GB)", flush=True)
        try:
            snap = await d.snapshot.create(
                CreateSnapshotParams(name=name, image=bake_agent_tooling(Image.from_dockerfile(str(df))),
                                     resources=Resources(cpu=2, memory=4, disk=disk)),
                on_logs=lambda chunk: (log.write(chunk), log.flush()),
                timeout=0,
            )
            dt = time.time() - t0
            print(f"[{repo}] DONE {snap.name} state={snap.state} size={snap.size} GB in {dt:.0f}s", flush=True)
            return {"repo": repo, "name": name, "state": str(snap.state), "size": snap.size, "seconds": round(dt)}
        except Exception as e:  # noqa: BLE001
            dt = time.time() - t0
            print(f"[{repo}] FAILED after {dt:.0f}s: {type(e).__name__}: {str(e)[:300]} (log: {log.name})", flush=True)
            return {"repo": repo, "name": name, "state": "failed", "error": f"{type(e).__name__}: {str(e)[:300]}", "seconds": round(dt)}
        finally:
            log.close()


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", required=True)
    ap.add_argument("--repos", default=None, help="comma list; default = every Dockerfile.<repo> in the tree")
    ap.add_argument("--parallel", type=int, default=4)
    ap.add_argument("--disk", type=int, default=10, help="snapshot default disk GB (sandbox storage)")
    ap.add_argument("--logs", default=None)
    ap.add_argument("--api-key-env", default="DAYTONA_API_KEY")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    tree = Path(a.tree)
    logs = Path(a.logs or (tree / "build_logs"))
    logs.mkdir(parents=True, exist_ok=True)
    repos = [r for r in a.repos.split(",")] if a.repos else None
    plan = repo_dockerfiles(tree, repos)
    for repo, (df, name) in plan.items():
        print(f"{repo:11} {name}  <- {df}")
    if a.dry_run:
        return 0
    from daytona import AsyncDaytona, DaytonaConfig
    key = load_secret(a.api_key_env)
    sem = asyncio.Semaphore(a.parallel)
    async with AsyncDaytona(DaytonaConfig(api_key=key)) as d:
        results = await asyncio.gather(*(build_one(d, r, df, n, logs, a.disk, sem) for r, (df, n) in plan.items()))
    import json
    (logs / "results.json").write_text(json.dumps(results, indent=1))
    print(json.dumps(results, indent=1))
    return 0 if all(r.get("state", "").lower().endswith("active") for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
