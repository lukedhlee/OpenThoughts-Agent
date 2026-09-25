#!/usr/bin/env python3
"""Resume per-base snapshot recovery without deleting snapshots (copied from data/r2egym/daytona_artifacts; a
manifest row's 'repo' is the CalibForge base label: ubuntu2404, tb2verifier, bookworm).

Prefer --manifest recovery/manifest.json (digest-pinned registry images when present).
--tree remains supported for Dockerfile rebuilds. Credentials default to the explicit
secrets file, never an inherited shell key. This tool never purges snapshots.
"""
from __future__ import annotations

import argparse
import hashlib
import asyncio
import json
import os
import re
import shlex
import time
from pathlib import Path


DEFAULT_KEY_FILE = '~/.config/otagent/daytona_eval.env'  # copy of Jupiter's keys/daytona_eval.env (key ...b61)


def load_secret(name: str, key_file: str | None = None) -> str:
    path = Path(key_file or DEFAULT_KEY_FILE).expanduser()
    for line in path.read_text().splitlines():
        fields = shlex.split(line, comments=True)
        if fields and fields[0] == 'export':
            fields = fields[1:]
        if len(fields) == 1 and fields[0].startswith(name + '='):
            value = fields[0].split('=', 1)[1]
            if value:
                return value
    raise ValueError(f'{name} missing from {path}')


def repo_dockerfiles(tree: Path, repos: list[str] | None) -> dict[str, tuple[Path, str]]:
    from harbor.utils.container_cache import environment_dir_hash_truncated
    tasks = (tree / 'TASKS.txt').read_text().splitlines()
    out = {}
    for df in sorted(tree.glob('Dockerfile.*')):
        repo = df.name.split('.', 1)[1]
        if repos and repo not in repos:
            continue
        content = df.read_bytes()
        for task in tasks:
            env = tree / task / 'environment'
            if (env / 'Dockerfile').is_file() and (env / 'Dockerfile').read_bytes() == content:
                out[repo] = (df, f'harbor__{environment_dir_hash_truncated(env, truncate=12)}__snapshot')
                break
        else:
            raise ValueError(f'{repo}: no matching task environment')
    if repos and set(repos) - out.keys():
        raise ValueError(f'Unknown repos: {set(repos) - out.keys()}')
    return out


def state(snapshot) -> str:
    value = getattr(snapshot.state, 'value', snapshot.state)
    return str(value).lower().replace('_', ' ').split('.')[-1]


def not_found(exc: Exception) -> bool:
    return getattr(exc, 'status_code', None) == 404 or '404' in str(exc) or 'not found' in str(exc).lower()


async def get_optional(service, name):
    try:
        return await asyncio.wait_for(service.get(name), 30)
    except Exception as exc:
        if not_found(exc):
            return None
        raise  # Auth/network errors must never be interpreted as permission to build.


async def ensure_snapshot(service, name, create, timeout=2400, poll=10):
    """Wait after ambiguous create failures; only create after a confirmed initial 404."""
    deadline = time.monotonic() + timeout
    snapshot = await get_optional(service, name)
    submitted = False
    activated = False
    create_error = None
    last_state = None
    while time.monotonic() < deadline:
        if snapshot is None:
            if not submitted:
                submitted = True
                try:
                    snapshot = await asyncio.wait_for(create(), max(0.01, deadline-time.monotonic()))
                    continue
                except Exception as exc:
                    create_error = type(exc).__name__
                    # A timeout/conflict can leave a live server-side build. Do not submit twice.
                    print(f'[{name}] create returned {create_error}; checking server state', flush=True)
            elif create_error and create_error not in {'TimeoutError', 'DaytonaConnectionTimeoutError',
                    'DaytonaBadGatewayError', 'DaytonaConflictError', 'DaytonaTimeoutError'}:
                raise RuntimeError(f'{name}: create failed ({create_error}), no snapshot found; check quota/credentials')
        else:
            current = state(snapshot)
            if current != last_state:
                print(f'[{name}] {current}', flush=True)
                last_state = current
            if current == 'active':
                return snapshot
            if current in {'error', 'build failed', 'failed'}:
                raise RuntimeError(f'{name}: server state {current}; preserved for inspection. Use snapshot census before any cleanup.')
            if current == 'inactive' and not activated:
                activated = True
                await asyncio.wait_for(service.activate(snapshot), max(0.01, deadline-time.monotonic()))
            elif current not in {'inactive', 'pending', 'building', 'pulling', 'snapshotting', 'removing', 'activating'}:
                raise RuntimeError(f'{name}: unrecognized state {current}; left untouched')
        await asyncio.sleep(min(poll, max(0, deadline-time.monotonic())))
        snapshot = await get_optional(service, name)
    raise TimeoutError(f'{name}: recovery wait expired; server build left untouched. Rerun to resume.')


def load_plan(args):
    repos = args.repos.split(',') if args.repos else None
    if args.manifest:
        manifest_path = Path(args.manifest).resolve()
        manifest = json.loads(manifest_path.read_text())
        if manifest['version'] != 1:
            raise ValueError('Unsupported recovery manifest version')
        plan = [dict(row) for row in manifest['images'] if not repos or row['repo'] in repos]
        if repos and set(repos) - {r['repo'] for r in plan}:
            raise ValueError('Requested repo absent from manifest')
        for row in plan:
            row['dockerfile'] = str(manifest_path.parent / row['dockerfile'])
            actual = hashlib.sha256(Path(row['dockerfile']).read_bytes()).hexdigest()
            if actual != row['dockerfile_sha256']:
                raise ValueError(f"{row['repo']}: recipe checksum mismatch")
            digest = row.get('image')
            if digest and not re.fullmatch(r'[^\s]+@sha256:[0-9a-f]{64}', digest):
                raise ValueError(f"{row['repo']}: registry image must be pinned by sha256 digest")
    else:
        plan = [dict(repo=r, dockerfile=str(df), name=n, baked=False)
                for r, (df, n) in repo_dockerfiles(Path(args.tree).resolve(), repos).items()]
    if not plan:
        raise ValueError('No images selected')
    return sorted(plan, key=lambda r: (-r.get('size_gb', 0), r['repo']))


async def main():
    ap = argparse.ArgumentParser(description=__doc__)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument('--tree')
    src.add_argument('--manifest')
    ap.add_argument('--repos', help='comma-separated; only restore what the run needs')
    ap.add_argument('--parallel', type=int, default=4)
    ap.add_argument('--disk', type=int, default=10)
    ap.add_argument('--logs', default='snapshot_recovery_logs')
    ap.add_argument('--key-file', default=DEFAULT_KEY_FILE)
    ap.add_argument('--api-key-env', default='DAYTONA_API_KEY', help='variable name inside key file')
    ap.add_argument('--wait-seconds', type=float, default=2400)
    ap.add_argument('--poll-seconds', type=float, default=10)
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--require-registry', action='store_true', help='refuse Dockerfile fallback')
    a = ap.parse_args()
    if not 1 <= a.parallel <= 5 or min(a.wait_seconds, a.poll_seconds, a.disk) <= 0:
        ap.error('parallel must be 1–5; timeouts and disk must be positive')
    plan = load_plan(a)
    if a.require_registry and any(not r.get('image') for r in plan):
        ap.error('Some selected images have no registry digest yet')
    for r in plan:
        print(f"{r['repo']:12} {r['name']} <- {r.get('image') or r['dockerfile']}", flush=True)
    if a.dry_run:
        return 0
    from daytona import AsyncDaytona, DaytonaConfig, CreateSnapshotParams, Image, Resources
    from harbor.environments.daytona.snapshots import bake_agent_tooling
    logs = Path(a.logs).resolve(); logs.mkdir(parents=True, exist_ok=True)
    sem = asyncio.Semaphore(a.parallel)
    results = []
    async with AsyncDaytona(DaytonaConfig(api_key=load_secret(a.api_key_env, a.key_file))) as d:
        async def run(row):
            async with sem:
                started = time.monotonic()
                result = dict(repo=row['repo'], name=row['name'])
                with (logs / f"{row['repo']}.build.log").open('a') as log:
                    async def create():
                        image = row.get('image')
                        if not image:
                            image = Image.from_dockerfile(row['dockerfile'])
                            if not row.get('baked'):
                                image = bake_agent_tooling(image)
                        return await d.snapshot.create(CreateSnapshotParams(name=row['name'], image=image,
                            resources=Resources(cpu=2, memory=4, disk=a.disk)),
                            on_logs=lambda chunk: (log.write(chunk), log.flush()), timeout=0)
                    try:
                        snap = await ensure_snapshot(d.snapshot, row['name'], create, a.wait_seconds, a.poll_seconds)
                        result.update(state='active', size=snap.size, snapshot_id=snap.id)
                    except Exception as exc:
                        result.update(state='failed', error=str(exc))
                result['seconds'] = round(time.monotonic()-started, 1)
                results.append(result)
                temp = logs / 'results.json.tmp'
                temp.write_text(json.dumps(results, indent=2)+'\n'); temp.replace(logs/'results.json')
                print(json.dumps(result), flush=True)
        await asyncio.gather(*(run(row) for row in plan))
    return 0 if all(r['state'] == 'active' for r in results) else 1


if __name__ == '__main__':
    raise SystemExit(asyncio.run(main()))
