"""Restore only missing CalibForge snapshots; never delete or replace existing ones."""
import asyncio
import json
import re
import sys
from pathlib import Path

async def main():
    from daytona import AsyncDaytona, DaytonaConfig, CreateSnapshotParams, Image, Resources
    from harbor.environments.daytona.snapshots import bake_agent_tooling
    from harbor.utils.container_cache import environment_dir_hash_truncated
    tree, keyfile, logs = map(Path, sys.argv[1:])
    logs.mkdir(parents=True, exist_ok=True)
    text = keyfile.read_text()
    key = re.search(r"DAYTONA_API_KEY\s*=\s*['\"]?([^'\"\s]+)", text).group(1)
    # Select one representative per Dockerfile; avoid stat'ing the full NFS tree.
    todo = {p.read_text(): p for p in tree.glob('Dockerfile.*')}
    plan = []
    for task in (tree / 'TASKS.txt').read_text().splitlines():
        env = tree / task / 'environment'
        df = env / 'Dockerfile'
        if not df.is_file():
            continue
        source = todo.pop(df.read_text(), None)
        if source:
            name = f'harbor__{environment_dir_hash_truncated(env, truncate=12)}__snapshot'
            plan.append((source, name))
        if not todo:
            break
    assert not todo and len(plan) == 3, 'Incomplete CalibForge snapshot plan'
    async with AsyncDaytona(DaytonaConfig(api_key=key)) as d:
        for df, name in plan:
            try:
                snap = await d.snapshot.get(name)
            except Exception as e:
                if '404' not in str(e) and 'not found' not in str(e).lower():
                    raise
                with (logs / f'{name}.log').open('a') as log:
                    snap = await d.snapshot.create(
                        CreateSnapshotParams(name=name, image=bake_agent_tooling(Image.from_dockerfile(str(df))),
                                             resources=Resources(cpu=2, memory=4, disk=10)),
                        on_logs=lambda chunk: (log.write(chunk), log.flush()), timeout=0)
            state = str(getattr(snap.state, 'value', snap.state)).lower()
            assert state == 'active', (name, state)
            print(json.dumps({'name': name, 'state': state}), flush=True)
    print('SNAPSHOTS_OK', flush=True)

if __name__ == '__main__':
    asyncio.run(main())
