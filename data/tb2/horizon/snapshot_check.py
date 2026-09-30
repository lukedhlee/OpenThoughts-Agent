#!/usr/bin/env python3
"""Read-only: does every task of a harbor task tree have a usable Daytona snapshot under the name harbor derives for it?

harbor's auto_snapshot looks up `harbor__<12-hex hash of the task's environment dir>__snapshot` (DAYTONA_TARGET unset)
and uses it when it exists. TB2 and SWE-bench Verified tasks resolve to Daytona's global snapshots under those names
(.claude/ops/jupiter/tb2_eval.md), so an eval needs none of our own; a task whose name does not resolve would make harbor
try to build one, which the eval org refuses or which takes one of its custom-snapshot slots. This script only GETs each
name; it never creates or deletes anything.

    PYTHONPATH=<harbor clone>/src python snapshot_check.py --key-file <daytona_eval.env> --tree <task tree> [--exclude a,b]

Prints one line per task that does not resolve (or is not ACTIVE) and a summary line.
"""
import argparse
import asyncio
import os
import re
import sys
from pathlib import Path


def load_key(path):
    text = Path(path).read_text().strip()
    m = re.search(r"DAYTONA_API_KEY\s*=\s*['\"]?([^'\"\s]+)", text)
    return m.group(1) if m else text.splitlines()[0].strip()


async def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--key-file', required=True)
    ap.add_argument('--tree', required=True)
    ap.add_argument('--exclude', default='')
    ap.add_argument('--conc', type=int, default=8)
    a = ap.parse_args()
    from daytona import AsyncDaytona, DaytonaConfig
    from harbor.utils.container_cache import environment_dir_hash_truncated
    skip = {t for t in a.exclude.split(',') if t}
    tasks = sorted(d.name for d in Path(a.tree).iterdir() if (d / 'task.toml').is_file() and d.name not in skip)
    names = {t: f'harbor__{environment_dir_hash_truncated(Path(a.tree) / t / "environment", truncate=12)}__snapshot' for t in tasks}
    sem = asyncio.Semaphore(a.conc)
    res = {}
    async with AsyncDaytona(DaytonaConfig(api_key=load_key(a.key_file))) as d:
        async def one(t):
            async with sem:
                try:
                    s = await d.snapshot.get(names[t])
                    st = getattr(s, 'state', None)
                    res[t] = (str(getattr(st, 'value', st)).lower(), bool(getattr(s, 'general', False)))
                except Exception as e:  # not found (404) or refused
                    res[t] = (f'missing: {type(e).__name__}: {str(e)[:120]}', False)
        await asyncio.gather(*(one(t) for t in tasks))
    bad = [t for t in tasks if res[t][0] != 'active']
    for t in bad:
        print(f'{t}\t{names[t]}\t{res[t][0]}')
    print(f'{a.tree}: {len(tasks)} tasks{" (excluded " + ",".join(sorted(skip)) + ")" if skip else ""}, '
          f'{len(tasks) - len(bad)} resolve to an ACTIVE snapshot ({sum(1 for t in tasks if res[t][1])} global), {len(bad)} do not; '
          f'{len(set(names.values()))} distinct snapshot names')
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    asyncio.run(main())
