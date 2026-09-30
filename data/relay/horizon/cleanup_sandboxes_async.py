#!/usr/bin/env python3
"""cleanup_sandboxes.py on the async Daytona client, for a driver behind a proxy (Horizon).

By diff from data/relay/pilot/cleanup_sandboxes.py (same CLI, same "ours" rule: a sandbox's harbor.instance label starts
with harbor-<a trial dir of these jobs>; snapshots are never touched). The only delta is the client: the sync SDK's main
API client (urllib3) ignores HTTPS_PROXY and connects straight out, which hangs in SYN-SENT on a Horizon compute node
(2026-09-29); AsyncDaytona honors the proxy environment, as harbor itself does.

    python cleanup_sandboxes_async.py --key-file <daytona_eval.env> [--delete] <jobs_dir>/<name>_<arm> ...
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
    ap.add_argument('jobs', nargs='+', help='harbor job directories')
    ap.add_argument('--key-file', default=os.path.expanduser('~/.config/otagent/daytona_eval.env'))
    ap.add_argument('--delete', action='store_true')
    a = ap.parse_args()
    trials = set()
    for j in a.jobs:
        for d in Path(j).iterdir() if Path(j).is_dir() else []:
            if d.is_dir():
                trials.add(d.name)
    if not trials:
        sys.exit('no trial directories under the given jobs')
    from daytona import AsyncDaytona, DaytonaConfig, ListSandboxesQuery  # noqa: E402
    async with AsyncDaytona(DaytonaConfig(api_key=load_key(a.key_file))) as client:
        ours, scanned = [], 0
        async for sb in client.list(ListSandboxesQuery(limit=100), request_timeout=120):
            scanned += 1
            label = (getattr(sb, 'labels', None) or {}).get('harbor.instance', '')
            if not label.startswith('harbor-'):
                continue
            body = label[len('harbor-'):]
            if any(body.startswith(t + '-') or body.startswith(t + '__attempt_') for t in trials):
                ours.append(sb)
        for sb in ours:
            print(sb.id, getattr(getattr(sb, 'state', ''), 'value', getattr(sb, 'state', '')), sb.labels.get('harbor.instance'))
        print(f'{len(ours)} sandboxes (of {scanned} in the org) from {len(trials)} trials of {len(a.jobs)} jobs')
        if a.delete:
            n = 0
            for sb in ours:
                try:
                    await client.delete(sb, timeout=120)
                    n += 1
                except Exception as e:  # noqa: BLE001
                    print('failed', sb.id, type(e).__name__)
            print(f'deleted {n}')


if __name__ == '__main__':
    asyncio.run(main())
