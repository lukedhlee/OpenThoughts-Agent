#!/usr/bin/env python3
"""sandbox_count.py — read-only count of the Daytona eval org's sandboxes by state, and by harbor.instance prefix.

By diff from data/relay/horizon/cleanup_sandboxes_async.py (same async client and listing; nothing is created or deleted).
Run it on the login node before ramping an arm's seats: the org's deal is ~1,000 concurrent sandboxes shared by everyone.

    python sandbox_count.py [--key-file ~/.config/otagent/daytona_eval.env] [--prefix <trial-name prefix> ...]
"""
import argparse
import asyncio
import collections
import os
import re
from pathlib import Path


def load_key(path):
    text = Path(path).read_text().strip()
    m = re.search(r"DAYTONA_API_KEY\s*=\s*['\"]?([^'\"\s]+)", text)
    return m.group(1) if m else text.splitlines()[0].strip()


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--key-file', default=os.path.expanduser('~/.config/otagent/daytona_eval.env'))
    ap.add_argument('--prefix', action='append', default=[], help='count sandboxes whose harbor.instance starts with harbor-<prefix>')
    a = ap.parse_args()
    from daytona import AsyncDaytona, DaytonaConfig, ListSandboxesQuery  # noqa: E402
    by_state, by_prefix, n = collections.Counter(), collections.Counter(), 0
    async with AsyncDaytona(DaytonaConfig(api_key=load_key(a.key_file))) as client:
        async for sb in client.list(ListSandboxesQuery(limit=100), request_timeout=120):
            n += 1
            st = getattr(getattr(sb, 'state', ''), 'value', getattr(sb, 'state', ''))
            by_state[str(st)] += 1
            label = (getattr(sb, 'labels', None) or {}).get('harbor.instance', '')
            for p in a.prefix:
                if label.startswith('harbor-' + p):
                    by_prefix[(p, str(st))] += 1
    print(f'org sandboxes: {n}; by state: {dict(by_state)}')
    for (p, st), k in sorted(by_prefix.items()):
        print(f'  prefix {p} {st}: {k}')


if __name__ == '__main__':
    asyncio.run(main())
