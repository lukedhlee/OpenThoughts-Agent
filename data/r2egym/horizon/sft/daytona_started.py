#!/usr/bin/env python3
"""Print how many sandboxes of the Daytona eval org (every user) are started, starting or being created; 9999 when the
API does not answer. Read-only.

    python3 daytona_started.py [--key-file ~/.config/otagent/daytona_eval.env]
"""
import argparse
import json
import os
import urllib.request


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--key-file', default=os.path.expanduser('~/.config/otagent/daytona_eval.env'))
    a = ap.parse_args()
    key = None
    for line in open(a.key_file):
        if line.strip().startswith('DAYTONA_API_KEY='):
            key = line.split('=', 1)[1].strip().strip('"\'')
    cur, n = None, 0
    try:
        for _ in range(100):
            u = 'https://app.daytona.io/api/sandbox?limit=200' + (f'&cursor={cur}' if cur else '')
            d = json.load(urllib.request.urlopen(urllib.request.Request(u, headers={'Authorization': f'Bearer {key}'}), timeout=60))
            items = d.get('items', [])
            n += sum(1 for x in items if x.get('state') in ('started', 'starting', 'creating'))
            cur = d.get('nextCursor')
            if not cur or not items:
                break
        print(n)
    except Exception:
        print(9999)


if __name__ == '__main__':
    main()
