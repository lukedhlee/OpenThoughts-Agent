#!/usr/bin/env python3
"""List, and with --delete remove, the Daytona sandboxes a pilot's harbor jobs left behind — by id, ours only.

    python cleanup_sandboxes.py --key-file <daytona_eval.env> <jobs_dir>/<name>_relay <jobs_dir>/<name>_control ...

harbor labels every sandbox `harbor.instance = harbor-<trial name>[__attempt_<n>]-<uuid>` (daytona/environment.py), and
each trial of a job has its own directory named after the trial, so a sandbox is ours exactly when its label starts
with `harbor-<a trial dir of these jobs>`; no other workstream's sandbox can match. Snapshots are never touched.
Dry run by default: prints id, state, label.
"""
import argparse
import os
import re
import sys
from pathlib import Path


def load_key(path):
    text = Path(path).read_text().strip()
    m = re.search(r"DAYTONA_API_KEY\s*=\s*['\"]?([^'\"\s]+)", text)
    return m.group(1) if m else text.splitlines()[0].strip()


def main():
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
    from daytona import Daytona, DaytonaConfig, ListSandboxesQuery  # noqa: E402
    client = Daytona(DaytonaConfig(api_key=load_key(a.key_file)))
    ours, scanned = [], 0
    for sb in client.list(ListSandboxesQuery(limit=100)):
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
                sb.delete()
                n += 1
            except Exception as e:  # noqa: BLE001
                print('failed', sb.id, e)
        print(f'deleted {n}')


if __name__ == '__main__':
    main()
