#!/usr/bin/env python3
"""screen_report.py — per-task readout of one or more eval-only screen probes (make_arm.py --probe K).

Reads every trial's result.json under <run>/trials/eval_sessions/ (read_outcome() is Jupiter's refresh_screen.py's:
only the head and tail of each file, which carry the task name and the verifier/exception fields). A trial counts when
its reward is 0 or 1 and its exception is not one the run masks; masked and unscored trials are reported, never
counted as failures. Tree entries <task>__rN fold into <task>.

    python screen_report.py --out <dir> --repo-map <coverage.csv or tsv: task,repo> <run dir> [<run dir> ...]

Writes <out>/per_task.tsv (task, repo, attempts, solved, invalid) and prints counts per repo: tasks, attempted, solved
>= 1 (the learnable band), 0/attempted, mean pass rate. --band <tasks.txt> restricts the band to those tasks (the train
set) and writes <out>/band.txt.
"""
import argparse
import collections
import csv
import json
import re
from pathlib import Path

TAIL_BYTES = 4 << 20
REP = re.compile(r"__r\d+$")


def full_outcome(path):
    r = json.loads(path.read_text())
    reward = ((r.get('verifier_result') or {}).get('rewards') or {}).get('reward')
    return r['task_name'], reward, (r.get('exception_info') or {}).get('exception_type')


def read_outcome(path):
    size = path.stat().st_size
    with path.open('rb') as f:
        head = f.read(4096)
        f.seek(max(0, size - TAIL_BYTES))
        tail = f.read()
    name = re.search(rb'"task_name":\s*"([^"\\]+)"', head)
    start = tail.rfind(b'"verifier_result":')
    region = tail[start:] if start >= 0 else b''
    split = region.find(b'"exception_info":')
    exc = re.match(rb'"exception_info":\s*(?:null|\{\s*"exception_type":\s*"([^"\\]+)")', region[split:]) if split >= 0 else None
    if name and exc and (size <= TAIL_BYTES or start > 0):
        verifier = region[:split]
        reward = None
        if not re.match(rb'"verifier_result":\s*null', verifier):
            found = re.search(rb'"rewards":\s*\{\s*"reward":\s*(-?[0-9.eE+-]+)\s*[,}]', verifier)
            if not found:
                return full_outcome(path)
            reward = float(found.group(1))
        return name.group(1).decode(), reward, exc.group(1).decode() if exc.group(1) else None
    return full_outcome(path)


def masks_of(run):
    cfg = json.loads(next((run / 'configs').glob('*_rl_config.json')).read_text())
    for a in cfg['skyrl_hydra_args']:
        if a.lstrip('+').startswith('terminal_bench_config.harbor.mask_exceptions='):
            return set(json.loads(a.split('=', 1)[1]))
    return set()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('runs', nargs='+', type=Path)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--repo-map', type=Path, help='csv/tsv with task (or task_name) and repo columns')
    ap.add_argument('--band', type=Path, help='task list the band is drawn from (the train set)')
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    repo = {}
    if a.repo_map:
        text = a.repo_map.read_text()
        rows = csv.DictReader(text.splitlines(), delimiter='\t' if '\t' in text.splitlines()[0] else ',')
        for r in rows:
            repo[r.get('task') or r.get('task_name')] = r['repo']
    vals, invalid, exc = collections.defaultdict(list), collections.Counter(), collections.Counter()
    total = 0
    for run in a.runs:
        m = masks_of(run)
        for p in sorted((run / 'trials').glob('eval_sessions/*/*/result.json')):
            try:
                task, reward, ex = read_outcome(p)
            except (ValueError, OSError, KeyError):
                continue
            total += 1
            task = REP.sub('', task)
            if ex:
                exc[ex] += 1
            if reward not in (0, 1) or ex in m:
                invalid[task] += 1
                continue
            vals[task].append(int(reward))
    tasks = sorted(set(vals) | set(invalid))
    with open(a.out / 'per_task.tsv', 'w') as f:
        f.write('task\trepo\tattempts\tsolved\tinvalid\n')
        for t in tasks:
            f.write(f"{t}\t{repo.get(t, '?')}\t{len(vals[t])}\t{sum(vals[t])}\t{invalid[t]}\n")
    pool = set(Path(a.band).read_text().split()) if a.band else set(tasks)
    band = sorted(t for t in tasks if t in pool and sum(vals[t]) >= 1)
    if a.band:
        (a.out / 'band.txt').write_text('\n'.join(band) + '\n')
    by = collections.defaultdict(lambda: collections.Counter())
    for t in tasks:
        r = repo.get(t, '?')
        by[r]['tasks'] += 1
        by[r]['attempts'] += len(vals[t])
        by[r]['solves'] += sum(vals[t])
        by[r]['band'] += int(t in pool and sum(vals[t]) >= 1)
        by[r]['zero'] += int(len(vals[t]) > 0 and sum(vals[t]) == 0)
    print(f'trials read {total}; invalid (masked/unscored) {sum(invalid.values())}; exceptions {dict(exc.most_common(8))}')
    print('repo\ttasks\tsolved>=1\t0/all\tpass rate')
    for r in sorted(by):
        c = by[r]
        print(f"{r}\t{c['tasks']}\t{c['band']}\t{c['zero']}\t{c['solves'] / max(1, c['attempts']):.3f}")
    c = sum(by.values(), collections.Counter())
    print(f"ALL\t{c['tasks']}\t{c['band']}\t{c['zero']}\t{c['solves'] / max(1, c['attempts']):.3f}")


if __name__ == '__main__':
    main()
