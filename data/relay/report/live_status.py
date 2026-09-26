#!/usr/bin/env python3
"""live_status.py — read-only snapshot of the live relay runs, printed as JSON. Runs ON the Jupiter login node, fed on
stdin by update.sh (`ssh jupiter python3 - < live_status.py`); stdlib only, one thread, touches no job, tmux or file.

A run is live when its run dir has a driver.log but no run.meta, and its serve job is still in squeue. For each live
run it reports: the serve job's state, start and time limit, the driver's last `watch:` node-hours and its last
`stop rule` line, when harbor started, how many trials have a result.json, passes (verifier reward 1), and the
exception types seen. Planned trials come from the driver's first line (`tasks=<file> xN`) or its `staggered: a + b`.
"""
import collections
import glob
import json
import os
import re
import subprocess
import time

RUNS = '/e/fscratch/reformo/lee27/experiments/relay/pilot/runs'
USER = os.environ.get('USER', 'lee27')


def squeue():
    out = subprocess.run(['squeue', '-u', USER, '--noheader', '-o', '%i|%j|%T|%M|%l|%D|%S'],
                         capture_output=True, text=True).stdout
    jobs = {}
    for line in out.splitlines():
        p = line.strip().split('|')
        if len(p) == 7:
            jobs[p[0]] = dict(job=p[0], name=p[1], state=p[2], elapsed=p[3], limit=p[4], nodes=int(p[5]), start=p[6])
    return jobs


def run_status(rd, jobs):
    log = open(os.path.join(rd, 'driver.log'), errors='replace').read().splitlines()
    if not log:
        return None
    m = re.search(r'job=(\d+)', log[0])
    job = m.group(1) if m else None
    if job not in jobs:
        return None
    name = os.path.basename(rd)
    arms = re.search(r'arms=\[([^\]]*)\]', log[0])
    arms = arms.group(1).split() if arms else []
    planned = None
    tm = re.search(r'tasks=(\S+) x(\d+)', log[0])
    if tm and os.path.exists(tm.group(1)):
        planned = sum(1 for l in open(tm.group(1)) if l.strip()) * int(tm.group(2))
    ts = lambda l: l[1:l.index(']')] if l.startswith('[') else None
    st = dict(run=name, **jobs[job], arms=arms, planned=planned, node_h=None, watch_at=None, harbor_start=None,
              stop_rule=None, last_line=log[-1][:300])
    for l in log:
        sm = re.search(r'staggered: (\d+) \+ (\d+)', l)
        if sm and planned is None:
            st['planned'] = int(sm.group(1)) + int(sm.group(2))
        if 'harbor' in l and 'started' in l and st['harbor_start'] is None:
            st['harbor_start'] = ts(l)
        wm = re.search(r'watch: node-h=([\d.]+)', l)
        if wm:
            st['node_h'], st['watch_at'] = float(wm.group(1)), ts(l)
        if 'stop rule' in l:
            st['stop_rule'] = l[:300]
    finished = passes = 0
    exc = collections.Counter()
    first_done = None
    for arm in arms:
        for jd in (f'{name}_{arm}', f'{name}_{arm}_p2'):
            for f in glob.glob(os.path.join(rd, 'jobs', jd, '*', 'result.json')):
                try:
                    d = json.load(open(f))
                except Exception:
                    continue
                finished += 1
                e = (d.get('exception_info') or {}).get('exception_type')
                exc[e or 'none'] += 1
                r = ((d.get('verifier_result') or {}).get('rewards') or {}).get('reward')
                passes += int(r == 1)
    st.update(finished=finished, passes=passes, exceptions=dict(exc))
    return st


def main():
    jobs = squeue()
    runs = []
    for rd in sorted(glob.glob(os.path.join(RUNS, '*'))):
        if os.path.isdir(rd) and os.path.exists(os.path.join(rd, 'driver.log')) \
                and not os.path.exists(os.path.join(rd, 'run.meta')):
            s = run_status(rd, jobs)
            if s:
                runs.append(s)
    print(json.dumps(dict(now_epoch=time.time(), squeue=list(jobs.values()), runs=runs), indent=1))


if __name__ == '__main__':
    main()
