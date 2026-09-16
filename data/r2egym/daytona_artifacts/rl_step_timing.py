#!/usr/bin/env python3
"""Step timestamps of a MarinSkyRL run from its .out log: for each `WANDB_MIRROR kind=train step=N` line, the last
timestamp printed before it, plus the job start from sacct if given. Prints time-to-step-1 and per-step intervals.

usage: rl_step_timing.py <log.out> [--start 'YYYY-MM-DDTHH:MM:SS'] [--max-step N]
"""
import argparse, re
from datetime import datetime

TS = re.compile(r'(20\d\d-\d\d-\d\d)[ T](\d\d:\d\d:\d\d)')


def main(a):
    last_ts, steps = None, {}
    for line in open(a.log, errors='replace'):
        m = TS.search(line)
        if m:
            last_ts = datetime.fromisoformat(m.group(1) + 'T' + m.group(2))
        s = re.search(r'WANDB_MIRROR kind=train step=(\d+) ', line)
        if s and int(s.group(1)) not in steps:
            steps[int(s.group(1))] = last_ts
            if a.max_step and int(s.group(1)) >= a.max_step:
                break
    start = datetime.fromisoformat(a.start) if a.start else None
    prev = start
    for k in sorted(steps):
        t = steps[k]
        d = (t - prev).total_seconds() / 60 if (prev and t) else None
        print(f"step {k}: {t}  {'+%.1f min' % d if d is not None else ''}{' since job start' if k == min(steps) and start else ''}")
        prev = t
    if start and steps:
        last = max(steps)
        print(f"steps 1..{last}: {(steps[last] - start).total_seconds() / 60:.1f} min from job start; "
              f"{last / ((steps[last] - start).total_seconds() / 3600):.2f} steps/h incl. startup")
        if last > 1:
            print(f"steps 2..{last}: {(last - 1) / ((steps[last] - steps[1]).total_seconds() / 3600):.2f} steps/h after step 1")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('log'); p.add_argument('--start'); p.add_argument('--max-step', type=int, default=0)
    main(p.parse_args())
