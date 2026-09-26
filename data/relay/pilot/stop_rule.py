#!/usr/bin/env python3
"""The full relay run's in-run stop rule (Luke 2026-09-26): once at least --after scored relay episodes have finished,
cancel if relay context overflow > --ovf-max of scored episodes, or the executed trace's valid-format rate
< --fmt-min, or harness errors > --herr-max of finished trials.

    python stop_rule.py <run dir> <run name> <arm> --after 300 --ovf-max 0.30 --fmt-min 0.98 --herr-max 0.10

Prints one line: "wait <scored>" below --after, else "ok|stop <scored> <ovf> <fmt> <herr> <n>" (exit 0 either way).
Definitions are check_decide.py --rule ctxbudget's (scored = usable or ContextLengthExceededError; format =
readout.format_validity over the trials' agent steps). Deadline censoring is not applied: before the deadline no
episode is censored. A trial's summary is cached in <run dir>/stop_rule_cache.jsonl (finished trials never change),
so each call reads only the new trajectories.
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import readout  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run_dir'); ap.add_argument('name'); ap.add_argument('arm')
    ap.add_argument('--after', type=int, default=300)
    ap.add_argument('--ovf-max', type=float, default=0.30)
    ap.add_argument('--fmt-min', type=float, default=0.98)
    ap.add_argument('--herr-max', type=float, default=0.10)
    a = ap.parse_args()
    cache_p = os.path.join(a.run_dir, 'stop_rule_cache.jsonl')
    cache = {}
    if os.path.exists(cache_p):
        for r in readout.read_jsonl(cache_p):
            cache[r['key']] = r
    new = []
    for d in readout.arm_job_dirs(a.run_dir, a.name, a.arm):
        done = {os.path.basename(os.path.dirname(p)) for p in glob.glob(os.path.join(d, '*', 'result.json'))}
        todo = {t for t in done if f'{os.path.basename(d)}/{t}' not in cache}
        if not todo:
            continue
        for t in readout.trials(d, only=todo):
            key = f'{os.path.basename(d)}/{t["trial"]}'
            if t['trial'] in todo and key not in cache:
                r = dict(key=key, exc=t['exc'], usable=readout.usable(t), harness_error=t['harness_error'],
                         steps=len(t['steps']), bad=sum(1 for s in t['steps'] if s['parse_error_obs']))
                cache[key] = r
                new.append(r)
    if new:
        with open(cache_p, 'a') as f:
            for r in new:
                f.write(json.dumps(r) + '\n')
    rows = list(cache.values())
    scored = [r for r in rows if r['usable'] or r['exc'] == 'ContextLengthExceededError']
    if len(scored) < a.after:
        print('wait', len(scored))
        return
    ovf = sum(1 for r in scored if r['exc'] == 'ContextLengthExceededError') / len(scored)
    steps = sum(r['steps'] for r in rows)
    fmt = 1 - sum(r['bad'] for r in rows) / steps if steps else 1.0
    herr = sum(1 for r in rows if r['harness_error']) / len(rows)
    stop = ovf > a.ovf_max or fmt < a.fmt_min or herr > a.herr_max
    print('%s %d %.4f %.4f %.4f %d' % ('stop' if stop else 'ok', len(scored), ovf, fmt, herr, len(rows)))


if __name__ == '__main__':
    main()
