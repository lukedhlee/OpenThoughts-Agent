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

--harness-exclude-tasks <file> (Luke 2026-09-26 20:55 PT, relay top-up 4): judge the harness-error part only on trials of
tasks NOT in the file (the tasks with a harness error in attempts 1-3; their errors are the diagnosed in-sandbox tmux
kill, and their errored trials are excluded from the data anyway). Overflow and format stay on all scored episodes, the
limit stays. Without the option, <run dir>/harness_exclude_tasks.txt is used when it exists (so a live driver picks it
up); with neither, the rule is unchanged. Every call appends both harness-error rates (judged and all-task) to
<run dir>/stop_rule.log, and the printed line carries the all-task rate and trial count as two extra fields.

The last field of every printed line (wait lines too) is the run's real failures so far: scored (usable) episodes with
reward < 1, readout.py's real_failures before deadline censoring. run_pilot.sh TARGET_FAIL ends the run on it.
"""
import argparse
import glob
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import readout  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run_dir'); ap.add_argument('name'); ap.add_argument('arm')
    ap.add_argument('--after', type=int, default=300)
    ap.add_argument('--ovf-max', type=float, default=0.30)
    ap.add_argument('--fmt-min', type=float, default=0.98)
    ap.add_argument('--herr-max', type=float, default=0.10)
    ap.add_argument('--harness-exclude-tasks', help='judge harness errors only on tasks not listed here')
    a = ap.parse_args()
    # a live run's limits can be changed without restarting its driver: <run dir>/stop_rule_override.json
    # ({"herr_max": .., "ovf_max": .., "fmt_min": ..}) replaces the flags, and the log line records it
    ovr_p = os.path.join(a.run_dir, 'stop_rule_override.json')
    ovr = json.load(open(ovr_p)) if os.path.exists(ovr_p) else None
    for k, v in (ovr or {}).items():
        setattr(a, k, float(v))
    excl_p =a.harness_exclude_tasks or os.path.join(a.run_dir, 'harness_exclude_tasks.txt')
    excl = {l.strip() for l in open(excl_p) if l.strip()} if os.path.exists(excl_p) else None
    cache_p = os.path.join(a.run_dir, 'stop_rule_cache.jsonl')
    cache = {}
    if os.path.exists(cache_p):
        for r in readout.read_jsonl(cache_p):
            if 'task' in r and 'reward' in r:        # rows cached before task / reward were recorded are re-read
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
                r = dict(key=key, task=t['task'], reward=t['reward'], exc=t['exc'], usable=readout.usable(t), harness_error=t['harness_error'],
                         steps=len(t['steps']), bad=sum(1 for s in t['steps'] if s['parse_error_obs']))
                cache[key] = r
                new.append(r)
    if new:
        with open(cache_p, 'a') as f:
            for r in new:
                f.write(json.dumps(r) + '\n')
    rows = list(cache.values())
    scored = [r for r in rows if r['usable'] or r['exc'] == 'ContextLengthExceededError']
    real_fail = sum(1 for r in rows if r['usable'] and (r['reward'] or 0) < 1)
    if len(scored) < a.after:
        print('wait', len(scored), real_fail)
        return
    ovf = sum(1 for r in scored if r['exc'] == 'ContextLengthExceededError') / len(scored)
    steps = sum(r['steps'] for r in rows)
    fmt = 1 - sum(r['bad'] for r in rows) / steps if steps else 1.0
    herr_all = sum(1 for r in rows if r['harness_error']) / len(rows)
    judged = rows if excl is None else [r for r in rows if r['task'] not in excl]
    herr = sum(1 for r in judged if r['harness_error']) / len(judged) if judged else 0.0
    stop = ovf > a.ovf_max or fmt < a.fmt_min or herr > a.herr_max
    with open(os.path.join(a.run_dir, 'stop_rule.log'), 'a') as f:
        f.write(json.dumps(dict(ts=time.time(), scored=len(scored), ovf=round(ovf, 4), fmt=round(fmt, 4),
                                herr_judged=round(herr, 4), judged_trials=len(judged), herr_all=round(herr_all, 4),
                                all_trials=len(rows), exclude_file=excl_p if excl is not None else None,
                                real_failures=real_fail, override=ovr, verdict='stop' if stop else 'ok', all_task_rule='stop' if (ovf > a.ovf_max or fmt < a.fmt_min
                                                                                           or herr_all > a.herr_max) else 'ok')) + '\n')
    print('%s %d %.4f %.4f %.4f %d %.4f %d %d' % ('stop' if stop else 'ok', len(scored), ovf, fmt, herr, len(judged), herr_all,
                                              len(rows), real_fail))


if __name__ == '__main__':
    main()
