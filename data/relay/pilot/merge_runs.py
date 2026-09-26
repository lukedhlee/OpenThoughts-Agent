#!/usr/bin/env python3
"""Merge a run and its top-up runs into one arm (e.g. Qwen-alone baseline 6 + 6b), so readout.py, select_kept.py and
sft/render.py read them as one run with every (task, rollout) slot counted once.

    python merge_runs.py --out <runs>/relay_full_baseline6m_20260926 --arm control \\
        <runs>/relay_full_baseline6_20260926 <runs>/relay_full_baseline6b_20260926
    python merge_runs.py --out <runs>/relay_full_relaym_20260926 --arm relay_repair --per-task 4 <run1> <run2> <run3>

Per task, in the order the runs are given: the first --per-task (default 1) SCORED trials (readout.usable: no harness
error, no verifier timeout, not deadline-censored) are kept; a task with fewer scored trials fills its remaining slots
with unscored trials, latest run first (so the top-up's harness errors stay visible). A top-up runs exactly the slots
the earlier runs did not score, so a task scored more than --per-task times would be a bug: it is reported and only
the first ones are kept.

Top-up list (no --out): --remaining <file> --tasks <list> writes each task of the list once per slot the runs have not
scored (--per-task minus its scored trials); run_pilot.sh orders repeated lines round by round.

Output (read-only links, nothing copied):
  jobs/<out name>_<arm>/<trial>   symlinks to the kept trial dirs (from both staggered halves of each run)
  router_<arm>/turns.jsonl        the router records of the kept trials' sessions (by session id), all runs
  router_<arm>/events.jsonl, engines.jsonl   every run's, concatenated
  run.meta                        node_hours = the sum of the runs' node-hours; nodes = the first run's
  MERGED.json                     sources, per-task source run and trial, counts
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import readout  # noqa: E402


def meta(run):
    p = os.path.join(run, 'run.meta')
    return dict(l.strip().split('=', 1) for l in open(p) if '=' in l) if os.path.exists(p) else {}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out')
    ap.add_argument('--arm', default='control')
    ap.add_argument('--per-task', type=int, default=1, help='rollout slots per task (the run\'s n_attempts)')
    ap.add_argument('--remaining', help='write the top-up list here instead of merging (needs --tasks)')
    ap.add_argument('--tasks', help='the task list the slots are counted over (with --remaining)')
    ap.add_argument('runs', nargs='+')
    a = ap.parse_args()
    arm, K = a.arm, a.per_task
    if not a.out and not a.remaining:
        sys.exit('--out or --remaining is required')
    if a.remaining and not a.tasks:
        sys.exit('--remaining needs --tasks')
    if a.out:
        out = os.path.abspath(a.out)
        oname = os.path.basename(out)
        if os.path.exists(out):
            sys.exit(f'{out} exists')
    per_run = []
    for run in a.runs:
        name = os.path.basename(os.path.normpath(run))
        rows, rv = [], readout.router_view(os.path.join(run, f'router_{arm}'))
        for d in readout.arm_job_dirs(run, name, arm):
            for t in readout.trials(d):
                t['dir'] = os.path.join(d, t['trial'])
                rows.append(t)
        readout.mark_censored(rv, rows)
        per_run.append((run, name, rows))
    kept, scored_twice = {}, []                  # task -> [(run name, trial, scored)], at most K
    for run, name, rows in per_run:
        for t in rows:
            if not t['task'] or not readout.usable(t):
                continue
            slots = kept.setdefault(t['task'], [])
            if len(slots) >= K:
                scored_twice.append((t['task'], name, t['trial']))
                continue
            slots.append((name, t, True))
    if a.remaining:
        ids = [l.strip() for l in open(a.tasks) if l.strip()]
        unknown = sorted(set(kept) - set(ids))
        rem = {task: K - len(kept.get(task, [])) for task in ids}
        with open(a.remaining, 'w') as f:
            for task in ids:
                f.write(f'{task}\n' * rem[task])
        print(json.dumps(dict(runs=[n for _, n, _ in per_run], arm=arm, per_task=K, tasks=len(ids),
                              slots=K * len(ids), scored_slots=sum(len(v) for v in kept.values()),
                              remaining_slots=sum(rem.values()), over_k=len(scored_twice),
                              scored_tasks_not_in_list=len(unknown),
                              remaining_hist={k: sum(1 for v in rem.values() if v == k) for k in range(K + 1)})))
        return
    for run, name, rows in reversed(per_run):    # slots left unscored: unscored trials, latest run first
        for t in rows:
            if t['task'] and not readout.usable(t) and len(kept.setdefault(t['task'], [])) < K:
                kept[t['task']].append((name, t, False))
    jd = os.path.join(out, 'jobs', f'{oname}_{arm}')
    os.makedirs(jd)
    os.makedirs(os.path.join(out, f'router_{arm}'))
    sids = set()
    for task, slots in sorted(kept.items()):
        for name, t, _ in slots:
            link = os.path.join(jd, f'{name}__{t["trial"]}')
            os.symlink(t['dir'], link)
            if t['sid']:
                sids.add(t['sid'])
    with open(os.path.join(out, f'router_{arm}', 'turns.jsonl'), 'w') as f:
        for run, _, _ in per_run:
            p = os.path.join(run, f'router_{arm}', 'turns.jsonl')
            for line in open(p) if os.path.exists(p) else []:
                try:
                    if json.loads(line).get('sid') in sids:
                        f.write(line if line.endswith('\n') else line + '\n')
                except json.JSONDecodeError:
                    pass
    for fn in ('events.jsonl', 'engines.jsonl'):
        with open(os.path.join(out, f'router_{arm}', fn), 'w') as f:
            for run, _, _ in per_run:
                p = os.path.join(run, f'router_{arm}', fn)
                if os.path.exists(p):
                    f.write(open(p).read())
    ms = [meta(r) for r, _, _ in per_run]
    nh = sum(float(m.get('node_hours') or 0) for m in ms)
    with open(os.path.join(out, 'run.meta'), 'w') as f:
        f.write(f'nodes={ms[0].get("nodes", "")}\nnode_hours={nh:.3f}\nmerged_from={",".join(n for _, n, _ in per_run)}\n'
                + ''.join(f'node_hours_{n}={m.get("node_hours")}\n' for (_, n, _), m in zip(per_run, ms)))
    flat = [v for slots in kept.values() for v in slots]
    counts = {n: dict(trials=len(rows), kept=sum(1 for v in flat if v[0] == n),
                      kept_scored=sum(1 for v in flat if v[0] == n and v[2])) for _, n, rows in per_run}
    summary = dict(out=out, arm=arm, runs=[n for _, n, _ in per_run], per_task_slots=K, tasks=len(kept),
                   scored_tasks=sum(1 for slots in kept.values() if any(v[2] for v in slots)),
                   scored_slots=sum(1 for v in flat if v[2]), kept_trials=len(flat), node_hours=round(nh, 3),
                   per_run=counts, scored_twice=scored_twice,
                   per_task={task: [dict(run=n, trial=t['trial'], scored=s) for n, t, s in slots]
                             for task, slots in sorted(kept.items())})
    json.dump(summary, open(os.path.join(out, 'MERGED.json'), 'w'), indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k != 'per_task'}, indent=1))


if __name__ == '__main__':
    main()
