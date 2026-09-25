#!/usr/bin/env python3
"""Pick the kept traces of a relay run: 1:1 pass:fail per arm, at most 2 kept passes per task, failures only real
model failures (harness errors, verifier timeouts and deadline-censored episodes dropped), each failure labelled by
cause (readout.failure_cause). relay_repair episodes count only when the teacher wrote at least one turn (repair or
takeover), since only teacher turns are trained on.

    python select_kept.py --run-dir <run> [--name <run name>] --target 2000 --out kept_manifest.jsonl

Pairing: for each kept pass, a failure from the same task is preferred (a pass/fail pair); the remaining failure slots
are filled from other tasks, fewest-kept-first. Per arm, kept = 2 x min(passes after the cap, failures, target / 2).
"""
import argparse
import collections
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import readout  # noqa: E402


def arm_rows(run_dir, name, arm):
    rv = readout.router_view(os.path.join(run_dir, f'router_{arm}'))
    rows = readout.trials(os.path.join(run_dir, 'jobs', f'{name}_{arm}')) + \
        readout.trials(os.path.join(run_dir, 'jobs', f'{name}_{arm}_p2'))
    readout.mark_censored(rv, rows)
    eps = {e['sid']: e for e in rv['eps'].values()}
    out = []
    for t in rows:
        e = eps.get(t['sid'])
        if not readout.usable(t) or e is None:
            continue
        teacher_turns = sum(1 for r in e['main'] if r.get('owner') == 'teacher')
        if arm != 'control' and teacher_turns == 0:
            continue
        out.append(dict(arm=arm, task=t['task'], trial=t['trial'], sid=t['sid'], reward=t['reward'],
                        passed=readout.is_pass(t),
                        cause=None if readout.is_pass(t) else readout.failure_cause(t, e['ending']),
                        teacher_turns=teacher_turns, student_turns=sum(1 for r in e['main'] if r.get('owner') == 'student'),
                        repairs=sum(1 for r in e['main'] if r.get('repair_kind') == 'parse_error'),
                        takeover=(e['takeover'] or {}).get('trigger')))
    return out


def select(rows, target, seed):
    rng = random.Random(seed)
    passes = collections.defaultdict(list)
    fails = collections.defaultdict(list)
    for r in rows:
        (passes if r['passed'] else fails)[r['task']].append(r)
    kept_p = []
    for t in sorted(passes):
        rng.shuffle(passes[t])
        kept_p += passes[t][:2]
    n = min(len(kept_p), sum(len(v) for v in fails.values()), target // 2)
    rng.shuffle(kept_p)
    kept_p = kept_p[:n]
    kept_f, used = [], set()
    for p in kept_p:                               # same-task failure first
        for f in fails.get(p['task'], []):
            if f['trial'] not in used:
                kept_f.append(f)
                used.add(f['trial'])
                break
    rest = [f for t in sorted(fails) for f in fails[t] if f['trial'] not in used]
    rng.shuffle(rest)
    kept_f += rest[:n - len(kept_f)]
    return kept_p, kept_f


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--name')
    ap.add_argument('--target', type=int, default=2000)
    ap.add_argument('--seed', type=int, default=20260925)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    name = a.name or os.path.basename(os.path.normpath(a.run_dir))
    summary = {}
    with open(a.out, 'w') as f:
        for arm in ('control', 'relay_repair'):
            if not os.path.isdir(os.path.join(a.run_dir, f'router_{arm}')):
                continue
            rows = arm_rows(a.run_dir, name, arm)
            kp, kf = select(rows, a.target, a.seed)
            for r in kp + kf:
                f.write(json.dumps(r) + '\n')
            summary[arm] = dict(candidates=len(rows), candidate_passes=sum(r['passed'] for r in rows),
                                kept=len(kp) + len(kf), kept_passes=len(kp), kept_failures=len(kf),
                                same_task_pairs=len({p['task'] for p in kp} & {x['task'] for x in kf}),
                                failure_causes=dict(collections.Counter(x['cause'] for x in kf)))
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
