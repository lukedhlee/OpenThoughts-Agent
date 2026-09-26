#!/usr/bin/env python3
"""Pick the kept traces of a relay run: 1:1 pass:fail per arm, at most 2 kept passes per task, failures only real
model failures (harness errors, verifier timeouts and deadline-censored episodes dropped), each failure labelled by
cause (readout.failure_cause). relay_repair episodes count only when the teacher wrote at least one turn (repair or
takeover), since only teacher turns are trained on. S5 as a keep filter (Luke 2026-09-25 17:40 PT): a done_claim
takeover episode is kept only if the teacher ran at least one command before confirming. A timeout failure is kept
only if the episode had stalled (stalled(): a loop / no-progress trigger fired, or no new terminal output in its last
3 turns).

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


def teacher_worked(e):
    """Did the teacher run at least one command between its done_claim takeover and the episode's end?"""
    t0 = e['takeover']['turn']
    for r in sorted((r for r in e['main'] if r.get('owner') == 'teacher' and (r.get('turn') or 0) >= t0),
                    key=lambda r: r['turn']):
        c = (((r.get('response') or {}).get('choices') or [{}])[0].get('message') or {}).get('content') or ''
        if readout.rt.parse_reply(c, 'terminus2')['cmds']:
            return True
    return False


STALL_TRIGGERS = ('loop', 'no_progress_wait')


def stalled(e, t):
    """A timeout counts as a real (kept) failure only if the episode had actually stalled (Luke 2026-09-25 21:15 PT):
    a stall trigger (exact-repeat loop or no-progress wait) fired on it (as a takeover, or, in control, as would_fire),
    or its last 3 executed turns produced no new terminal output (relay_triggers' own "no new output" rule)."""
    if (e['takeover'] or {}).get('trigger') in STALL_TRIGGERS:
        return 'trigger'
    if any(f.get('trigger') in STALL_TRIGGERS for r in e['main'] for f in (r.get('would_fire') or [])):
        return 'trigger'
    if not t.get('traj_path'):
        return None
    try:
        traj = json.load(open(t['traj_path']))
    except (OSError, ValueError):
        return None
    msgs = []
    for s in traj['steps']:
        if s.get('source') == 'user' and not msgs:
            msgs.append(dict(role='user', content=s.get('message') or ''))
        elif s.get('source') == 'agent' and not s.get('is_copied_context'):
            msgs.append(dict(role='assistant', content=s['message'] if isinstance(s.get('message'), str) else json.dumps(s.get('message'))))
            obs = '\n'.join(r.get('content') or '' for r in (s.get('observation') or {}).get('results', [])
                            if isinstance(r.get('content'), str))
            msgs.append(dict(role='user', content=obs))
    sc = readout.rt.scan_messages(msgs, harness='terminus2')
    acts = [a for a in sc.actions if not a['mark']]
    return 'no_new_output' if len(acts) >= 3 and all(a['empty'] for a in acts[-3:]) else None


def arm_rows(run_dir, name, arm, timeouts='all'):
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
        if (e['takeover'] or {}).get('trigger') == 'done_claim' and not teacher_worked(e):
            continue
        out.append(dict(arm=arm, task=t['task'], trial=t['trial'], sid=t['sid'], reward=t['reward'],
                        passed=readout.is_pass(t),
                        cause=None if readout.is_pass(t) else readout.failure_cause(t, e['ending']),
                        teacher_turns=teacher_turns, student_turns=sum(1 for r in e['main'] if r.get('owner') == 'student'),
                        repairs=sum(1 for r in e['main'] if r.get('repair_kind') == 'parse_error'),
                        takeover=(e['takeover'] or {}).get('trigger'),
                        teacher_cut_turns=sum(1 for r in e['main'] if r.get('teacher_cut_at_cap'))))
        x = out[-1]
        if x['cause'] == 'timeout':
            x['stalled'] = stalled(e, t)
            if timeouts == 'stalled' and not x['stalled']:
                out.pop()        # old clock: a timeout on a working episode was often serving latency, not the model
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
    ap.add_argument('--timeouts', choices=['all', 'stalled'], default='all',
                    help="all: every timeout is a real failure (runs whose clock pauses on model calls, 21:15 PT on); "
                         "stalled: only stalled timeouts (runs under the old clock, which charged serving latency)")
    a = ap.parse_args()
    name = a.name or os.path.basename(os.path.normpath(a.run_dir))
    summary = {}
    with open(a.out, 'w') as f:
        for arm in ('control', 'relay_repair'):
            if not os.path.isdir(os.path.join(a.run_dir, f'router_{arm}')):
                continue
            rows = arm_rows(a.run_dir, name, arm, a.timeouts)
            kp, kf = select(rows, a.target, a.seed)
            for r in kp + kf:
                f.write(json.dumps(r) + '\n')
            summary[arm] = dict(candidates=len(rows), candidate_passes=sum(r['passed'] for r in rows),
                                kept=len(kp) + len(kf), kept_passes=len(kp), kept_failures=len(kf),
                                same_task_pairs=len({p['task'] for p in kp} & {x['task'] for x in kf}),
                                failure_causes=dict(collections.Counter(x['cause'] for x in kf)),
                                kept_timeouts_stalled=dict(collections.Counter(str(x.get('stalled')) for x in kf
                                                                               if x['cause'] == 'timeout')))
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
