#!/usr/bin/env python3
"""Plan the full run's relay arm from the finished baseline and the check run (Luke 2026-09-25 18:40 PT).

The relay arm runs only on the tasks the baseline (Qwen alone, 1 rollout on the 2,043-task pool) solved, with r
rollouts per task, on 4 x 09-21 + 4 x Qwen (8 nodes, 400 relay agents). Its ceiling is what the full run's 42
node-hours leave after the baseline.

    python relay_plan.py --baseline <baseline run dir> --check <check run dir> --check-decision <json> \
        --out-tasks solvable.txt > plan.json          # exit 0 LAUNCH, 2 HOLD

  solvable      baseline passes (reward >= 1, no harness error, not censored)
  p             the check run's relay pass rate on tasks run 3's control solved (the same kind of task)
  r             the smallest of 1..4 whose expected kept relay traces, 2 x min(passes capped at 2 per task,
                real failures, 1000), reach 2,000; else 3 (Luke's default) and the shortfall is reported
  wall          startup + n x r x D / 400 + the check run's p90 relay episode (the tail); D = its mean episode
  node-hours    8 x wall; LAUNCH only if <= 42 - the baseline's node-hours (the relay job's --time is that / 8)
"""
import argparse
import glob
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import readout  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--baseline', required=True)
    ap.add_argument('--check', required=True)
    ap.add_argument('--check-decision', required=True)
    ap.add_argument('--run3', default='/e/fscratch/reformo/lee27/experiments/relay/pilot/runs/relay_run3b_20260925')
    ap.add_argument('--total-ceiling', type=float, default=42.0)
    ap.add_argument('--nodes', type=int, default=8)
    ap.add_argument('--conc', type=int, default=400)
    ap.add_argument('--startup-h', type=float, default=0.2)
    ap.add_argument('--target', type=int, default=2000)
    ap.add_argument('--out-tasks', required=True)
    a = ap.parse_args()
    bname = os.path.basename(os.path.normpath(a.baseline))
    meta = dict(l.strip().split('=', 1) for l in open(os.path.join(a.baseline, 'run.meta')) if '=' in l)
    base_nh = float(meta['node_hours'])
    rows = readout.trials(os.path.join(a.baseline, 'jobs', f'{bname}_control'))
    readout.mark_censored(readout.router_view(os.path.join(a.baseline, 'router_control')), rows)
    solved = sorted({t['task'] for t in rows if readout.usable(t) and readout.is_pass(t)})
    open(a.out_tasks, 'w').write(''.join(t + '\n' for t in solved))
    dec = json.load(open(a.check_decision))
    cname = os.path.basename(os.path.normpath(a.check))
    chk = readout.trials(os.path.join(a.check, 'jobs', f'{cname}_relay_repair'))
    r3 = readout.trials(os.path.join(a.run3, 'jobs', f'{os.path.basename(os.path.normpath(a.run3))}_control'))
    pp = readout.paired_pass(r3, chk)
    p = pp['both_pass'] / max(1, pp['both_pass'] + pp['control_only'])
    real_fail = 1.0   # every non-pass on a scored episode is a real failure; harness errors are budgeted below
    usable_share = sum(1 for t in chk if readout.usable(t)) / max(1, len(chk))
    walls = []
    rv = readout.router_view(os.path.join(a.check, 'router_relay_repair'))
    for e in rv['eps'].values():
        ts = [x['ts'] for x in e['main'] + e['aux']]
        if ts and e['ending'] != 'deadline':
            walls.append(max(ts) - min(ts) + 60)
    d, tail = sum(walls) / len(walls), readout.q(walls, .9)
    n = len(solved)
    plan = None
    for r in (1, 2, 3, 4):
        eps = n * r * usable_share
        passes = min(eps * p, 2 * n)
        fails = eps * (1 - p) * real_fail
        kept = 2 * min(passes, fails, a.target / 2)
        if kept >= a.target * 0.98 or r == 4:
            plan = (r, kept)
            break
    r, kept = plan if plan[1] >= a.target * 0.98 else (3, 2 * min(min(n * 3 * usable_share * p, 2 * n),
                                                                n * 3 * usable_share * (1 - p), a.target / 2))
    wall_s = a.startup_h * 3600 + n * r * d / a.conc + tail
    node_h = a.nodes * wall_s / 3600
    ceiling = a.total_ceiling - base_nh
    time_h = math.floor(ceiling / a.nodes * 60) / 60
    reasons = []
    if dec.get('decision') != 'PASS':
        reasons.append(f"check run {dec.get('decision')}: {[c['check'] for c in dec.get('checks', []) if not c['ok']]}")
    if node_h > ceiling:
        reasons.append(f'projected relay {node_h:.1f} node-h > {ceiling:.1f} left of {a.total_ceiling} after the baseline')
    if time_h * 3600 < wall_s:
        reasons.append(f'the ceiling allows {time_h:.2f} h of wall, projected {wall_s / 3600:.2f} h')
    out = dict(decision='LAUNCH' if not reasons else 'HOLD', reasons=reasons, baseline_node_h=base_nh,
               baseline_trials=len(rows), solvable_tasks=n, relay_pass_on_solvable=round(p, 3),
               relay_usable_share=round(usable_share, 3), rollouts_per_task=r, relay_episodes=n * r,
               expected_kept_relay=round(kept), relay_episode_mean_s=round(d), relay_episode_p90_s=round(tail),
               relay_wall_h=round(wall_s / 3600, 2), relay_node_h=round(node_h, 1), relay_ceiling_node_h=round(ceiling, 2),
               relay_time_h=time_h, total_projected_node_h=round(base_nh + node_h, 1), tasks_file=a.out_tasks)
    print(json.dumps(out, indent=1))
    sys.exit(0 if not reasons else 2)


if __name__ == '__main__':
    main()
