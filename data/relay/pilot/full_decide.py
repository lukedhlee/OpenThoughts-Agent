#!/usr/bin/env python3
"""Launch rule for the full run (notes/relay/relay_full_t2.md): read run 3's readout and logs, recompute S6 for the
10-node layout, and say LAUNCH or HOLD with the reason. It never submits anything; the caller does, on LAUNCH only.

    python full_decide.py --run3 <run 3 dir> [--pool 2043] [--ceiling 35] > decision.json

LAUNCH needs all of: every harness check of run 3 passed; S1-S6 passed; the projected node-hours of the 10-node
layout <= BORDERLINE x ceiling (default 0.85 x 35); projected kept traces >= 1,500 per arm (the 2,000 target, less
what one control rollout per task can give). Anything else is HOLD.

Projection (from run 3's measured episode wall times, pass rates and startup):
  control     pool x 1 episode, CONC_BASE concurrent (8 Qwen nodes x 50)
  relay p1    pool x 1 episode, CONC_RELAY concurrent (2 student nodes x 100)
  relay p2    2 x (tasks control solves) episodes, same concurrency, after p1
  wall        startup + max(control, relay p1) + relay p2 + tail (run 3's p90 relay episode)
  node-hours  6 core nodes x wall + 4 burst nodes x (startup + control + 10 min drain)
"""
import argparse
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import readout  # noqa: E402


def episode_walls(run_dir, arm):
    rv = readout.router_view(os.path.join(run_dir, f'router_{arm}'))
    out = []
    for e in rv['eps'].values():
        ts = [r['ts'] for r in e['main'] + e['aux']]
        if ts and e['ending'] != 'deadline':
            out.append(max(ts) - min(ts) + 60)      # + setup and verification, about a minute on Daytona
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run3', required=True)
    ap.add_argument('--readout', help="a readout.json other than the run's own (e.g. recomputed under the amended rule)")
    ap.add_argument('--pool', type=int, default=2043)
    ap.add_argument('--ceiling', type=float, default=35.0)
    ap.add_argument('--borderline', type=float, default=0.85)
    ap.add_argument('--conc-base', type=int, default=400)
    ap.add_argument('--student-nodes', type=int, default=2, help='100 relay agents per student node')
    ap.add_argument('--core-qwen', type=int, default=4, help='Qwen nodes kept to the end (the rest are released)')
    ap.add_argument('--burst-qwen', type=int, default=4, help='Qwen nodes released once control drains')
    ap.add_argument('--relay-episode-s', type=float, default=None,
                    help="override run 3's mean relay episode wall (e.g. adjusted for autofix and the cap)")
    ap.add_argument('--relay-pass', type=float, default=None, help="override run 3's relay pass rate")
    ap.add_argument('--startup-h', type=float, default=0.15)
    ap.add_argument('--min-kept', type=int, default=1500)
    a = ap.parse_args()
    r = json.load(open(a.readout or os.path.join(a.run3, 'readout.json')))
    reasons = []
    bad_h = [c['check'] for c in r.get('harness_checks', []) if not c['ok']]
    bad_s = [c['check'] for c in r.get('science_checks', []) if not c['ok']]
    if bad_h:
        reasons.append(f'harness gate failed: {bad_h}')
    if not r.get('science_checks'):
        reasons.append('no science checks in the readout')
    if bad_s:
        reasons.append(f'scale-up checks failed: {bad_s}')
    db, dr = episode_walls(a.run3, 'control'), episode_walls(a.run3, 'relay_repair')
    ctl, rel = r['control'].get('outcomes') or {}, r['relay_repair'].get('outcomes') or {}
    pb = ctl.get('pass_rate') or 0.0
    rp = r['relay_repair'].get('repair') or {}
    pr = rel.get('pass_rate') or 0.0
    proj = {}
    a.conc_relay = 100 * a.student_nodes
    core_nodes = a.student_nodes + a.core_qwen
    if a.relay_pass is not None:
        pr = a.relay_pass
    if db and dr:
        mb, mr, tail = statistics.mean(db), statistics.mean(dr), readout.q(dr, .9)
        if a.relay_episode_s:
            tail, mr = tail * a.relay_episode_s / mr, a.relay_episode_s
        t_base = a.pool * mb / a.conc_base
        t_r1 = a.pool * mr / a.conc_relay
        n_p2 = 2 * round(a.pool * pb)
        t_r2 = n_p2 * mr / a.conc_relay
        wall = a.startup_h * 3600 + max(t_base, t_r1) + t_r2 + tail
        burst = a.startup_h * 3600 + t_base + 600
        nh = (core_nodes * wall + a.burst_qwen * burst) / 3600
        # kept: control 1 rollout per task; relay p1 + p2 (p2 on solved tasks, pass rate taken from run 3's relay
        # episodes on tasks control passed), passes capped at 2 per task
        pp = r.get('paired_pass_relay_vs_control') or {}
        p_rel_given_ctl = (pp.get('both_pass', 0) / max(1, pp.get('both_pass', 0) + pp.get('control_only', 0)))
        kept_ctl = 2 * min(round(a.pool * pb), round(a.pool * (1 - pb) * (ctl.get('real_failures', 0) / max(1, ctl.get('scored', 1) - ctl.get('passes', 0)))), 1000)
        rel_pass = a.pool * pr + n_p2 * p_rel_given_ctl
        rel_pass_capped = min(rel_pass, 2 * a.pool)
        rel_fail = a.pool * (1 - pr) + n_p2 * (1 - p_rel_given_ctl)
        kept_rel = 2 * min(round(rel_pass_capped), round(rel_fail), 1000)
        proj = dict(control_episode_mean_s=round(mb), relay_episode_mean_s=round(mr), relay_episode_p90_s=round(tail),
                    control_pass=pb, relay_pass=pr, relay_pass_given_control_pass=round(p_rel_given_ctl, 3),
                    phase2_episodes=n_p2, hours_control=round(t_base / 3600, 2), hours_relay_p1=round(t_r1 / 3600, 2),
                    hours_relay_p2=round(t_r2 / 3600, 2), wall_h=round(wall / 3600, 2),
                    burst_h=round(burst / 3600, 2), node_h=round(nh, 1), kept_control=kept_ctl, kept_relay=kept_rel,
                    layout=f'{a.student_nodes} x 09-21 + {a.core_qwen + a.burst_qwen} x Qwen ({a.burst_qwen} released after control)',
                    core_time_h=round(min(wall / 3600 * 1.25, (a.ceiling - a.burst_qwen * burst * 1.25 / 3600) / core_nodes), 2),
                    burst_time_h=round(burst / 3600 * 1.25, 2))
        if nh > a.borderline * a.ceiling:
            reasons.append(f'projected {nh:.1f} node-h > {a.borderline} x {a.ceiling} ceiling (S6 for the 10-node layout)')
        if proj['core_time_h'] < wall / 3600:
            reasons.append(f'the ceiling leaves the core job {proj["core_time_h"]} h < projected wall {wall / 3600:.2f} h')
        if min(kept_ctl, kept_rel) < a.min_kept:
            reasons.append(f'projected kept traces control {kept_ctl} / relay {kept_rel} < {a.min_kept}')
    else:
        reasons.append('no episode wall times in run 3')
    out = dict(decision='LAUNCH' if not reasons else 'HOLD', reasons=reasons, projection=proj,
               run3_verdict=r.get('verdict'))
    print(json.dumps(out, indent=1))
    sys.exit(0 if not reasons else 2)


if __name__ == '__main__':
    main()
