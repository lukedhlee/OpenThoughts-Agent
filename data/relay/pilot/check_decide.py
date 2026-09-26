#!/usr/bin/env python3
"""Pass rule of the check run (relay_repair with every fix, the pilot's 100 tasks), pre-registered in
notes/relay/relay_full_t2.md before its job started (Luke 2026-09-25 18:25 PT). PASS -> the 12-node full run launches.

    python check_decide.py --check <check run dir> --run3 <run 3 dir>     # prints JSON, exit 0 PASS / 2 FAIL

  C1 harness gate: every harness check of the check run's readout (H0, H1 router + trials, H2, H3 agent turns, H4) and
     H5, the executed trace >= 99 % valid format
  C2 relay context overflow <= 20 % of scored relay episodes
  C3 (S2) the student owns >= 50 % of executed turns (autofixed turns are the student's)
  C4 (S4) recovery after a sticky takeover >= 0.20
  C5 relay pass rate not below run 3's control by more than 15 points, paired by task (mean over tasks scored in both)
"""
import argparse
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import readout  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--check', required=True)
    ap.add_argument('--run3', required=True)
    ap.add_argument('--rule', choices=['check', 'recheck'], default='check',
                    help="recheck: the 21:15 PT re-check's pre-registered rule (C3 = zero Qwen context 400s; the paired "
                         "pass vs run 3's control is informational, the clock change makes it not like-for-like)")
    a = ap.parse_args()
    cname, rname = (os.path.basename(os.path.normpath(x)) for x in (a.check, a.run3))
    # re-read the check run with this readout (the run's own final readout may predate an amendment)
    import subprocess
    amended = os.path.join(a.check, 'readout_decide.json')
    subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'readout.py'), '--run-dir',
                    a.check, '--gate', 'final', '--json', amended], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    r = json.load(open(amended if os.path.exists(amended) else os.path.join(a.check, 'readout.json')))
    o = r['relay_repair']
    rows = readout.trials(os.path.join(a.check, 'jobs', f'{cname}_relay_repair'))
    readout.mark_censored(readout.router_view(os.path.join(a.check, 'router_relay_repair')), rows)
    ctl = readout.trials(os.path.join(a.run3, 'jobs', f'{rname}_control'))
    checks = []

    def check(name, ok, detail):
        checks.append(dict(check=name, ok=bool(ok), detail=detail))
    bad = [c['check'] for c in r.get('harness_checks', []) if not c['ok']]
    fv = o.get('format') or {}
    check('C1 harness gate (H0-H4) and H5 valid format >= 99 %', not bad and (fv.get('valid_format_rate') or 0) >= 0.99,
          dict(failed=bad, format=fv))
    oc = o.get('outcomes') or {}
    scored = [t for t in rows if readout.usable(t) or t['exc'] == 'ContextLengthExceededError']
    ovf = sum(1 for t in scored if t['exc'] == 'ContextLengthExceededError')
    check('C2 relay context overflow <= 20 % of scored episodes', scored and ovf / len(scored) <= 0.20,
          dict(overflow=ovf, scored=len(scored), rate=round(ovf / len(scored), 4) if scored else None))
    share = (o.get('repair') or {}).get('student_share_of_executed_turns')
    tk = o.get('takeovers') or {}
    pp = readout.paired_pass(ctl, rows)
    rv0 = readout.router_view(os.path.join(a.check, 'router_relay_repair'))
    q400 = [x for x in rv0['recs'] if x.get('owner') == 'teacher' and x.get('upstream_status') == 400
            and 'maximum context length' in (x.get('upstream_error') or '')]
    if a.rule == 'recheck':
        check('C3 zero Qwen context-length 400s', not q400,
              dict(final_400s=len(q400), cap_dropped_then_ok=sum(1 for x in rv0['recs'] if x.get('teacher_max_tokens_dropped')
                                                               and x.get('upstream_status') == 200)))
        check('C4 student owns >= 50 % of executed turns', (share or 0) >= 0.5, share)
        check('C5 recovery after a sticky takeover >= 0.20', (tk.get('recovery') or 0) >= 0.20,
              [tk.get('recovery'), tk.get('recovery_ci95'), tk.get('takeover_episodes')])
    else:
        check('C3 student owns >= 50 % of executed turns', (share or 0) >= 0.5, share)
        check('C4 recovery after a sticky takeover >= 0.20', (tk.get('recovery') or 0) >= 0.20,
              [tk.get('recovery'), tk.get('recovery_ci95'), tk.get('takeover_episodes')])
        check('C5 relay pass not below run 3 control by more than 15 points (paired)',
              pp['relay_minus_control'] is not None and pp['relay_minus_control'] >= -0.15, pp)
    walls = []
    rv = readout.router_view(os.path.join(a.check, 'router_relay_repair'))
    for e in rv['eps'].values():
        ts = [x['ts'] for x in e['main'] + e['aux']]
        if ts and e['ending'] != 'deadline':
            walls.append(max(ts) - min(ts) + 60)
    out = dict(decision='PASS' if all(c['ok'] for c in checks) else 'FAIL', checks=checks,
               paired_pass_vs_run3_control=pp,
               relay_pass=oc.get('pass_rate'), relay_pass_ci95=oc.get('pass_rate_ci95'),
               relay_episode_mean_s=round(statistics.mean(walls)) if walls else None,
               autofixes=sum(1 for x in rv['recs'] if x.get('autofix')),
               repairs=(o.get('repair') or {}).get('repairs'),
               teacher_cut_at_cap=o['reasoning'].get('teacher_replies_cut_at_cap'),
               cap_retry_400_overflows=sum(1 for t in rows if t.get('cap_retry_400')))
    print(json.dumps(out, indent=1))
    sys.exit(0 if out['decision'] == 'PASS' else 2)


if __name__ == '__main__':
    main()
