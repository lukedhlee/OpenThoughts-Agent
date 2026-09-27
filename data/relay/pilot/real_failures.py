#!/usr/bin/env python3
"""Count the real failures of one or more run dirs of an arm, with match_kept.py's definitions, for the "stop on the
number" rule of a top-up (baseline 6d, 2026-09-27):

    python real_failures.py --arm control --quarantine quarantine_tasks.txt <run dir> [<run dir> ...]

A real failure is a kept-candidate failure (select_kept.arm_rows: scored, reward < 1, the relay arm's teacher/S5
filters) of a task not in the quarantine, that is not a weak timeout (match_kept.weak_timeout: a timeout failure
that had not stalled). Prints one JSON line: per run and total real failures, passes, weak timeouts and failure
causes. Read-only; login-node safe (stdlib + the pilot modules).
"""
import argparse
import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import match_kept  # noqa: E402
import select_kept  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--arm', default='control')
    ap.add_argument('--quarantine', action='append', default=[])
    ap.add_argument('runs', nargs='+')
    a = ap.parse_args()
    quarantine = {t.strip() for q in a.quarantine for t in open(q) if t.strip()}
    out, total = {}, collections.Counter()
    for run in a.runs:
        name = os.path.basename(os.path.normpath(run))
        rows = [r for r in select_kept.arm_rows(run, name, a.arm) if r['task'] not in quarantine]
        fails = [r for r in rows if not r['passed']]
        weak = [r for r in fails if match_kept.weak_timeout(r)]
        c = dict(candidates=len(rows), passes=len(rows) - len(fails), failures=len(fails), weak_timeouts=len(weak),
                 real_failures=len(fails) - len(weak),
                 causes=dict(collections.Counter(r['cause'] for r in fails if not match_kept.weak_timeout(r))))
        out[name] = c
        for k in ('candidates', 'passes', 'failures', 'weak_timeouts', 'real_failures'):
            total[k] += c[k]
    print(json.dumps(dict(runs=out, total=dict(total), quarantined_tasks=len(quarantine))))


if __name__ == '__main__':
    main()
