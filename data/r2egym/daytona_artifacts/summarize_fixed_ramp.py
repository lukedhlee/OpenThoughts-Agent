#!/usr/bin/env python3
"""Validate and tabulate a cold-heavy command ramp (command_ramp.py --cold-every N --cold-gap S).

Builds on summarize_command_ramp.py (per-stage counts checked against every command row and the
final cleanup census) and adds what the fixed-transport acceptance checks need:

- cold vs warm split: with --cold-every N the worker sleeps --cold-gap seconds after every N-th call,
  so the call with ordinal k > 0 and k % N == 0 reopens an idle connection (cold); ordinal 0 is the
  first call after seat setup (reported separately); every other call is warm.
- failed-seat concentration and the count of seats that never completed a command.
- the engineering thresholds from the 2026-09-15 plan: error rate < 0.1 %, warm p99 <= 2 s,
  cold p99 <= 5 s, every seat issued successful commands, cleanup verified.

No network calls. Writes <root>/validated_fixed_summary.json.
"""
import argparse
import collections
import json
from pathlib import Path

from summarize_command_ramp import summarize, q

ERROR_RATE_MAX = 0.001
WARM_P99_MAX = 2.0
COLD_P99_MAX = 5.0


def split(rows, cold_every):
    first, cold, warm = [], [], []
    for r in rows:
        k = r['ordinal']
        if k == 0:
            first.append(r)
        elif cold_every and k % cold_every == 0:
            cold.append(r)
        else:
            warm.append(r)
    return first, cold, warm


def stats(rows):
    good = [r['seconds'] for r in rows if r['ok']]
    return dict(commands=len(rows), errors=sum(not r['ok'] for r in rows),
                p50=q(good, .5), p90=q(good, .9), p99=q(good, .99), max=round(max(good), 4) if good else None)


def fmt(block):
    return f"{block['p50']} / {block['p99']} ({block['commands']})" if block['commands'] else '-'


def main(root):
    base = summarize(root)
    events = [json.loads(s) for s in (root / 'events.jsonl').read_text().splitlines()]
    meta = base['meta'] or {}
    cold_every = int((meta.get('args') or {}).get('cold_every') or 0)
    cold_gap = (meta.get('args') or {}).get('cold_gap')
    out = []
    for stage in base['stages']:
        rows = [r for r in events if r['kind'] == 'command' and r['stage'] == stage['stage']]
        stage_cold_every = 0 if stage['stage'] == 'warm1056' else cold_every
        first, cold, warm = split(rows, stage_cold_every)
        per_seat_err = collections.Counter(r['seat'] for r in rows if not r['ok'])
        seats_ok = {r['seat'] for r in rows if r['ok']}
        checks = dict(
            every_seat_succeeded=len(seats_ok) == stage['seats'],
            error_rate_ok=(stage['error_rate'] or 0) < ERROR_RATE_MAX,
            warm_p99_ok=(stats(warm)['p99'] or 0) <= WARM_P99_MAX if warm else None,
            cold_p99_ok=(stats(cold)['p99'] or 0) <= COLD_P99_MAX if cold else None,
        )
        out.append(dict(stage=stage['stage'], seats=stage['seats'], coordinators=stage['coordinators'],
                        commands=stage['commands'], errors=stage['errors'], error_rate=stage['error_rate'],
                        error_types=stage['error_types'],
                        seats_without_success=stage['seats'] - len(seats_ok),
                        max_errors_one_seat=max(per_seat_err.values(), default=0),
                        seats_with_errors=len(per_seat_err),
                        first=stats(first), cold=stats(cold), warm=stats(warm),
                        pooled=dict(p50=stage['success_p50'], p90=stage['success_p90'], p99=stage['success_p99']),
                        commands_per_second=stage['commands_per_second'], peak_inflight=stage['peak_inflight'],
                        loop_lag_max=stage['loop_lag_max'], command_kinds=stage['command_kinds'], checks=checks))
    creates = [r for r in events if r['kind'] == 'create']
    started = [r['seconds'] for r in creates if r['started']]
    setup_errors = [r for r in events if r['kind'] == 'setup_error']
    replacement = next((r for r in events if r['kind'] == 'replacement_complete'), None)
    report = dict(cold_every=cold_every, cold_gap=cold_gap, stages=out,
                  creates=dict(total=len(creates), started=sum(bool(r['started']) for r in creates),
                               retries_429=sum(r.get('retries_429', 0) for r in creates),
                               create_seconds_p50=q(started, .5), create_seconds_p99=q(started, .99)),
                  setup_errors=len(setup_errors), replacement=replacement,
                  cleanup_verified=base['cleanup_verified'], cleanup_final=base['cleanup_final'],
                  stages_present=[s['stage'] for s in out],
                  meta=dict(label=meta.get('label'), host=meta.get('host'), sdk_version=meta.get('sdk_version'),
                            driver_sha256=meta.get('driver_sha256'), args=meta.get('args')))
    (root / 'validated_fixed_summary.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('root', type=Path)
    a = p.parse_args()
    r = main(a.root)
    print(f"cold every {r['cold_every']} calls, gap {r['cold_gap']} s; creates {r['creates']['started']}/{r['creates']['total']} "
          f"started, 429 retries {r['creates']['retries_429']}, setup errors {r['setup_errors']}")
    print('| stage | seats | commands | errors | cold p50 / p99 | warm p50 / p99 | first p50 / p99 | pooled p99 | cmd/s | seats w/o success |')
    print('|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
    for s in r['stages']:
        rate = f"{s['error_rate']:.3%}" if s['error_rate'] is not None else '-'
        print(f"| {s['stage']} | {s['seats']} | {s['commands']:,} | {s['errors']} ({rate}) | "
              f"{fmt(s['cold'])} | {fmt(s['warm'])} | {fmt(s['first'])} | {s['pooled']['p99']} | "
              f"{s['commands_per_second']:.1f} | {s['seats_without_success']} |")
        print('  checks:', s['checks'], 'loop lag max', s['loop_lag_max'], 'peak inflight', s['peak_inflight'])
    print('replacement:', r['replacement'])
    print('cleanup verified:', r['cleanup_verified'], r['cleanup_final'])
