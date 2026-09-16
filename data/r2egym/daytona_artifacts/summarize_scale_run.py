#!/usr/bin/env python3
"""Tabulate a Daytona capacity run (scale_1024.sbatch / scale_tok256.sbatch): all node_*/c*_ files.

Reports, across all coordinators and per node: trials that reached agent execution, exception classes,
transport-class call errors, per-phase latency (tokenize / terminal_exec / generation) and trial phase
durations (environment setup, agent setup, agent execution, verifier) from the compact result files.
Applies the pass rule written before the run. No network calls.
"""
import argparse
import collections
import glob
import json
from datetime import datetime
from pathlib import Path

CONTEXT_ERRORS = {'ContextLengthExceededError', 'ContextBudgetExceededError'}
# asyncio cancellation raised inside a timed call when harbor's agent deadline fires; a harness deadline, not transport
DEADLINE_ERRORS = {'CancelledError'}
RULE = dict(reach_agent_min=0.99, transport_err_max=0.005, terminal_p50_max=2.5, terminal_p99_max=15.0,
            tokenize_p99_max=1.0, setup_p50_max=30.0)


def q(xs, p):
    xs = sorted(xs)
    return round(xs[int((len(xs) - 1) * p)], 3) if xs else None


def secs(t):
    if not t or not t.get('started_at') or not t.get('finished_at'):
        return None
    a, b = (datetime.fromisoformat(str(t[k]).replace('Z', '+00:00')) for k in ('started_at', 'finished_at'))
    return (b - a).total_seconds()


def main(root):
    calls = collections.defaultdict(list)
    call_errors = collections.Counter()
    transport_errors = 0
    deadline_cancels = collections.Counter()
    trials, per_node = [], collections.defaultdict(lambda: dict(trials=0, reached=0, exc=collections.Counter()))
    for tf in sorted(glob.glob(str(root / 'node_*' / 'c*_timing.jsonl'))):
        node = Path(tf).parent.name
        for line in open(tf):
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r['kind'] == 'environment_created':
                continue
            calls[r['kind']].append(r)
            if r.get('error'):
                call_errors[f"{r['kind']}:{r['error']}"] += 1
                if r['error'] in DEADLINE_ERRORS:
                    deadline_cancels[r['kind']] += 1
                elif r['error'] not in CONTEXT_ERRORS:
                    transport_errors += 1
    for rf in sorted(glob.glob(str(root / 'node_*' / 'c*_result.json'))):
        node = Path(rf).parent.name
        d = json.loads(open(rf).read())
        for t in d.get('trials', []):
            t['node'] = node
            trials.append(t)
            per_node[node]['trials'] += 1
            if t.get('agent_execution') and t['agent_execution'].get('started_at'):
                per_node[node]['reached'] += 1
            if t.get('exception'):
                per_node[node]['exc'][t['exception']] += 1
    n_calls = sum(len(v) for v in calls.values())
    kinds = {}
    for k, rows in calls.items():
        ok = [r['seconds'] for r in rows if not r.get('error')]
        kinds[k] = dict(calls=len(rows), errors=sum(bool(r.get('error')) for r in rows),
                        p50=q(ok, .5), p90=q(ok, .9), p99=q(ok, .99), max=round(max(ok), 3) if ok else None)
    reached = [t for t in trials if t.get('agent_execution') and t['agent_execution'].get('started_at')]
    exc = collections.Counter(t['exception'] for t in trials if t.get('exception'))
    rewards = [((t.get('reward') or {}).get('reward')) for t in trials]
    phases = {p: q([s for s in (secs(t.get(p)) for t in trials) if s is not None], .5) for p in
              ('environment_setup', 'agent_setup', 'agent_execution', 'verifier')}
    phases_p90 = {p: q([s for s in (secs(t.get(p)) for t in trials) if s is not None], .9) for p in
                  ('environment_setup', 'agent_setup', 'agent_execution', 'verifier')}
    start_to_agent = [s for s in (
        (datetime.fromisoformat(str(t['agent_execution']['started_at']).replace('Z', '+00:00')) -
         datetime.fromisoformat(str(t['started_at']).replace('Z', '+00:00'))).total_seconds()
        if t.get('started_at') and t.get('agent_execution') and t['agent_execution'].get('started_at') else None
        for t in trials) if s is not None]
    term, tok = kinds.get('terminal_exec', {}), kinds.get('tokenize', {})
    checks = dict(
        reach_agent=(len(reached) / len(trials) if trials else 0) >= RULE['reach_agent_min'],
        transport_errors=(transport_errors / n_calls if n_calls else 1) < RULE['transport_err_max'],
        terminal_p50=(term.get('p50') or 99) <= RULE['terminal_p50_max'],
        terminal_p99=(term.get('p99') or 99) <= RULE['terminal_p99_max'],
        tokenize_p99=((tok.get('p99') or 99) <= RULE['tokenize_p99_max']) if tok else None,
        setup_p50=(q(start_to_agent, .5) or 99) <= RULE['setup_p50_max'],
    )
    report = dict(trials=len(trials), reached_agent=len(reached), exceptions=dict(exc),
                  rewards=dict(scored=sum(r is not None for r in rewards), mean=(round(sum(r for r in rewards if r is not None) / max(1, sum(r is not None for r in rewards)), 3))),
                  calls=n_calls, transport_errors=transport_errors, deadline_cancels=dict(deadline_cancels),
                  call_errors=dict(call_errors), kinds=kinds,
                  phase_p50=phases, phase_p90=phases_p90,
                  start_to_agent=dict(p50=q(start_to_agent, .5), p90=q(start_to_agent, .9), p99=q(start_to_agent, .99)),
                  per_node={n: dict(trials=v['trials'], reached=v['reached'], exc=dict(v['exc'])) for n, v in sorted(per_node.items())},
                  checks=checks, rule=RULE)
    (root / 'validated_scale_summary.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('root', type=Path)
    r = main(p.parse_args().root)
    print(f"trials {r['trials']}, reached agent {r['reached_agent']}, exceptions {r['exceptions']}, rewards {r['rewards']}")
    print(f"calls {r['calls']}, transport errors {r['transport_errors']}, deadline cancellations {r['deadline_cancels']}, all call errors {r['call_errors']}")
    for k, v in r['kinds'].items():
        print(f"  {k:14s} calls {v['calls']:6d} errors {v['errors']:4d} p50 {v['p50']} p90 {v['p90']} p99 {v['p99']} max {v['max']}")
    print('phase p50 s', r['phase_p50'], '| p90', r['phase_p90'])
    print('trial start -> agent start s', r['start_to_agent'])
    for n, v in r['per_node'].items():
        print(f"  {n}: trials {v['trials']} reached {v['reached']} exc {v['exc']}")
    print('CHECKS', r['checks'])
