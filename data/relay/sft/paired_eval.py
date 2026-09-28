#!/usr/bin/env python3
"""Relay SFT comparison (pre-registered 2026-09-28, notes/relay/sft_launch_plan.md "Eval"): pass rates with CIs for
09-21, A (relay, 09-21 turns masked), B (Qwen alone) and C (relay + autofix loss) on the 300 held-out CalibForge tasks
and on TB2, paired differences, and the decision-rule verdict.

    python paired_eval.py --heldout M=<held-out run dir> ... --tb2 M=<TB2 harbor job dir> ... --out <results.json> [--md <table.md>]
    (M in 0921, A, B, C; a model given twice (a retry) is merged per task, the later directory winning)

Outcomes. Held-out: every trial of the run's student_only harbor job(s) (readout.trials, both staggered halves),
usable = not a harness error, not a verifier timeout, not censored by the run's deadline (readout.usable; with CLOCK=wall
an AgentTimeoutError / ContextLengthExceededError is a scored model failure); pass = reward >= 1. TB2: every trial of the
harbor job dir, usable = the verifier returned a reward (summarize_tb2.py's "scored"; AgentTimeoutError and the other
model outcomes are scored); pass = reward >= 1. One trial per task, so a model's per-task outcome is 0/1.
Pass rates: over that model's usable tasks, Wilson 95 % CI. Paired differences (X - Y): tasks usable for both, mean of
the per-task difference, 95 % bootstrap CI over tasks (10,000 resamples, seed 20260928).
Decision rule: A wins iff the held-out paired A - B CI lies above 0 AND TB2 is not worse (the paired TB2 A - B CI does
not lie entirely below 0). Sanity: an SFT arm "beats 09-21" iff its held-out paired gain over 09-21 has a CI above 0;
if neither A nor B beats 09-21, the SFT failed and the A-vs-B comparison says nothing.
"""
import argparse
import collections
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'pilot'))
import readout  # noqa: E402

SEED, NBOOT = 20260928, 10000
MODELS = ('0921', 'A', 'B', 'C')


def heldout_outcomes(run_dir):
    name = os.path.basename(os.path.normpath(run_dir))
    rows = readout.arm_trials(run_dir, name, 'student_only')
    rdir = os.path.join(run_dir, 'router_student_only')
    if os.path.isdir(rdir):
        readout.mark_censored(readout.router_view(rdir), rows)
    out, c = {}, collections.Counter()
    for t in rows:
        c['trials'] += 1
        if readout.usable(t):
            out[t['task']] = int(readout.is_pass(t))
        else:
            c['unusable'] += 1
    return out, dict(c)


def tb2_outcomes(job_dir):
    out, c = {}, collections.Counter()
    for t in readout.trials(job_dir):
        c['trials'] += 1
        if t['reward'] is None:
            c['unscored'] += 1
            continue
        out[t['task']] = int((t['reward'] or 0) >= 1)
    return out, dict(c)


def rate(o):
    k, n = sum(o.values()), len(o)
    lo, hi = readout.wilson(k, n) if n else (None, None)
    return dict(n=n, passes=k, rate=round(k / n, 4) if n else None, ci95=[round(lo, 4), round(hi, 4)] if n else None)


def paired(x, y):
    both = sorted(set(x) & set(y))
    d = [x[t] - y[t] for t in both]
    if not d:
        return dict(tasks=0, diff=None, ci95=None)
    rng = random.Random(SEED)
    ms = sorted(sum(rng.choice(d) for _ in d) / len(d) for _ in range(NBOOT))
    return dict(tasks=len(both), diff=round(sum(d) / len(d), 4), ci95=[round(ms[int(0.025 * NBOOT)], 4),
                round(ms[int(0.975 * NBOOT) - 1], 4)], x_only=sum(1 for t in both if x[t] and not y[t]),
                y_only=sum(1 for t in both if y[t] and not x[t]))


def parse(specs):
    m = collections.defaultdict(list)
    for s in specs or []:
        k, v = s.split('=', 1)
        assert k in MODELS, f'unknown model {k}'
        m[k].append(v)
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--heldout', nargs='*')
    ap.add_argument('--tb2', nargs='*')
    ap.add_argument('--out', required=True)
    ap.add_argument('--md')
    a = ap.parse_args()
    res = dict(heldout={}, tb2={}, paired=dict(heldout={}, tb2={}), counts=dict(heldout={}, tb2={}))
    oc = dict(heldout={}, tb2={})
    for bench, specs, fn in (('heldout', parse(a.heldout), heldout_outcomes), ('tb2', parse(a.tb2), tb2_outcomes)):
        for m, dirs in specs.items():
            merged, cnt = {}, collections.Counter()
            for d in dirs:
                o, c = fn(d)
                merged.update(o)
                cnt.update(c)
            oc[bench][m] = merged
            res[bench][m] = rate(merged)
            res['counts'][bench][m] = dict(cnt, dirs=dirs)
        for x, y in (('A', 'B'), ('C', 'A'), ('A', '0921'), ('B', '0921'), ('C', '0921'), ('C', 'B')):
            if x in oc[bench] and y in oc[bench]:
                res['paired'][bench][f'{x}-{y}'] = paired(oc[bench][x], oc[bench][y])
    P = res['paired']
    above = lambda p: bool(p and p.get('ci95') and p['ci95'][0] > 0)  # noqa: E731
    below = lambda p: bool(p and p.get('ci95') and p['ci95'][1] < 0)  # noqa: E731
    ab_h, ab_t = P['heldout'].get('A-B'), P['tb2'].get('A-B')
    sanity = {x: above(P['heldout'].get(f'{x}-0921')) for x in ('A', 'B', 'C')}
    complete = bool(ab_h and ab_h.get('ci95') and ab_t and ab_t.get('ci95'))
    res['decision'] = dict(
        complete=complete,
        heldout_A_minus_B_ci_above_0=above(ab_h), tb2_A_not_worse=(not below(ab_t)) if ab_t else None,
        sanity_beats_0921=sanity, sft_failed=not (sanity['A'] or sanity['B']),
        verdict=('incomplete' if not complete else
                 'SFT failed: neither A nor B beats 09-21 on held-out; A vs B says nothing' if not (sanity['A'] or sanity['B']) else
                 'A wins: scale the relay' if above(ab_h) and not below(ab_t) else
                 'A does not win: scale Qwen-alone traces'),
        autofix_loss_C_minus_A=dict(heldout=P['heldout'].get('C-A'), tb2=P['tb2'].get('C-A')))
    json.dump(res, open(a.out, 'w'), indent=1)
    lines = ['| model | held-out 300: pass rate [95 % CI] (n) | TB2: pass rate [95 % CI] (n) |', '|---|---|---|']
    f = lambda r: f"{r['rate']:.3f} [{r['ci95'][0]:.3f}, {r['ci95'][1]:.3f}] ({r['n']})" if r and r.get('n') else '—'  # noqa: E731
    names = {'0921': '09-21 (before)', 'A': 'A relay, 09-21 turns masked', 'B': 'B Qwen alone', 'C': 'C relay + autofix loss'}
    for m in MODELS:
        lines.append(f"| {names[m]} | {f(res['heldout'].get(m))} | {f(res['tb2'].get(m))} |")
    lines += ['', '| paired (X - Y, tasks both usable) | held-out diff [95 % CI] (tasks) | TB2 diff [95 % CI] (tasks) |', '|---|---|---|']
    g = lambda p: f"{p['diff']:+.3f} [{p['ci95'][0]:+.3f}, {p['ci95'][1]:+.3f}] ({p['tasks']})" if p and p.get('ci95') else '—'  # noqa: E731
    for k in ('A-B', 'C-A', 'A-0921', 'B-0921', 'C-0921'):
        lines.append(f"| {k} | {g(P['heldout'].get(k))} | {g(P['tb2'].get(k))} |")
    lines += ['', f"**Decision rule:** {res['decision']['verdict']}"]
    md = '\n'.join(lines) + '\n'
    if a.md:
        open(a.md, 'w').write(md)
    print(md)


if __name__ == '__main__':
    main()
