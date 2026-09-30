#!/usr/bin/env python3
"""Census of the Horizon relay remainder runs (09-29, student = relay SFT arm A) as SFT candidates, with the three QA
drop filters tagged per episode, and the rows 09-28's final_v2 rule would keep with and without them.

Candidates: select_kept.arm_rows (scored episodes, the teacher wrote a turn, S5 done_claim filter, stalled check).
Eligibility as final_v2.py: weak timeouts dropped, the verifier's pytest ran (both job halves). The report's over_64k
count uses the largest prompt + completion of the episode's main turns, which overstates the rendered row (the render
cuts earlier turns' thinking): it drops 2,585 episodes where the render drops ~1 %, so read the scenarios from
census.jsonl without that filter (09-30: 1,819 + 1,819 rows vs the Mac build's 1,791 + 1,791 after the render's length check).
Tags per episode:
  leak     a teacher reply (content or reasoning; trained text) mentions the verify note's framing: another / the
           previous / prior / earlier / other agent, or the note itself;
  hunt     a command in any main turn (student or teacher) reads the grader's files: /tests, /logs/verifier,
           /setup_files, or Daytona's session logs;
  canary   the trajectory contains a benchmark canary GUID.
Selection per scenario (final_v2, single arm): at most 2 eligible rows per task (seeded), then N = min(passes, fails)
of each (seeded). Scenarios: all; drop hunt + canary; drop hunt + canary + leak.

    python hz_pool_census.py --runs <run dir> ... --out census.jsonl [--procs 12]   (one process per run)
"""
import argparse
import collections
import json
import multiprocessing as mp
import os
import random
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'pilot'))
import readout  # noqa: E402
import select_kept  # noqa: E402

ARM = 'relay_repair'
MAX_TOKENS = 65536
RAN = re.compile(r'test session starts|\b\d+ (passed|failed|errors?)\b|^[.FEsx]+\s+\[100%\]', re.M)
LEAK = re.compile(r'\b(another|the previous|previous|prior|earlier|other|original) agent\b|\bthe (verify |verification )?note\b', re.I)
HUNT = re.compile(r'(^|[\s\'"=:(])/tests\b|/logs/verifier|/setup_files|daytona', re.I)
CANARY = re.compile(r'canary GUID [0-9a-f-]{8,}', re.I)


def verifier_ran(run, name, trial):
    for jd in readout.arm_job_dirs(run, name, ARM):
        p = os.path.join(jd, trial, 'result.json')
        if os.path.exists(p):
            so = (json.load(open(p)).get('verifier_result') or {}).get('stdout') or ''
            return bool(RAN.search(so))
    return False


def census(run):
    name = os.path.basename(os.path.normpath(run))
    rv = readout.router_view(os.path.join(run, f'router_{ARM}'))
    eps = {e['sid']: e for e in rv['eps'].values()}
    trials = {t['sid']: t for t in readout.arm_trials(run, name, ARM)}
    out = []
    for r in select_kept.arm_rows(run, name, ARM):
        e, t = eps[r['sid']], trials.get(r['sid']) or {}
        leak_turns = hunt_turns = 0
        max_ctx = 0
        for m in e['main']:
            u = m.get('usage') or {}
            max_ctx = max(max_ctx, (u.get('prompt_tokens') or 0) + (u.get('completion_tokens') or 0))
            msg = (((m.get('response') or {}).get('choices') or [{}])[0].get('message') or {})
            content = msg.get('content') or ''
            if m.get('owner') == 'teacher':
                if LEAK.search(content) or LEAK.search(msg.get('reasoning_content') or msg.get('reasoning') or ''):
                    leak_turns += 1
            cmds = readout.rt.parse_reply(content, 'terminus2')['cmds']
            if any(HUNT.search(c if isinstance(c, str) else json.dumps(c)) for c in cmds):
                hunt_turns += 1
        canary = False
        if t.get('traj_path'):
            try:
                canary = bool(CANARY.search(open(t['traj_path']).read()))
            except OSError:
                pass
        r.update(run=name, weak_timeout=r['cause'] == 'timeout' and not r.get('stalled'),
                 verifier_ran=r['passed'] or verifier_ran(run, name, r['trial']), max_ctx=max_ctx,
                 leak_turns=leak_turns, teacher_turns=r['teacher_turns'], hunt_turns=hunt_turns, canary=canary)
        out.append(r)
    return out


def select(rows, seed=20260927, per_task=2):
    rng = random.Random(seed)
    by = collections.defaultdict(list)
    for r in rows:
        by[r['task']].append(r)
    kept = []
    for t in sorted(by):
        rs = by[t]
        rng.shuffle(rs)
        kept += rs[:per_task]
    p = [r for r in kept if r['passed']]
    f = [r for r in kept if not r['passed']]
    n = min(len(p), len(f))
    rng.shuffle(p)
    rng.shuffle(f)
    return dict(eligible_after_cap_pass=len(p), eligible_after_cap_fail=len(f), N=n, rows=2 * n,
                tasks=len({r['task'] for r in p[:n] + f[:n]}))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--runs', nargs='+', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--procs', type=int, default=12)
    a = ap.parse_args()
    with mp.Pool(min(a.procs, len(a.runs))) as pool:
        rows = [r for rs in pool.map(census, a.runs) for r in rs]
    with open(a.out, 'w') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')
    elig = [r for r in rows if not r['weak_timeout'] and r['verifier_ran'] and r['max_ctx'] <= MAX_TOKENS]
    P = [r for r in elig if r['passed']]
    F = [r for r in elig if not r['passed']]
    tag = lambda rs, k: sum(1 for r in rs if (r[k] if k == 'canary' else r[k] > 0))
    report = dict(
        candidates=len(rows), candidates_pass=sum(r['passed'] for r in rows),
        dropped=dict(weak_timeout=sum(r['weak_timeout'] for r in rows), verifier_never_ran=sum(not r['verifier_ran'] for r in rows),
                     over_64k=sum(r['max_ctx'] > MAX_TOKENS for r in rows)),
        eligible=dict(pass_=len(P), fail=len(F), tasks=len({r['task'] for r in elig})),
        tags_eligible={k: dict(pass_=tag(P, k), fail=tag(F, k)) for k in ('leak_turns', 'hunt_turns', 'canary')},
        leak_by_takeover=dict(collections.Counter(r['takeover'] for r in P if r['leak_turns'])),
        leak_turn_share=round(sum(r['leak_turns'] for r in P if r['leak_turns']) / max(1, sum(r['teacher_turns'] for r in P if r['leak_turns'])), 3),
        scenarios={
            'all (09-28 rule)': select(elig),
            'drop hunt + canary': select([r for r in elig if not r['hunt_turns'] and not r['canary']]),
            'drop hunt + canary + leak': select([r for r in elig if not r['hunt_turns'] and not r['canary'] and not r['leak_turns']]),
        })
    print(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
