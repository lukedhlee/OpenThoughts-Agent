#!/usr/bin/env python3
"""Pick the episodes to score: a seeded, domain-stratified sample of teacher-only CalibForge MSA2 episodes.

Pool = the vanilla arm's selected rows of the clean 09-21 MSA2 pair (msa0921_clean_pair/matched_sft/selected/
msaqwen_rows.jsonl, msa_select.py --outcome pass --ends-submit --per-task 2 --seed 0): Qwen3.8-27B alone, mini-swe-agent
2.4.6 tool mode, the 2026-10-03 msa_full6 control runs; verifier-scored passes that end on the submit, no leak / hunt /
canary / weak timeout, no held-out-300 task, <= 2 episodes per task. These are the episodes the vanilla SFT arm trains on.

Sampling: tasks are grouped by CalibForge domain (the category in the task id, = task.toml's category); the --n slots
are split over domains in proportion to their task counts (largest remainder); within a domain tasks are shuffled with
--seed and taken in order, one episode per task (seeded choice among its <= 2). An episode is skipped, and recorded
under exclusions, if its trajectory file is missing or ambiguous or its full-history render does not fit --max-tokens.

    python sample.py --out sample.json [--n 20] [--seed 20261014] [--tokenizer-dir <student dir>]
"""
import argparse
import collections
import glob
import hashlib
import json
import os
import random
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'pilot'))
import readout  # noqa: E402
from render_full import load_student, render  # noqa: E402

X = '/scratch/11584/lukedhlee/experiments'
POOL = f'{X}/msa0921_clean_pair/matched_sft/selected/msaqwen_rows.jsonl'
RUNS = f'{X}/relay_msa/runs'
TASK_RE = re.compile(r'calibforge__([a-z_]+)__([a-z0-9-]+?)_\d{8}_\d{6}_\d{3}$')
KEEP = ('task', 'sid', 'run', 'trial', 'arm', 'reward', 'passed', 'teacher_turns', 'n_tokens', 'verifier_ran')


def load_pool(path):
    rows = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            rows.append({k: r.get(k) for k in KEEP})
    return rows


def allocate(counts, n):
    """Largest-remainder split of n slots over domains in proportion to counts (ties broken by name)."""
    total = sum(counts.values())
    quota = {d: n * c / total for d, c in counts.items()}
    out = {d: int(q) for d, q in quota.items()}
    for d in sorted(quota, key=lambda d: (-(quota[d] - out[d]), d))[:n - sum(out.values())]:
        out[d] += 1
    return out


def locate(row):
    """(MSA trajectory, harbor trajectory, trial dir) of a row: the attempt whose trajectory.json has the row's sid."""
    run_dir = os.path.join(RUNS, row['run'])
    hits = []
    for jd in readout.arm_job_dirs(run_dir, row['run'], row['arm']):
        for tj in glob.glob(os.path.join(jd, row['trial'], '**', 'agent', 'trajectory.json'), recursive=True):
            if json.load(open(tj)).get('session_id') == row['sid']:
                hits.append(tj)
    if len(hits) != 1:
        return None, f'{len(hits)} trajectories with this sid'
    tj = hits[0]
    msa = os.path.join(os.path.dirname(tj), readout.MSA_TRAJECTORY)
    trial_dir = tj.split('/attempts/')[0] if '/attempts/' in tj else os.path.dirname(os.path.dirname(tj))
    return dict(msa_trajectory=msa, harbor_trajectory=tj, trial_dir=trial_dir), None


def task_meta(trial_dir):
    cfg = json.load(open(os.path.join(trial_dir, 'config.json')))
    tdir = cfg['task']['path']
    meta = dict(task_dir=tdir)
    try:
        import tomllib
        t = tomllib.load(open(os.path.join(tdir, 'task.toml'), 'rb')).get('metadata', {})
        meta.update({k: t.get(k) for k in ('difficulty', 'category', 'tags', 'expert_time_estimate_min',
                                           'junior_time_estimate_min')})
    except (OSError, ValueError):
        pass
    try:
        meta['instruction'] = open(os.path.join(tdir, 'instruction.md')).read()
    except OSError:
        pass
    return meta


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--pool', default=POOL)
    ap.add_argument('--out', required=True)
    ap.add_argument('--n', type=int, default=20)
    ap.add_argument('--seed', type=int, default=20261014)
    ap.add_argument('--tokenizer-dir', default='/scratch/11584/lukedhlee/models/grug-datakit-sft-20260921')
    ap.add_argument('--max-tokens', type=int, default=131072)
    a = ap.parse_args()
    rows = load_pool(a.pool)
    by_task = collections.defaultdict(list)
    for r in rows:
        by_task[r['task']].append(r)
    domain = {}
    for t in by_task:
        m = TASK_RE.match(t)
        domain[t] = m.group(2) if m else 'unknown'
    counts = collections.Counter(domain.values())
    slots = allocate(counts, a.n)
    rng = random.Random(a.seed)
    tok, tpl, bos = load_student(a.tokenizer_dir)
    episodes, exclusions = [], []
    for d in sorted(slots):
        tasks = sorted(t for t in by_task if domain[t] == d)
        rng.shuffle(tasks)
        taken = 0
        for t in tasks:
            if taken == slots[d]:
                break
            row = rng.choice(sorted(by_task[t], key=lambda r: r['sid']))
            where, why = locate(row)
            if where is None:
                exclusions.append(dict(task=t, sid=row['sid'], reason=why))
                continue
            msgs = json.load(open(where['msa_trajectory']))['messages']
            n_tok = len(render(msgs, tok, tpl, bos)['ids'])
            if n_tok + 1 > a.max_tokens:
                exclusions.append(dict(task=t, sid=row['sid'], reason=f'render {n_tok} tokens > {a.max_tokens}'))
                continue
            m = TASK_RE.match(t)
            episodes.append(dict(row, **where, domain=d, generator=m.group(1) if m else None,
                                 render_tokens=n_tok, sft_render_tokens=row['n_tokens'],
                                 msa_trajectory_sha256=hashlib.sha256(open(where['msa_trajectory'], 'rb').read()).hexdigest(),
                                 task_meta=task_meta(where['trial_dir'])))
            taken += 1
        if taken < slots[d]:
            exclusions.append(dict(domain=d, reason=f'only {taken} of {slots[d]} slots filled'))
    episodes.sort(key=lambda e: (e['domain'], e['task']))
    for k, e in enumerate(episodes):
        e['traj_id'] = f't{k + 1:02d}'
        e.pop('n_tokens', None)
    man = dict(pool=a.pool, pool_rows=len(rows), pool_tasks=len(by_task), domain_task_counts=dict(counts),
               slots=slots, seed=a.seed, n=a.n, max_tokens=a.max_tokens, tokenizer_dir=a.tokenizer_dir,
               teacher='Qwen/Qwen3.8-27B @ 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0',
               harness='harbor mini-swe-agent-host, mini-swe-agent 2.4.6 tool mode, interleaved_thinking true',
               method=__doc__.split('Sampling: ')[1].split('\n\n')[0].replace('\n', ' '),
               exclusions=exclusions, episodes=episodes)
    json.dump(man, open(a.out, 'w'), indent=1)
    print(json.dumps(dict(episodes=len(episodes), exclusions=exclusions, slots=slots,
                          tokens=[e['render_tokens'] for e in episodes]), indent=1))


if __name__ == '__main__':
    main()
