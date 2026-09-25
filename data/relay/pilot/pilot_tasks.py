#!/usr/bin/env python3
"""Pick the relay pilot's CalibForge tasks, and write the router's task file.

    # Mac: pick 100 from the Daytona-covered pool (bundle tree) with the pool check's parquet
    python pilot_tasks.py pick --tree <task_tree> --parquet <calibforge_tasks.parquet> --n 100 \
        --exclude ../../mini_swe_host/smoke_tasks.txt --out relay100_tasks.txt
    # anywhere with the tree: the router's {task_id, instruction, agent_timeout_sec} list
    python pilot_tasks.py router-json --tree <tree> --list relay100_tasks.txt --out relay100_router_tasks.json

Pool: the 2,457 CalibForge tasks that run on Daytona's three snapshots (coverage.tsv status `covered`; all are in the
pool check's recommended 2,500, so the 86 TB2-templated and the 393 agent-internet tasks are already out). Removed
before sampling: the 8 mini-swe-agent smoke tasks, tasks whose agent budget is above 1,800 s (130 of 2,457; they would
stretch the pilot's wall to 2 x 3,600 s for 5 % of the pool), and any task whose instruction text duplicates another's
(the router matches an episode to its task by instruction).

Strata: subset (contrastive_solver / multi_solver) x tb2_gap (non-Python or compiles), allocated in proportion to the
pool by largest remainder, and inside a stratum over categories the same way (a seeded random draw per cell), so the
100 keep the pool's category mix. The list is written longest agent budget first
(harbor starts trials in config order; the longest ones should not start last).
"""
import argparse
import collections
import hashlib
import json
import os
import random
import re


def instruction_of(tree, tid):
    with open(os.path.join(tree, tid, 'instruction.md')) as f:
        return f.read()


def budget_of(tree, tid):
    s = open(os.path.join(tree, tid, 'task.toml')).read()
    m = re.search(r'\[agent\][^\[]*?timeout_sec\s*=\s*([0-9.]+)', s)
    return float(m.group(1)) if m else 0.0


def _stratified(d, n, rng):
    """n task ids: largest-remainder quotas over strata, then over categories inside each stratum; seeded draws."""
    sizes = d.stratum.value_counts().sort_index()
    quota = {s: int(n * c / len(d)) for s, c in sizes.items()}
    rema = sorted(sizes.index, key=lambda s: -(n * sizes[s] / len(d) - quota[s]))
    for s in rema[:n - sum(quota.values())]:
        quota[s] += 1
    picked = []
    for s in sorted(quota):
        g = d[d.stratum == s]
        cats = g.category.value_counts().sort_index()
        cq = {c: int(quota[s] * k / len(g)) for c, k in cats.items()}
        order = sorted(cats.index, key=lambda c: (-(quota[s] * cats[c] / len(g) - cq[c]), c))
        for c in order[:quota[s] - sum(cq.values())]:
            cq[c] += 1
        for c in sorted(cq):
            picked += rng.sample(sorted(g[g.category == c].task_id), cq[c])
    return picked


def pick(a):
    import pandas as pd
    d = pd.read_parquet(a.parquet)
    cov = pd.read_csv(os.path.join(a.tree, 'coverage.tsv'), sep='\t')
    covered = set(cov[cov.status == 'covered'].task_id)
    exclude = {l.strip() for l in open(a.exclude)} if a.exclude else set()
    d = d[d.task_id.isin(covered) & d.recommended_2500 & ~d.task_id.isin(exclude)].copy()
    d = d[[os.path.isdir(os.path.join(a.tree, t)) for t in d.task_id]]
    d['budget'] = [budget_of(a.tree, t) for t in d.task_id]
    n_pool = len(d)
    d = d[d.budget <= a.max_budget]
    d['ihash'] = [hashlib.sha1(instruction_of(a.tree, t).strip().encode()).hexdigest() for t in d.task_id]
    dup = d.ihash.duplicated(keep=False)
    d = d[~dup]
    d['stratum'] = d.subset.str.replace('_solver', '') + '/' + d.tb2_gap.map({True: 'gap', False: 'nogap'})
    picked = _stratified(d, a.n, random.Random(a.seed))
    sizes = d.stratum.value_counts().sort_index()
    sel = d[d.task_id.isin(picked)].sort_values(['budget', 'task_id'], ascending=[False, True])
    with open(a.out, 'w') as f:
        for t in sel.task_id:
            f.write(t + '\n')
    strata = a.out.rsplit('.', 1)[0] + '_strata.tsv'
    sel[['task_id', 'stratum', 'category', 'difficulty', 'budget', 'verifier_timeout_s']].merge(
        cov[['task_id', 'daytona_base']], on='task_id').to_csv(strata, sep='\t', index=False)
    print(f'pool {n_pool} (covered, recommended, not smoke) -> {len(d)} after budget <= {a.max_budget:.0f} s and '
          f'unique instructions; picked {len(sel)} -> {a.out}')
    print('per stratum     :', sel.stratum.value_counts().sort_index().to_dict())
    print('pool share      :', {s: round(c / len(d), 3) for s, c in sizes.items()})
    print('categories      :', sel.category.value_counts().to_dict())
    print('budgets         :', sel.budget.value_counts().sort_index().to_dict())
    print('bases           :', sel.merge(cov[['task_id', 'daytona_base']], on='task_id').daytona_base.value_counts().to_dict())


def heldout(a):
    """A held-out eval split: n tasks from the same Daytona pool, disjoint from every --exclude list (the pilot's
    tasks, the smoke tasks) and from any task whose instruction text matches an excluded one. Same stratification
    as `pick` (subset x tb2_gap, then category, largest remainder), no budget cap: an eval split keeps the pool's
    budget mix. Never generate relay or SFT data from these tasks."""
    import pandas as pd
    d = pd.read_parquet(a.parquet)
    cov = pd.read_csv(os.path.join(a.tree, 'coverage.tsv'), sep='\t')
    covered = set(cov[cov.status == 'covered'].task_id)
    exclude = set()
    for path in a.exclude:
        exclude |= {l.strip() for l in open(path) if l.strip()}
    d = d[d.task_id.isin(covered) & d.recommended_2500].copy()
    d = d[[os.path.isdir(os.path.join(a.tree, t)) for t in d.task_id]]
    d['ihash'] = [hashlib.sha1(instruction_of(a.tree, t).strip().encode()).hexdigest() for t in d.task_id]
    banned = set(d[d.task_id.isin(exclude)].ihash)
    d = d[~d.task_id.isin(exclude) & ~d.ihash.isin(banned)]
    d = d[~d.ihash.duplicated(keep='first')]
    d['budget'] = [budget_of(a.tree, t) for t in d.task_id]
    d['stratum'] = d.subset.str.replace('_solver', '') + '/' + d.tb2_gap.map({True: 'gap', False: 'nogap'})
    sel = _stratified(d, a.n, random.Random(a.seed))
    sel = d[d.task_id.isin(sel)].sort_values('task_id')
    with open(a.out, 'w') as f:
        for t in sel.task_id:
            f.write(t + '\n')
    sel[['task_id', 'stratum', 'category', 'difficulty', 'budget', 'verifier_timeout_s']].merge(
        cov[['task_id', 'daytona_base']], on='task_id').to_csv(a.out.rsplit('.', 1)[0] + '_strata.tsv', sep='\t', index=False)
    clash = exclude & set(sel.task_id)
    assert not clash, clash
    print(f'held-out {len(sel)} -> {a.out}; disjoint from {len(exclude)} excluded tasks (and their instruction texts)')
    print('strata    :', sel.stratum.value_counts().to_dict())
    print('budgets   :', sel.budget.value_counts().sort_index().to_dict())


def full_pool(a):
    """The full run's pool: every Daytona-covered task, minus the held-out split, minus tasks whose agent budget is
    above --max-budget, minus exact duplicate instructions (one copy kept; the router matches episodes to tasks by
    instruction). Pilot tasks may be in it. Written longest budget first."""
    cov = [l.split('\t') for l in open(os.path.join(a.tree, 'coverage.tsv')).read().splitlines()[1:]]
    covered = sorted(r[0] for r in cov if r[-1] == 'covered')
    held = {l.strip() for l in open(a.heldout) if l.strip()}
    rows, seen = [], set()
    n_held = n_budget = n_dup = 0
    for t in covered:
        if t in held:
            n_held += 1
            continue
        b = budget_of(a.tree, t)
        if b > a.max_budget:
            n_budget += 1
            continue
        h = hashlib.sha1(instruction_of(a.tree, t).strip().encode()).hexdigest()
        if h in seen:
            n_dup += 1
            continue
        seen.add(h)
        rows.append((t, b))
    rows.sort(key=lambda r: (-r[1], r[0]))
    with open(a.out, 'w') as f:
        for t, _ in rows:
            f.write(t + '\n')
    print(f'covered {len(covered)} - held-out {n_held} - budget > {a.max_budget:.0f} s {n_budget} - duplicate '
          f'instructions {n_dup} = {len(rows)} -> {a.out}')
    print('budgets:', dict(collections.Counter(b for _, b in rows)))


def router_json(a):
    ids = [l.strip() for l in open(a.list) if l.strip()]
    out = []
    for t in ids:
        out.append(dict(task_id=t, instruction=instruction_of(a.tree, t), agent_timeout_sec=budget_of(a.tree, t)))
    keys = [o['instruction'].strip()[:400] for o in out]
    clash = [(i, j) for i in range(len(keys)) for j in range(len(keys)) if i != j and keys[i] in out[j]['instruction']]
    if clash:
        raise SystemExit(f'instruction prefixes that match another task: {clash[:5]}')
    json.dump(out, open(a.out, 'w'), indent=1)
    print(f'{len(out)} tasks -> {a.out}')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = p.add_subparsers(dest='cmd', required=True)
    q = sp.add_parser('pick')
    q.add_argument('--tree', required=True)
    q.add_argument('--parquet', required=True)
    q.add_argument('--n', type=int, default=100)
    q.add_argument('--exclude')
    q.add_argument('--max-budget', type=float, default=1800.0)
    q.add_argument('--seed', type=int, default=20260925)
    q.add_argument('--out', required=True)
    h = sp.add_parser('heldout')
    h.add_argument('--tree', required=True)
    h.add_argument('--parquet', required=True)
    h.add_argument('--n', type=int, default=300)
    h.add_argument('--exclude', action='append', default=[], help='task-id list to keep out (repeatable)')
    h.add_argument('--seed', type=int, default=926)
    h.add_argument('--out', required=True)
    f = sp.add_parser('full-pool')
    f.add_argument('--tree', required=True)
    f.add_argument('--heldout', required=True)
    f.add_argument('--max-budget', type=float, default=1800.0)
    f.add_argument('--out', required=True)
    r = sp.add_parser('router-json')
    r.add_argument('--tree', required=True)
    r.add_argument('--list', required=True)
    r.add_argument('--out', required=True)
    a = p.parse_args()
    {'pick': pick, 'heldout': heldout, 'full-pool': full_pool, 'router-json': router_json}[a.cmd](a)


if __name__ == '__main__':
    main()
