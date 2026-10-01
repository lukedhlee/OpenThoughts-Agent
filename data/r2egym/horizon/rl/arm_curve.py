#!/usr/bin/env python3
"""arm_curve.py <run name> [...] — per-step table of a Horizon RL arm from its job log's WANDB_MIRROR lines (all jobs
in the run dir's jobs.txt, later jobs win on a repeated step). Columns are the in-run health keys: tis log-ratio, entropy,
raw reward, q = reward given done, mixed-group fraction, mean tokens per trajectory, masked trajectories, step seconds.
Stop rules (STATUS): q < .60 at two consecutive steps with a flat finish rate = collapse; tis > 2x gate 5's 0.0368."""
import json
import re
import sys
from pathlib import Path

E = Path('/scratch/11584/lukedhlee/experiments/rl')
COLS = [('tis', 'policy/tis/log_ratio_abs_mean', '{:.4f}'), ('entropy', 'policy/policy_entropy', '{:.3f}'),
        ('reward', 'reward/avg_raw_reward', '{:.3f}'), ('q', 'diag/reward_given_done', '{:.3f}'),
        ('mixed', 'diag/frac_groups_mixed', '{:.2f}'), ('tokens', 'generate/avg_num_tokens', '{:.0f}'),
        ('masked', 'generate/num_masked_trajectories', '{:.0f}'), ('step_s', 'timing/step', '{:.0f}')]


def rows(run):
    steps = {}
    for job in (E / run / 'jobs.txt').read_text().split():
        log = E / run / 'logs' / f'{run}_{job}.out'
        if not log.exists():
            continue
        for line in log.open(errors='replace'):
            m = re.search(r'WANDB_MIRROR kind=train step=(\d+) metrics=(.*)', line)
            if m:
                try:
                    steps[int(m.group(1))] = json.JSONDecoder().raw_decode(m.group(2))[0]
                except ValueError:
                    pass
    return steps


for run in sys.argv[1:]:
    st = rows(run)
    print(f'{run}: {len(st)} steps')
    print('| step | ' + ' | '.join(c[0] for c in COLS) + ' |')
    print('|---' * (len(COLS) + 1) + '|')
    for s in sorted(st):
        d = st[s]
        print(f'| {s} | ' + ' | '.join(f.format(d[k]) if k in d else '-' for _, k, f in COLS) + ' |')
