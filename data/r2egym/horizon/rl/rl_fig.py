#!/usr/bin/env python3
"""rl_fig.py — standalone figure of one RL arm's effect: pass@1 of the base, the RL start checkpoint and the RL checkpoint
on TB2.1, SWE-bench Verified random-100, TB-lite (eval_readout.py --out JSON, mean of 3 runs) and the clean held-out
R2E-Gym slice (screen_report.py per_task.tsv of the held-out probes, 8 attempts per task). Bars carry their 95 % task
bootstrap range; above the RL bar, the paired change vs the start with its 95 % range.

    python rl_fig.py --readout r.json --tags base0921,h9acont,rlh9s30 --labels "09-21 SFT (base),H9,H9 + GRPO 30" \
        --heldout h9acont=<dir with per_task.tsv> rlh9s30=<dir> --out fig.png [--title ...]
"""
import argparse
import csv
import json
import random

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

SETS = (('tb21', 'TB2.1'), ('swe', 'SWE-bench Verified\nrandom-100'), ('tblite', 'OpenThoughts-TBLite'),
        ('heldout', 'Held-out R2E-Gym\n(123 clean tasks, 8 tries)'))
COLORS = ('#9e9d98', '#8fb3d9', '#1f4e79')


def boot(xs, n=20000, seed=0):
    rng = random.Random(seed)
    m = sorted(sum(xs[rng.randrange(len(xs))] for _ in xs) / len(xs) for _ in range(n))
    return m[int(0.025 * n)], m[int(0.975 * n)]


def heldout(path):
    d = {}
    for r in csv.DictReader(open(f'{path}/per_task.tsv'), delimiter='\t'):
        a = int(r['attempts']) - int(r['invalid'] or 0)
        if a > 0:
            d[r['task']] = int(r['solved']) / a
    return d


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--readout', required=True)
    ap.add_argument('--tags', required=True, help='base,start,rl')
    ap.add_argument('--labels', required=True)
    ap.add_argument('--heldout', nargs='*', default=[], help='tag=<dir with per_task.tsv>')
    ap.add_argument('--title')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    r = json.load(open(a.readout))
    tags, labels = a.tags.split(','), a.labels.split(',')
    start, rl = tags[1], tags[2]
    ho = {t: heldout(p) for t, p in (x.split('=', 1) for x in a.heldout)}
    if start in ho and rl in ho:   # held-out entries in the readout's shape: per-model mean + range, paired diff
        common = sorted(set(ho[start]) & set(ho[rl]))
        for t in [x for x in tags if x in ho]:
            xs = [ho[t][k] for k in common if k in ho[t]]
            r[f'{t}/heldout'] = {'pass_at_1': sum(xs) / len(xs), 'ci95': list(boot(xs)), 'tasks': len(xs)}
        dd = [ho[start][k] - ho[rl][k] for k in common]
        r[f'{start}-{rl}/heldout'] = {'diff': sum(dd) / len(dd), 'ci95': list(boot(dd)), 'tasks': len(dd)}

    fig, axes = plt.subplots(1, 4, figsize=(13.5, 4.3))
    for ax, (s, title) in zip(axes, SETS):
        top = 0
        for i, (t, lab, col) in enumerate(zip(tags, labels, COLORS)):
            v = r.get(f'{t}/{s}')
            if not v:
                ax.text(i, 3, 'not run', ha='center', va='bottom', fontsize=9, color='#777')
                continue
            p, ci = 100 * v['pass_at_1'], [100 * c for c in v['ci95']]
            ax.bar(i, p, color=col, width=0.72)
            ax.errorbar(i, p, yerr=[[p - ci[0]], [ci[1] - p]], fmt='none', ecolor='#333', capsize=3, lw=1)
            ax.text(i, ci[1] + 1.2, f'{p:.1f}', ha='center', va='bottom', fontsize=9)
            top = max(top, ci[1])
        d = r.get(f'{start}-{rl}/{s}')
        if d:   # readout diff is start minus rl; show rl minus start
            dv, lo, hi = -100 * d['diff'], -100 * d['ci95'][1], -100 * d['ci95'][0]
            sig = lo > 0 or hi < 0
            ax.text(1.5, top + 7, f'GRPO {dv:+.1f}\n[{lo:+.1f}, {hi:+.1f}]', ha='center', va='bottom', fontsize=9,
                    fontweight='bold' if sig else 'normal', color='#1f4e79' if dv > 0 else '#a33')
        ax.set_title(title, fontsize=10)
        ax.set_xlim(-0.6, len(tags) - 0.4)
        ax.set_xticks(range(len(tags)))
        ax.set_xticklabels(labels, fontsize=8, rotation=20, ha='right')
        ax.set_ylim(0, max(top + 20, 30))
        ax.grid(axis='y', alpha=0.3)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].set_ylabel('pass@1 (%)')
    fig.suptitle(a.title or '', fontsize=11, x=0.01, ha='left')
    fig.text(0.01, 0.005, 'Benchmarks: mean of 3 runs. Held-out: 8 attempts per task. Bars: 95 % bootstrap over tasks. '
             'GRPO change: paired per task vs the start checkpoint, 95 % range (bold = excludes 0).',
             fontsize=8, color='#555')
    fig.tight_layout(rect=(0, 0.03, 1, 0.94))
    fig.savefig(a.out, dpi=160)
    print('wrote', a.out)


if __name__ == '__main__':
    main()
