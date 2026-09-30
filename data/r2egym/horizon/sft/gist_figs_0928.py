#!/usr/bin/env python3
"""Redraw the relay SFT gist's two 09-28 figures (Jupiter runs) without arm C, from the counts in the gist text:
fig1_results.png (solved / tasks per model and benchmark, one try per task, Wilson 95 % intervals) and fig2_why.png
(harness-rejected turns, tokens written per TB2.1 task, TB2.1 tasks that hit the 30-min clock).

    python gist_figs_0928.py --out-dir <dir>
"""
import argparse
import math
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

BG = '#fcfcfa'
COL = {'09-21': '#9e9d98', 'A': '#2b7bd6', 'B': '#eb6a3a', 'MIX': '#f0a202'}
LABEL = {'09-21': '09-21 (before SFT)', 'A': 'A  relay, 09-21 turns masked', 'B': 'B  Qwen alone',
         'MIX': 'MIX  relay + Qwen-alone rows'}
# solved, tasks (gist table, 2026-09-28/29, Jupiter)
RES = {
    'Held-out CalibForge (300)': {'09-21': (18, 300), 'A': (29, 300), 'B': (26, 300)},
    'TB2.1 (88)': {'09-21': (5, 88), 'A': (9, 88), 'B': (4, 88), 'MIX': (9, 88)},
    'SWE-bench Verified (100)': {'09-21': (9, 100), 'A': (30, 100), 'B': (17, 100), 'MIX': (32, 100)},
    'OpenThoughts-TBLite (100)': {'09-21': (15, 100), 'A': (17, 100), 'B': (12, 100), 'MIX': (18, 100)},
}
WHY = {'Replies the harness rejects (% of turns)': ({'09-21': 89, 'A': 15, 'B': 20}, '{:d}%', 115),
       'Tokens written per task (thousands, median)': ({'09-21': 56, 'A': 130, 'B': 120}, '{:d}k', 190),
       'Hit the 30-min clock (of 88 tasks)': ({'09-21': 29, 'A': 60, 'B': 62}, '{:d}', 88)}


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def style(ax):
    ax.set_facecolor(BG)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color('#bbb')


def fig1(out):
    models = ['09-21', 'A', 'B', 'MIX']
    fig, ax = plt.subplots(figsize=(17.6, 6.7), facecolor=BG)
    style(ax)
    w, xt = 0.18, []
    for g, (bench, rs) in enumerate(RES.items()):
        xt.append(g)
        for i, m in enumerate(models):
            if m not in rs:
                continue
            k, n = rs[m]
            x = g + (i - 1.5) * (w + 0.02)
            p = 100 * k / n
            lo, hi = wilson(k, n)
            ax.bar(x, p, w, color=COL[m], label=LABEL[m] if g == 2 or (g == 0 and m != 'MIX') else None)
            ax.vlines(x, 100 * lo, 100 * hi, color='k', lw=1.5)
            ax.text(x, 100 * hi + 0.6, f'{p:.0f}%\n{k}/{n}', ha='center', va='bottom', fontsize=11)
    h, l = ax.get_legend_handles_labels()
    seen = dict(zip(l, h))
    ax.legend([seen[LABEL[m]] for m in models], [LABEL[m] for m in models], loc='upper left', frameon=False, fontsize=13)
    ax.set_xticks(xt)
    ax.set_xticklabels(list(RES), fontsize=14)
    ax.set_ylim(0, 52)
    ax.set_ylabel('tasks solved (%), one try per task', fontsize=13)
    ax.grid(axis='y', color='#e6e6e6')
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out, dpi=100, facecolor=BG)


def fig2(out):
    models = ['09-21', 'A', 'B']
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), facecolor=BG)
    for ax, (title, (vals, fmt, xmax)) in zip(axes, WHY.items()):
        style(ax)
        ys = list(range(len(models)))[::-1]
        for y, m in zip(ys, models):
            ax.barh(y, vals[m], 0.62, color=COL[m])
            ax.text(vals[m] + xmax * 0.012, y, fmt.format(vals[m]), va='center', fontsize=13)
        ax.set_yticks(ys)
        ax.set_yticklabels(models, fontsize=14)
        ax.set_xlim(0, xmax)
        ax.set_title(title, fontsize=13, loc='left')
        ax.grid(axis='x', color='#e6e6e6')
        ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out, dpi=100, facecolor=BG)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out-dir', required=True)
    a = ap.parse_args()
    fig1(os.path.join(a.out_dir, 'fig1_results.png'))
    fig2(os.path.join(a.out_dir, 'fig2_why.png'))


if __name__ == '__main__':
    main()
