#!/usr/bin/env python3
"""Shareable results figure: the 09-21 base, the two released relay SFT models and the Qwen3.8 teacher on TB2.1,
SWE-bench Verified (random 100) and OpenThoughts-TBLite, pass@1 over 3 runs with ±1 standard error. Input:
gist_table.py --save JSON.

    python share_fig.py --data table.json --out relay_sft_results.png
"""
import argparse
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

MODELS = (('base0921', 'Grug 09-21 (base)', '#b8b6b0'),
          ('h8allkimi', 'Relay SFT from 09-21\n(…-relay-sft-allkimi-step1203)', '#3a86c8'),
          ('h9acont', 'Relay SFT, continued\n(…-relay-sft-acont-step999)', '#1d4e89'),
          ('qwen38', 'Qwen3.8-27B (teacher)', '#9a7cc0'))
SETS = (('tb21', 'Terminal-Bench 2.1'), ('swe', 'SWE-bench Verified\n(random 100)'), ('tblite', 'OpenThoughts-TBLite'))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    r = json.load(open(a.data))['readout']
    plt.rcParams.update({'font.size': 15, 'font.family': 'DejaVu Sans'})
    fig, ax = plt.subplots(figsize=(13, 7.2))
    w = 0.2
    for i, (tag, name, col) in enumerate(MODELS):
        xs, ps, ses = [], [], []
        for g, (s, _) in enumerate(SETS):
            v = r[f'{tag}/{s}']
            xs.append(g + (i - 1.5) * (w + 0.015))
            ps.append(100 * v['pass_at_1'])
            ses.append(100 * v['se'])
        ax.bar(xs, ps, w, color=col, label=name, zorder=2)
        ax.errorbar(xs, ps, yerr=ses, fmt='none', ecolor='#444', elinewidth=1.2, capsize=4, zorder=3)
        for x, p, se in zip(xs, ps, ses):
            ax.text(x, p + se + 1.2, f'{p:.1f}', ha='center', va='bottom', fontsize=13,
                    fontweight='bold' if tag in ('h8allkimi', 'h9acont') else 'normal')
    ax.set_xticks(range(len(SETS)))
    ax.set_xticklabels([t for _, t in SETS], fontsize=15)
    ax.set_ylabel('pass@1 (%), mean of 3 runs')
    ax.set_ylim(0, 85)
    ax.grid(axis='y', color='#e5e5e5', zorder=0)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.16), frameon=False, fontsize=12.5, ncol=4)
    ax.set_title('Relay SFT on Grug 67B-A2B 09-21 (Qwen3.8-27B teacher + Kimi SWE-smith traces)', fontsize=16, pad=14)
    fig.text(0.99, 0.01, 'Error bars: ±1 standard error over tasks. Models: huggingface.co/laion', ha='right', fontsize=10.5,
             color='#666')
    fig.tight_layout()
    fig.savefig(a.out, dpi=160, facecolor='white')


if __name__ == '__main__':
    main()
