#!/usr/bin/env python3
"""Figure for the relay SFT gist: pass@1 of every Horizon SFT arm with a complete eval record, one panel per benchmark
(TB2.1, SWE-bench Verified random-100, TB-lite), bars with their 95 % task-bootstrap range, the control highlighted and
its level drawn across each panel. Input: gist_table.py --save JSON.

    python gist_fig.py --data table.json --out fig_horizon.png
"""
import argparse
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

SETS = (('tb21', 'TB2.1 (88 tasks)'), ('swe', 'SWE-bench Verified random-100'), ('tblite', 'OpenThoughts-TBLite (100)'))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    d = json.load(open(a.data))
    r, arms, ctl = d['readout'], d['arms'], d['control']
    names = [x[1].split(',')[0] for x in arms]
    fig, axes = plt.subplots(1, 3, figsize=(4.2 * 3, 0.42 * len(arms) + 1.8), sharey=True)
    for ax, (s, title) in zip(axes, SETS):
        ys = list(range(len(arms)))[::-1]
        for y, (tag, label, *_), in zip(ys, arms):
            v = r.get(f'{tag}/{s}') or {}
            p, ci = v.get('pass_at_1'), v.get('ci95')
            if p is None:
                continue
            col = '#1f4e79' if tag == ctl else '#8fb3d9'
            ax.barh(y, 100 * p, color=col, height=0.7)
            ax.errorbar(100 * p, y, xerr=[[100 * (p - ci[0])], [100 * (ci[1] - p)]], fmt='none', ecolor='#333', capsize=3, lw=1)
            ax.text(100 * ci[1] + 1, y, f'{100 * p:.1f}', va='center', fontsize=9)
        c = (r.get(f'{ctl}/{s}') or {}).get('pass_at_1')
        if c is not None:
            ax.axvline(100 * c, color='#1f4e79', ls='--', lw=1)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('pass@1, mean of 3 runs (%)')
        ax.set_xlim(0, max(60, ax.get_xlim()[1]))
        ax.grid(axis='x', alpha=0.3)
    axes[0].set_yticks(list(range(len(arms)))[::-1])
    axes[0].set_yticklabels([x[1] for x in arms], fontsize=9)
    fig.suptitle('Horizon SFT arms on 09-21 (dark = control A; dashed line = A; bars show 95 % ranges)', fontsize=11)
    fig.tight_layout()
    fig.savefig(a.out, dpi=150)


if __name__ == '__main__':
    main()
