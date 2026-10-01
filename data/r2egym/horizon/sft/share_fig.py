#!/usr/bin/env python3
"""Shareable results figure: the 09-21 base, SFT on matched Qwen-only traces (+ Kimi), the two released relay SFT models and the
Qwen3.8 teacher on TB2.1,
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
          ('mqwenk', 'SFT on Qwen-only traces\n(same tasks, + same Kimi)', '#e08a3c'),
          ('h8allkimi', 'Relay SFT from 09-21\n(…-relay-sft-allkimi-step1203)', '#3a86c8'),
          ('h9acont', 'Relay SFT, continued\n(…-relay-sft-acont-step999)', '#1d4e89'),
          ('qwen38', 'Qwen3.8-27B (teacher)', '#9a7cc0'))
MATCHED = (('base0921', 'Grug 09-21 (base)', '#b8b6b0'),
           ('mqwen', 'Qwen-only traces', '#f2c08f'),
           ('mrel', 'Relay traces', '#9cc3e6'),
           ('mqwenk', 'Qwen-only traces + Kimi', '#e08a3c'),
           ('mrelk', 'Relay traces + Kimi', '#2f6fb3'))
GRPO = ('rlh9s30', 'Relay SFT, continued + GRPO\n(30 steps on clean R2E-Gym)', '#0f2c52')
SETS = (('tb21', 'Terminal-Bench 2.1'), ('swe', 'SWE-bench Verified\n(random 100)'), ('tblite', 'OpenThoughts-TBLite'))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--extra', help='another eval_readout.py --out JSON to merge (e.g. the GRPO checkpoint)')
    ap.add_argument('--grpo', action='store_true', help='add the H9 + GRPO checkpoint after H9 (its readout via --extra)')
    ap.add_argument('--matched', action='store_true',
                    help='the matched comparison: Qwen-only vs relay traces on the same 3,872 task slots, each +/- Kimi')
    a = ap.parse_args()
    models = MATCHED if a.matched else MODELS
    if a.grpo and not a.matched:
        k = [m[0] for m in models].index('h9acont') + 1
        models = tuple(m for m in models[:k] + (GRPO,) + models[k:] if m[0] != 'h8allkimi')   # GRPO started from H9
    bold = ('mrel', 'mrelk') if a.matched else ('h8allkimi', 'h9acont', 'rlh9s30')
    r = json.load(open(a.data))['readout']
    if a.extra:
        r.update(json.load(open(a.extra)))
    plt.rcParams.update({'font.size': 15, 'font.family': 'DejaVu Sans'})
    fig, ax = plt.subplots(figsize=(15, 7.6))
    w = min(0.16, 0.74 / len(models))
    for i, (tag, name, col) in enumerate(models):
        xs, ps, ses = [], [], []
        for g, (s, _) in enumerate(SETS):
            v = r[f'{tag}/{s}']
            xs.append(g + (i - (len(models) - 1) / 2) * (w + 0.012))
            ps.append(100 * v['pass_at_1'])
            ses.append(100 * v['se'])
        ax.bar(xs, ps, w, color=col, label=name, zorder=2)
        ax.errorbar(xs, ps, yerr=ses, fmt='none', ecolor='#444', elinewidth=1.2, capsize=4, zorder=3)
        for x, p, se in zip(xs, ps, ses):
            ax.text(x, p + se + 1.2, f'{p:.1f}', ha='center', va='bottom', fontsize=12 if len(models) < 6 else 11,
                    fontweight='bold' if tag in bold else 'normal')
    ax.set_xticks(range(len(SETS)))
    ax.set_xticklabels([t for _, t in SETS], fontsize=15)
    ax.set_ylabel('pass@1 (%), mean of 3 runs')
    ax.set_ylim(0, 60 if a.matched else 85)
    ax.grid(axis='y', color='#e5e5e5', zorder=0)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.16), frameon=False, fontsize=12.5, ncol=5 if a.matched else 3)
    if a.matched:
        ax.set_title('Relay vs Qwen-only traces, matched: SFT of Grug 67B-A2B 09-21', fontsize=16, pad=14)
        fig.text(0.01, 0.01, 'Both trace sets cover the same 3,872 CalibForge task slots (2,204 tasks, 3,006 passes + 866 failures); only who played '
                 'the early turns differs.\nTrained tokens: Qwen-only 37.2M, relay 32.8M (+ 21.6M Kimi in both). Same recipe: 3 epochs, LR 3e-4. '
                 'Error bars: ±1 standard error over tasks.', ha='left', fontsize=10.5, color='#555')
    else:
        ax.set_title('Relay SFT on Grug 67B-A2B 09-21 (Qwen3.8-27B teacher + Kimi SWE-smith traces)', fontsize=16, pad=14)
        note = 'Error bars: ±1 standard error over tasks. Models: huggingface.co/laion'
        if a.grpo:
            note = ('GRPO vs its start (paired per task, 95 % range): TB2.1 +4.2 [-0.2, +8.6], SWE -3.5 [-8.2, +1.0], TBLite +5.4 [-0.3, +11.6].\n'
                    + note)
        fig.text(0.99, 0.01, note, ha='right', fontsize=10.5, color='#666')
    fig.tight_layout(rect=(0, 0.06 if a.matched else 0.02, 1, 1))
    fig.savefig(a.out, dpi=160, facecolor='white')


if __name__ == '__main__':
    main()
