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

# Qwen-only baselines are orange, relay models a family of close blues. In the matched figures each ablation pair
# (Qwen-only, relay on the same slots) stands side by side; None leaves a gap between pairs.
QWEN, QWEN_PLUS = '#eda25e', '#d17a2e'
RELAY = ('#5b9ad6', '#3a7cc2', '#245fa3')
MODELS = (('base0921', 'Grug 09-21 (base)', '#b8b6b0'),
          ('mqwenk', 'SFT on Qwen-only traces\n(same tasks, + same Kimi)', QWEN_PLUS),
          ('h8allkimi', 'Relay SFT from 09-21\n(…-relay-sft-allkimi-step1203)', RELAY[0]),
          ('h9acont', 'Relay SFT, continued\n(…-relay-sft-acont-step999)', RELAY[1]),
          ('qwen38', 'Qwen3.8-27B (teacher)', '#9a7cc0'))
MATCHED = (('base0921', 'Grug 09-21 (base)', '#b8b6b0'), None,
           ('mqwen', 'Qwen-only traces', QWEN),
           ('mrel', 'Relay traces', RELAY[0]), None,
           ('mqwenk', 'Qwen-only traces\n+ Kimi SWE-smith traces', QWEN_PLUS),
           ('mrelk', 'Relay traces\n+ Kimi SWE-smith traces', RELAY[1]))
TMAX = (('base0921', 'Grug 09-21 (base)', '#b8b6b0'), None,
        ('t3qtmax', 'TMax Qwen-only traces', QWEN),
        ('t3tmax', 'TMax relay traces', RELAY[0]), None,
        ('t1qtmax', 'TMax Qwen-only traces\n+ CalibForge relay + Kimi', QWEN_PLUS),
        ('t1tmax', 'TMax relay traces\n+ CalibForge relay + Kimi', RELAY[1]))
# --harness: the best Terminus-2 recipe (H8) under Terminus-2, the same recipe ported to MSA2, and the best MSA2 recipe;
# hatched bars are evaluated under MSA2, plain ones under Terminus-2, and the recipe keeps its colour across harnesses
HARNESS = (('base0921', 'Grug 09-21 (base)\nTerminus-2 eval', '#b8b6b0'),
           ('h8allkimi', 'H8, best Terminus-2 recipe\n(relay + Kimi), Terminus-2 eval', RELAY[1]), None,
           ('m22base', 'Grug 09-21 (base)\nMSA2 eval', '#b8b6b0'),
           ('m22msaallswe', 'H8 recipe ported to MSA2\n(relay + SWE traces), MSA2 eval', RELAY[1]),
           ('m22msafin', 'msafin, best MSA2 recipe\n(filtered relay + strong SWE), MSA2 eval', '#2a9d5c'))
# --status: the best model under each harness and the MSA2 steps that got there (gist header, 2026-10-10)
STATUS = (('base0921', 'Grug 09-21 (base)\nTerminus-2 eval', '#b8b6b0'),
          ('h8allkimi', 'H8, best Terminus-2 model\n(relay + Kimi), Terminus-2 eval', RELAY[1]), None,
          ('m22base', 'Grug 09-21 (base)\nMSA2 eval', '#b8b6b0'),
          ('m22msafin', 'msafin: filtered relay\n+ Orchard / SI2CA SWE traces', '#8fd1a8'),
          ('m22msafing2', 'msafing2: + teacher assists\n(guided relay)', '#4fb37a'),
          ('m22msafing2nrcf', 'msafing2nrcf: + rows rendered\nto the eval context (best MSA2)', '#1f7a47'))
HATCHED = ('m22base', 'm22msaallswe', 'm22msafin', 'm22msafing2', 'm22msafing2nrcf')
GAP = 0.5   # width of a None gap, in bar slots
GRPO = ('rlh9s30', 'Relay SFT, continued + GRPO\n(30 steps on clean R2E-Gym)', RELAY[2])
SETS = (('tb21', 'Terminal-Bench 2.1'), ('swe', 'SWE-bench Verified\n(random 100)'), ('tblite', 'OpenThoughts-TBLite'))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--extra', help='another eval_readout.py --out JSON to merge (e.g. the GRPO checkpoint)')
    ap.add_argument('--grpo', action='store_true', help='add the H9 + GRPO checkpoint after H9 (its readout via --extra)')
    ap.add_argument('--matched', action='store_true',
                    help='the matched comparison: Qwen-only vs relay traces on the same 3,872 task slots, each +/- Kimi')
    ap.add_argument('--tmax', action='store_true',
                    help='the TMax matched comparison: Qwen-only vs relay traces on 2,730 TMax task slots, alone and + the 5,916 CalibForge relay and 4,392 Kimi SWE-smith traces')
    ap.add_argument('--status', action='store_true',
                    help='best model per harness + the MSA2 steps (--data may be a flat eval_readout.py --out JSON)')
    ap.add_argument('--harness', action='store_true',
                    help='Terminus-2 vs MSA2: H8 under Terminus-2, the H8 recipe ported to MSA2, msafin (MSA2 readout via --extra)')
    a = ap.parse_args()
    models = STATUS if a.status else HARNESS if a.harness else TMAX if a.tmax else MATCHED if a.matched else MODELS
    if a.grpo and not a.matched:
        k = [m[0] for m in models].index('h9acont') + 1
        models = tuple(m for m in models[:k] + (GRPO,) + models[k:] if m[0] != 'h8allkimi')   # GRPO started from H9
    slots, at = [], 0.0
    for m in models:
        if m is None:
            at += GAP
        else:
            slots.append(at)
            at += 1
    slots = [x - (at - 1) / 2 for x in slots]
    models = tuple(m for m in models if m is not None)
    bold = ('h8allkimi', 'm22msafing2nrcf') if a.status else ('h8allkimi', 'm22msafin') if a.harness else ('t3tmax', 't1tmax') if a.tmax else ('mrel', 'mrelk') if a.matched else ('h8allkimi', 'h9acont', 'rlh9s30')
    r = json.load(open(a.data))
    r = r.get('readout', r)
    if a.extra:
        r.update(json.load(open(a.extra)))
    plt.rcParams.update({'font.size': 15, 'font.family': 'DejaVu Sans'})
    fig, ax = plt.subplots(figsize=(15, 7.6))
    w = min(0.16, 0.74 / (slots[-1] - slots[0] + 1))
    for i, (tag, name, col) in enumerate(models):
        xs, ps, ses = [], [], []
        for g, (s, _) in enumerate(SETS):
            v = r[f'{tag}/{s}']
            xs.append(g + slots[i] * (w + 0.012))
            ps.append(100 * v['pass_at_1'])
            ses.append(100 * v['se'])
        ax.bar(xs, ps, w, color=col, label=name, zorder=2, hatch='//' if tag in HATCHED else None,
               edgecolor='white' if tag in HATCHED else None, linewidth=0)
        ax.errorbar(xs, ps, yerr=ses, fmt='none', ecolor='#444', elinewidth=1.2, capsize=4, zorder=3)
        for x, p, se in zip(xs, ps, ses):
            ax.text(x, p + se + 1.2, f'{p:.1f}', ha='center', va='bottom', fontsize=12 if len(models) < 6 else 11,
                    fontweight='bold' if tag in bold else 'normal')
    ax.set_xticks(range(len(SETS)))
    ax.set_xticklabels([t for _, t in SETS], fontsize=15)
    ax.set_ylabel('pass@1 (%), mean of 3 runs')
    ax.set_ylim(0, 60 if a.matched or a.tmax or a.harness or a.status else 85)
    ax.grid(axis='y', color='#e5e5e5', zorder=0)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.16), frameon=False, fontsize=12.5,
              ncol=5 if a.matched or a.tmax else 4 if a.status else 3)
    if a.status:
        ax.set_title('Best SFT of Grug 67B-A2B 09-21 under each harness (2026-10-10)', fontsize=16, pad=14)
        fig.text(0.01, 0.01, 'Plain bars: Terminus-2 eval (compacts context when it fills). Hatched bars: mini-swe-agent 2 (MSA2) eval: 2.4.6 tool mode, 65,536 context, '
                 'earlier reasoning not re-sent, the attempt ends when the context fills.\nH8 = 5,916 CalibForge relay + 4,392 Kimi SWE-smith traces. '
                 'msafin = CalibForge + TMax relay rows that passed and end on the submit + Orchard MiniMax-M2.5 + SI2CA Qwen3.5-122B SWE traces.\n'
                 'msafing2 = its CalibForge rows regenerated with one-turn teacher assists. msafing2nrcf = those rows rendered as the eval sees them (no earlier reasoning).\n'
                 'Seed 0 of each; 3 epochs, LR 3e-4, 3 eval runs; error bars ±1 SE over tasks.', ha='left', fontsize=10, color='#555')
    elif a.harness:
        ax.set_title('Terminus-2 vs mini-swe-agent 2 (MSA2): best recipes, SFT of Grug 67B-A2B 09-21', fontsize=16, pad=14)
        fig.text(0.01, 0.01, 'Plain bars: Terminus-2 eval (compacts context when it fills). Hatched bars: MSA2 eval (mini-swe-agent 2.4.6 tool mode, 65,536 context, '
                 'earlier reasoning not re-sent; the attempt ends when the context fills).\nH8 = 5,916 CalibForge relay + 4,392 Kimi SWE-smith traces. Its MSA2 port '
                 '(msaallswe) = 3,561 CalibForge relay rows rendered for MSA2 + 4,500 NVIDIA mini-swe-agent SWE traces (Kimi\'s are Terminus-2 format).\n'
                 'msafin = CalibForge + TMax relay rows that passed and end on the submit + Orchard MiniMax-M2.5 + SI2CA Qwen3.5-122B SWE traces (22,372 rows). '
                 'All: 3 epochs, LR 3e-4, 3 eval runs; error bars ±1 SE over tasks.', ha='left', fontsize=10, color='#555')
    elif a.tmax:
        ax.set_title('Relay vs Qwen-only traces on TMax, matched: SFT of Grug 67B-A2B 09-21', fontsize=16, pad=14)
        fig.text(0.01, 0.01, 'Both trace sets cover the same 2,730 TMax task slots (2,126 tasks, 2,229 passes + 501 failures); only who played '
                 'the early turns differs. Trained tokens: Qwen-only 17.6M, relay 13.8M.\n"+ CalibForge relay + Kimi" adds the same '
                 '10,308 traces to both: 5,916 CalibForge relay traces and 4,392 Kimi SWE-smith traces. '
                 'Same recipe: 3 epochs, LR 3e-4. Error bars: ±1 standard error over tasks.', ha='left', fontsize=10.5, color='#555')
    elif a.matched:
        ax.set_title('Relay vs Qwen-only traces on CalibForge, matched: SFT of Grug 67B-A2B 09-21', fontsize=16, pad=14)
        fig.text(0.01, 0.01, 'Both trace sets cover the same 3,872 CalibForge task slots (2,204 tasks, 3,006 passes + 866 failures); only who played '
                 'the early turns differs.\nTrained tokens: Qwen-only 37.2M, relay 32.8M (+ 21.6M of Kimi SWE-smith traces in both). Same recipe: 3 epochs, LR 3e-4. '
                 'Error bars: ±1 standard error over tasks.', ha='left', fontsize=10.5, color='#555')
    else:
        ax.set_title('Relay SFT on Grug 67B-A2B 09-21 (Qwen3.8-27B teacher + Kimi SWE-smith traces)', fontsize=16, pad=14)
        note = 'Error bars: ±1 standard error over tasks. Models: huggingface.co/laion'
        if a.grpo:
            note = ('GRPO vs its start (paired per task, 95 % range): TB2.1 +4.2 [-0.2, +8.6], SWE -3.5 [-8.2, +1.0], TBLite +5.4 [-0.3, +11.6].\n'
                    + note)
        fig.text(0.99, 0.01, note, ha='right', fontsize=10.5, color='#666')
    fig.tight_layout(rect=(0, 0.13 if a.status else 0.09 if a.harness else 0.06 if a.matched or a.tmax else 0.02, 1, 1))
    fig.savefig(a.out, dpi=160, facecolor='white')


if __name__ == '__main__':
    main()
