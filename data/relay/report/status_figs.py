#!/usr/bin/env python3
"""status_figs.py — the teacher-relay status gist: six figures and a short README, from the run readouts.

    python status_figs.py figs   --cache <dir> --out <dir>
    python status_figs.py readme --cache <dir> --out <dir> --fig-base <raw url prefix ending in />

<cache> is what update.sh pulls from Jupiter: one dir per run under runs/ (readout*.json, check_decision*.json,
decide*.json, run.meta, ABORT, SUPERSEDED, launch logs), verify_note/{classified,replies}.jsonl, and
live_status.json (live_status.py's snapshot). Every number is read from those files, except the few that exist only
in the notes; those sit in NOTES_SPEND / NOTES_* below with the note they come from.

The narrative lines (VERDICT, BULLETS, NEXT) are the part to edit when the state of the project changes; their numbers
are filled from the data.
"""
import argparse
import datetime as dt
import glob
import json
import math
import os
import random
import re
import statistics
from zoneinfo import ZoneInfo

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

PT = ZoneInfo('America/Los_Angeles')
CEST = ZoneInfo('Europe/Berlin')

# ---- colors: one per model, fixed everywhere (validated reference palette, slots 1-3) ----
C_STUDENT = '#2a78d6'   # 09-21
C_QWEN = '#eb6834'      # Qwen3.8-27B
C_RELAY = '#1baf7a'     # relay (09-21 + Qwen)
C_QWEN_LIGHT = '#f5b99c'  # Qwen, one-turn repair
C_NONE = '#a8a79f'      # "without" / not gated
INK, INK2, GRID, SURF = '#0b0b0b', '#52514e', '#e4e3de', '#fcfcfb'
C_PASS_MARK, C_FAIL_MARK = '#0ca30c', '#d03b3b'   # status: always shown with a check / cross label
CAUSE = {  # failure causes (categorical slots 4+, never the model colors)
    'passed': '#d9d8d2', 'false done': '#4a3aa7', 'context overflow': '#e87ba4', 'timeout': '#eda100',
    'other failure': '#008300', 'harness error': '#6d6c66'}

plt.rcParams.update({
    'font.family': ['Helvetica Neue', 'Helvetica', 'Arial', 'DejaVu Sans'], 'font.size': 10,
    'axes.edgecolor': GRID, 'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2,
    'axes.titlesize': 11, 'axes.titleweight': 'bold', 'axes.titlecolor': INK, 'axes.titlelocation': 'left',
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.color': GRID,
    'grid.linewidth': 0.6, 'axes.axisbelow': True, 'figure.facecolor': SURF, 'axes.facecolor': SURF,
    'savefig.facecolor': SURF, 'legend.frameon': False, 'hatch.color': '#ffffff', 'hatch.linewidth': 1.2})

# ---- runs ----
# The relay checks in order: (label, run dir, decision file, readout file, format floor, overflow ceiling).
# The floors are each run's pre-registered rule (notes/relay/relay_full_t2.md); run 3 had no overflow gate.
CHECKS = [
    ('Run 3', 'relay_run3b_20260925', None, 'readout_amended.json', 0.99, None),
    ('Check', 'relay_check_20260925', 'check_decision_corrected.json', 'readout_decide.json', 0.99, 0.20),
    ('Re-check', 'relay_recheck_20260925', 'check_decision.json', 'readout_decide.json', 0.99, 0.20),
    ('Context\nbudget 1', 'relay_ctxb_20260926', 'decide_merged.json', 'readout_decide.json', 0.98, 0.20),
    ('Context\nbudget 2', 'relay_ctxb2_20260926', 'check_decision.json', 'readout.json', 0.98, 0.20),
]
LAUNCH_BAND = (0.20, 0.25)   # overflow band the plan pre-stated as "launch anyway" (relay_full_t2.md, 15:00 PT)
HELDOUT = ('heldout0921_20260925', 'student_only')
# Failure split: (label, run dir, arm). Rows whose readout.json is missing are skipped (live runs appear when done).
FAIL_RUNS = [
    ('Qwen alone, baseline 1\n2,043 tasks, queued serving', 'relay_full_baseline_20260925', 'control'),
    ('Qwen alone, baseline 4\nper-GPU serving, stopped at 25 min', 'relay_full_baseline4_20260926', 'control'),
    ('Relay, context budget 2\n100 pilot tasks', 'relay_ctxb2_20260926', 'relay_repair'),
    ('Qwen alone, baseline 6\n2,043 tasks', 'relay_full_baseline6_20260926', 'control'),
    ('Relay, full run\n945 tasks x 4', 'relay_full_relay_20260926', 'relay_repair'),
]
# Compute spend. Runs with a run.meta are read from it; the rest exist only in the notes.
SPEND_RUNS = [  # (label, run dir, date)
    ('Pilot run 1', 'relay_pilot_20260925', '09-25'), ('Pilot run 2', 'relay_repair_20260925', '09-25'),
    ('Pilot run 3', 'relay_run3b_20260925', '09-25'), ('09-21 held-out', 'heldout0921_20260925', '09-25'),
    ('Check', 'relay_check_20260925', '09-25'), ('Baseline 1', 'relay_full_baseline_20260925', '09-25'),
    ('Re-check', 'relay_recheck_20260925', '09-25'), ('Baseline rerun', 'relay_full_baseline2_20260925', '09-26'),
    ('Baseline 4', 'relay_full_baseline4_20260926', '09-26'), ('Context budget 1', 'relay_ctxb_20260926', '09-26'),
    ('Baseline 5', 'relay_full_baseline5_20260926', '09-26'), ('Context budget 2', 'relay_ctxb2_20260926', '09-26'),
]
NOTES_SPEND = [  # (label, node-h, status, date, source)
    ('MSA', 2.2, 'used', '09-24', 'session spend line'), ('Bench', 1.3, 'used', '09-24', 'session spend line'),
    ('IF', 0.9, 'used', '09-24', 'session spend line'),
    ('Run 3 failed start', 0.263, 'stopped', '09-25', 'relay_pilot_100.md, run 3 spend'),
    ('Baseline 3 failed start', 0.17, 'stopped', '09-26', 'relay_full_t2.md, clean baseline cost'),
    ('Verify-note replay', 0.49, 'used', '09-26', '2026-09-26_verify_note_replay.md'),
]
# Qwen's check quality: hand-labelled audit, numbers exist only in the note
# (ai_memory/active/snowball-sft/research/2026-09-26_qwen_teacher_check_quality.md).
NOTES_CHECKQ = dict(
    wrong=dict(fixed_passed=6, wrong_fix=2, missed=2), right=dict(confirmed=16, broken=0),   # ctxb2, 26 scored
    blind=dict(no_note=(14, 27), note=(0, 27)), pass_after=(22, 26), claim_right=(16, 26),
    confirmed_then_failed=[  # (label, k, n, note on, whose claim, approximate labels)
        ("09-21's claim, no note\n(context budget 1)", 7, 26, False, 'student', False),
        ("09-21's claim, note\n(context budget 2)", 4, 26, True, 'student', False),
        ("Qwen's own claim after a\ncontext-budget takeover", 10, 30, False, 'qwen', True),
        ('Qwen alone, no note\n(baseline 4)', 60, 374, False, 'qwen', True),
        ('Qwen alone, note\n(baseline 6, partial)', 9, 46, True, 'qwen', True)])
LIVE_CEILING = {'relay_full_relay_20260926': 30.0, 'relay_full_baseline6_20260926': 11.7}   # relay_full_t2.md
LIVE_LABEL = {'relay_full_relay_20260926': 'Relay full run', 'relay_full_baseline6_20260926': 'Baseline 6'}
LIVE_WHAT = {'relay_full_relay_20260926': 'Relay full run (945 solvable tasks x 4)',
             'relay_full_baseline6_20260926': 'Baseline 6, Qwen alone (2,043 tasks)'}

# ---- narrative (edit when the state changes; numbers come from the data) ----
VERDICT = ('The relay now passes as often as Qwen alone and its format problem is solved; '
           'overflow ({ovf:.0%}) is the one open issue, so the full relay run launched inside the pre-stated '
           '20–25 % band and is live now.')
BULLETS = [
    'Last check (100 pilot tasks): relay {relay_p:.0%} vs Qwen alone {qwen_p:.0%} on the same {n_pair} tasks '
    '({diff:+.0f} points, CI {dlo:+.0f} to {dhi:+.0f}). 09-21 alone passes {s_p:.0%} (held-out set).',
    'Format is fixed: {fmt:.2%} of executed steps parse, and 09-21 still plays {share:.0%} of the turns.',
    'Overflow is what is left: all {hard} overflows are hard ends after a context-budget takeover, and Qwen recovers '
    '{rec_cb:.0%} of those episodes vs {rec_dc:.0%} after a done claim.',
    'Spend since 09-24: {total:.1f} node-h, {waste:.1f} of it on runs that were stopped, failed to start or were '
    'superseded; the live relay run is capped at 30.',
]
NEXT = [
    ('Keep 1:1 pass:fail traces for both arms (`select_kept.py`)', 'after both runs finish'),
    ('SFT 09-21 on relay vs Qwen-alone traces; eval both on the 300 held-out tasks', 'after the keep step'),
]


# ---- helpers ----
def load(cache, *parts):
    p = os.path.join(cache, *parts)
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def wilson(k, n, z=1.96):
    if n == 0:
        return (0, 0, 0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, c - h, c + h


def meta(cache, run):
    p = os.path.join(cache, 'runs', run, 'run.meta')
    if not os.path.exists(p):
        return {}
    return dict(l.strip().split('=', 1) for l in open(p) if '=' in l)


def tidy(ax, xgrid=False):
    ax.grid(axis='x', visible=xgrid)
    ax.tick_params(length=0)


def save(fig, out, name):
    fig.savefig(os.path.join(out, name), dpi=150, bbox_inches='tight', pad_inches=0.25)
    plt.close(fig)


def check_rows(cache):
    """One dict per relay check run: the four gated numbers, their floors, and the verdict."""
    rows = []
    for label, run, dec_f, ro_f, fmt_floor, ovf_max in CHECKS:
        dec = load(cache, 'runs', run, dec_f) if dec_f else None
        ro = load(cache, 'runs', run, ro_f)
        rr = ro['relay_repair']
        oc = rr['outcomes']
        if dec:
            c2 = next(c for c in dec['checks'] if 'overflow' in c['check'])['detail']
            ovf = (c2['overflow'], c2['scored'])
            fmt = None
            for c in dec['checks']:
                d = c['detail'] if isinstance(c['detail'], dict) else {}
                fmt = fmt or d.get('valid_format_rate') or (d.get('format') or {}).get('valid_format_rate')
            paired = dec.get('paired_pass_vs_run3_control') or next(
                c['detail'] for c in dec['checks'] if 'paired' in c['check'])
            fails = []
            for c in dec['checks']:
                if c['ok']:
                    continue
                if 'overflow' in c['check']:
                    fails.append('overflow')
                elif 'format' in c['check'] and not (c['detail'] or {}).get('failed'):
                    fails.append('format')
                else:
                    fails.append('harness' if c['check'].startswith('C1') else c['check'].split()[0])
                    if 'format' in c['check'] and fmt is not None and fmt < fmt_floor:
                        fails.append('format')
            verdict = 'FAIL: ' + ', '.join(dict.fromkeys(fails)) if fails else 'PASS'
        else:
            ovf = (oc['context_overflow'], oc['scored'])
            fmt = rr['format']['valid_format_rate']
            paired = ro['paired_pass_relay_vs_control']
            fd = load(cache, 'runs', run, 'full_decision.json') or {}
            verdict = f"{fd.get('decision', '')}, then\nrule amended".strip(', ')
        ex = rr['repair']['executed_turns']
        tk = rr['takeovers']
        rows.append(dict(
            label=label, run=run, ovf=ovf[0] / ovf[1], ovf_n=ovf, ovf_max=ovf_max, fmt=fmt, fmt_floor=fmt_floor,
            share=ex['student'] / (ex['student'] + ex['teacher_repair'] + ex['teacher_sticky']),
            rec=tk['recovery'], rec_ci=tk['recovery_ci95'], paired=paired, verdict=verdict, ex=ex, tk=tk, ro=rr))
    launched = ''
    for f in glob.glob(os.path.join(cache, 'runs', 'relay_full_relay_*.launch.log')):
        txt = open(f).read()
        if 'LAUNCHED' in txt and rows[-1]['run'] in txt:
            launched = '\n→ full run launched'
    rows[-1]['verdict'] += launched
    return rows


# ---- figures ----
def fig_checks(cache, out, rows):
    x = list(range(len(rows)))
    fig, axs = plt.subplots(2, 2, figsize=(12, 7.6))
    panels = [
        ('Episodes that overflow the 64k context', 'ovf', 'ovf_max', 'max', (0, 0.6)),
        ('Executed steps in valid Terminus-2 format', 'fmt', 'fmt_floor', 'min', (0.955, 1.006)),
        ("09-21's share of executed turns", 'share', None, 'min', (0, 1)),
        ('Pass rate after Qwen takes over', 'rec', None, 'min', (0, 1)),
    ]
    for ax, (title, key, thr_key, kind, ylim) in zip(axs.flat, panels):
        vals = [r[key] for r in rows]
        thr = [r[thr_key] for r in rows] if thr_key else [0.50 if key == 'share' else 0.20] * len(rows)
        ax.plot(x, vals, color=C_RELAY, lw=2, zorder=2)
        if key == 'rec':
            for i, r in enumerate(rows):
                ax.plot([i, i], r['rec_ci'], color=C_RELAY, lw=1.2, alpha=0.5)
        # threshold as a step line (it changed between rules)
        for i, t in enumerate(thr):
            if t is not None:
                ax.plot([i - 0.4, i + 0.4], [t, t], color=INK2, lw=1.2, ls=(0, (3, 2)))
        if key == 'ovf':
            ax.axhspan(*LAUNCH_BAND, xmin=0.8, xmax=1.0, color=CAUSE['timeout'], alpha=0.18, lw=0)
            ax.text(len(rows) - 1.45, LAUNCH_BAND[1] + 0.01, 'launch band\n20–25 %', fontsize=8, color=INK2,
                    va='bottom', ha='center')
        for i, (v, t) in enumerate(zip(vals, thr)):
            if t is None:
                ax.scatter(i, v, s=70, facecolor=SURF, edgecolor=C_NONE, lw=2, zorder=3)
                mark, col = 'no gate', INK2
            else:
                ok = v <= t if kind == 'max' else v >= t
                ax.scatter(i, v, s=70, color=C_PASS_MARK if ok else C_FAIL_MARK, zorder=3, edgecolor=SURF, lw=1.5)
                mark, col = ('✓ ' if ok else '✗ '), INK
            txt = f'{v:.2%}' if key == 'fmt' else f'{v:.0%}' if key != 'rec' else f'{v:.2f}'
            dy = 10 if (key != 'ovf' or i != len(rows) - 1) else -16
            ax.annotate(f'{mark}{txt}' if t is not None else f'{txt}\n({mark})', (i, v), textcoords='offset points',
                        xytext=(0, dy), ha='center', fontsize=8.5, color=col)
        ax.set_ylim(*ylim)
        ax.set_xlim(-0.5, len(rows) - 0.5)
        ax.set_xticks(x)
        ax.set_xticklabels([r['label'] for r in rows], fontsize=8.5)
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0 if key != 'fmt' else 1)
                                     if key != 'rec' else matplotlib.ticker.FormatStrFormatter('%.1f'))
        sense = 'lower is better' if kind == 'max' else 'higher is better'
        ax.set_title(f'{title}  ', loc='left')
        ax.set_title(f'{title}\n', loc='left')
        ax.text(0, 1.02, f'dashed line = pass line ({sense})', transform=ax.transAxes, ha='left', fontsize=8,
                color=INK2)
        tidy(ax)
    for ax in axs[1]:   # verdict under each run, on the bottom row
        ax.set_xticklabels([f"{r['label']}\n\n{r['verdict'].replace(': ', ':' + chr(10))}" for r in rows], fontsize=8.5)
        for tl, r in zip(ax.get_xticklabels(), rows):
            tl.set_color(C_FAIL_MARK if r['verdict'].startswith(('FAIL', 'HOLD')) else INK2)
    last = rows[-1]
    fig.suptitle(f"How we got here: every relay check now passes except overflow ({last['ovf']:.0%}), which landed "
                 'in the launch band', x=0.01, ha='left', fontsize=13, fontweight='bold', color=INK)
    fig.text(0.01, 0.945, 'Relay on the same 100 pilot tasks, one run per check. ✓/✗ = against that run\'s own '
             'pre-registered line. Bars on the last panel are 95 % CIs.', fontsize=9, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.94), h_pad=3.5)
    save(fig, out, 'fig1_checks.png')


def fig_pass(cache, out, rows):
    ho = load(cache, 'runs', HELDOUT[0], 'readout.json')[HELDOUT[1]]['outcomes']
    last = rows[-1]
    pp = last['paired']
    n = pp['tasks']
    q_k, r_k = pp['both_pass'] + pp['control_only'], pp['both_pass'] + pp['relay_only']
    bars = [('09-21 alone', ho['passes'], ho['scored'], C_STUDENT, f"{ho['scored']} held-out tasks\n(different set)"),
            ('Qwen alone', q_k, n, C_QWEN, f'{n} pilot tasks\n(run 3)'), ('Relay', r_k, n, C_RELAY, f'same {n} tasks')]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw=dict(width_ratios=[1, 1.25]))
    for i, (lab, k, m, col, sub) in enumerate(bars):
        p, lo, hi = wilson(k, m)
        a1.bar(i, p, width=0.6, color=col, hatch='//' if i == 0 else None, edgecolor=SURF, lw=0)
        a1.errorbar(i, p, yerr=[[p - lo], [hi - p]], color=INK, lw=1.2, capsize=4)
        a1.text(i, hi + 0.03, f'{p:.0%}', ha='center', fontsize=11, fontweight='bold', color=INK)
        a1.text(i, -0.1, sub, ha='center', va='top', fontsize=8, color=INK2, transform=a1.get_xaxis_transform())
    a1.set_xticks(range(3))
    a1.set_xticklabels([b[0] for b in bars], fontsize=10)
    a1.set_ylim(0, 0.85)
    a1.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    a1.set_title('Pass rate, latest check (95 % Wilson CIs)')
    tidy(a1)
    # right: the paired gap across the checks
    xs = list(range(len(rows)))
    d = [r['paired']['relay_minus_control'] for r in rows]
    ci = [r['paired']['ci95_bootstrap'] for r in rows]
    a2.axhline(0, color=INK2, lw=1)
    for i in xs:
        a2.plot([i, i], ci[i], color=C_RELAY, lw=2, alpha=0.45)
    a2.plot(xs, d, color=C_RELAY, lw=2, marker='o', ms=7, markeredgecolor=SURF)
    for i in xs:
        a2.annotate(f'{d[i] * 100:+.0f} pts\nn={rows[i]["paired"]["tasks"]}', (i, ci[i][1]), textcoords='offset points',
                    xytext=(0, 5), ha='center', fontsize=8, color=INK)
    a2.set_xticks(xs)
    a2.set_xticklabels([r['label'] for r in rows], fontsize=8.5)
    a2.set_ylim(-0.3, 0.3)
    a2.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f'{v * 100:+.0f} pts'))
    a2.set_title('Relay minus Qwen alone (run 3), paired by task, per check')
    tidy(a2)
    fig.suptitle(f"Relay passes as often as Qwen alone ({d[-1] * 100:+.0f} points, CI {ci[-1][0] * 100:+.0f} to "
                 f"{ci[-1][1] * 100:+.0f}); "
                 '09-21 alone almost never does', x=0.01, ha='left', fontsize=13, fontweight='bold', color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93), w_pad=3)
    save(fig, out, 'fig2_pass.png')


def fig_work(cache, out, rows):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw=dict(width_ratios=[1.35, 1]))
    ys = list(range(len(rows)))[::-1]
    for y, r in zip(ys, rows):
        ex = r['ex']
        tot = ex['student'] + ex['teacher_repair'] + ex['teacher_sticky']
        left = 0
        for key, col, hatch in (('student', C_STUDENT, None), ('teacher_repair', C_QWEN_LIGHT, None),
                                ('teacher_sticky', C_QWEN, None)):
            w = ex[key] / tot
            a1.barh(y, w, left=left, color=col, hatch=hatch, edgecolor=SURF, lw=2, height=0.62)
            if w >= 0.07:
                a1.text(left + w / 2, y, f'{w:.0%}', ha='center', va='center', fontsize=8.5,
                        color=INK if key == 'teacher_repair' else 'white', fontweight='bold')
            left += w
    a1.set_yticks(ys)
    a1.set_yticklabels([r['label'].replace('\n', ' ') for r in rows], fontsize=9)
    a1.set_xlim(0, 1)
    a1.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    a1.set_title('Who plays the executed turns')
    a1.legend(handles=[Patch(color=C_STUDENT, label='09-21'),
                       Patch(color=C_QWEN_LIGHT, label='Qwen repairs a bad reply (09-21 keeps the episode)'),
                       Patch(color=C_QWEN, label='Qwen after a takeover')],
              loc='upper center', bbox_to_anchor=(0.5, -0.08), ncol=3, fontsize=8.5)
    tidy(a1, xgrid=True)
    a1.grid(axis='y', visible=False)
    # right: takeovers by trigger in the last check, with the pass rate after each
    last = rows[-1]
    trig = last['tk']['by_trigger']
    names = {'context_budget': 'Context budget\n(09-21 view ≥ 32k)', 'done_claim': '09-21 claims done',
             'loop': 'Loop', 'no_progress_wait': 'No progress'}
    items = sorted(trig.items(), key=lambda kv: -kv[1]['n'])
    nt = last['tk']['no_takeover']
    items.append(('none', dict(n=nt['n'], scored=nt['scored'], passes=nt['passes'])))
    names['none'] = 'No takeover\n(09-21 finishes)'
    ys2 = list(range(len(items)))[::-1]
    mx = max(v['n'] for _, v in items)
    for y, (k, v) in zip(ys2, items):
        col = C_STUDENT if k == 'none' else C_QWEN
        a2.barh(y, v['n'], color=col, height=0.55, edgecolor=SURF)
        p, lo, hi = wilson(v['passes'], v['scored'])
        a2.text(v['n'] + mx * 0.03, y, f"{v['n']} episodes · {v['passes']}/{v['scored']} pass = {p:.0%}  "
                f"[{lo:.0%}–{hi:.0%}]", va='center', fontsize=8.5, color=INK)
    a2.set_yticks(ys2)
    a2.set_yticklabels([names.get(k, k) for k, _ in items], fontsize=9)
    a2.set_xlim(0, mx * 2.3)
    a2.set_title(f"Why Qwen takes over ({last['label'].replace(chr(10), ' ')})")
    a2.set_xlabel('episodes (of 100)')
    tidy(a2, xgrid=True)
    a2.grid(axis='y', visible=False)
    cb = trig.get('context_budget')
    dc = trig.get('done_claim')
    t = ("09-21 still plays half the turns; the context-budget takeover is the common one and recovers least "
         f"({cb['recovery']:.0%} vs {dc['recovery']:.0%})" if cb and dc else 'Who does the work in a relay episode')
    fig.suptitle(t, x=0.01, ha='left', fontsize=13, fontweight='bold', color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93), w_pad=3)
    save(fig, out, 'fig3_work.png')


def fail_split(ro_arm):
    oc = ro_arm['outcomes']
    herr = {k: v for k, v in oc['harness_errors'].items() if k != 'CancelledError'}
    fc = dict(oc['failure_causes'])
    parts = {'passed': oc['passes'], 'false done': fc.pop('false_done', 0),
             'context overflow': fc.pop('context_overflow', 0), 'timeout': fc.pop('timeout', 0)}
    parts['other failure'] = sum(fc.values())
    parts['harness error'] = sum(herr.values()) + oc.get('verifier_timeouts', 0) + oc.get('censored_by_deadline', 0)
    return parts, sum(parts.values()), oc['harness_errors'].get('CancelledError', 0)


def fig_fail(cache, out):
    runs = []
    for label, run, arm in FAIL_RUNS:
        ro = load(cache, 'runs', run, 'readout.json')
        if ro and arm in ro:
            runs.append((label, *fail_split(ro[arm])))
    fig, ax = plt.subplots(figsize=(12, 1.2 + 1.0 * len(runs)))
    ys = list(range(len(runs)))[::-1]
    for y, (label, parts, tot, canc) in zip(ys, runs):
        left = 0
        for k, v in parts.items():
            w = v / tot
            ax.barh(y, w, left=left, color=CAUSE[k], edgecolor=SURF, lw=2, height=0.6)
            if w >= 0.035:
                ax.text(left + w / 2, y, f'{w:.0%}', ha='center', va='center', fontsize=8.5,
                        color=INK if k in ('passed', 'timeout') else 'white', fontweight='bold')
            left += w
        ax.text(1.01, y, f'n={tot:,}' + (f'\n(+{canc} cancelled)' if canc else ''), va='center', fontsize=8,
                color=INK2, transform=ax.get_yaxis_transform())
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in runs], fontsize=9)
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.legend(handles=[Patch(color=c, label=k) for k, c in CAUSE.items()], loc='upper center',
              bbox_to_anchor=(0.5, -0.12 - 0.02 * (5 - len(runs))), ncol=6, fontsize=9)
    tidy(ax, xgrid=True)
    ax.grid(axis='y', visible=False)
    b1 = runs[0][1]['timeout'] / runs[0][2]
    fig.suptitle(f"Timeouts are gone ({b1:.0%} of baseline 1 were queue waits); false done claims and context "
                 'overflow are what is left', x=0.01, ha='left', fontsize=13, fontweight='bold', color=INK)
    fig.text(0.01, 1 - 0.55 / (1.2 + 1.0 * len(runs)), 'How each finished episode ended (share of finished '
             'episodes). Baseline 4 only scored its quickest episodes before it was stopped.', fontsize=9, color=INK2)
    fig.tight_layout(rect=(0, 0, 0.95, 1 - 0.7 / (1.2 + 1.0 * len(runs))))
    save(fig, out, 'fig4_failures.png')


def verify_numbers(cache):
    rows = [json.loads(l) for l in open(os.path.join(cache, 'verify_note', 'classified.jsonl'))]
    by = {}
    for r in rows:
        by.setdefault(r['key'], {}).setdefault(r['variant'], []).append(r)
    keys = [k for k, v in by.items() if v.get('A') and v.get('B')]
    preds = {'runs commands first': lambda r: r['cls'].startswith('runs_first'),
             'confirms blind': lambda r: r['cls'] == 'confirm',
             'format error': lambda r: r['cls'] == 'parse_error'}
    rnd = random.Random(0)
    res = {}
    for name, f in preds.items():
        per = {v: [sum(map(f, by[k][v])) / len(by[k][v]) for k in keys] for v in 'AB'}
        res[name] = {}
        for v in 'AB':
            xs = per[v]
            bs = sorted(sum(rnd.choice(xs) for _ in xs) / len(xs) for _ in range(2000))
            res[name][v] = (sum(xs) / len(xs), bs[50], bs[1949])
    reps = {}
    for l in open(os.path.join(cache, 'verify_note', 'replies.jsonl')):
        r = json.loads(l)
        if r.get('status') == 200:
            reps.setdefault((r['key'], r['variant'], r['sample']), r)
    tok = {v: dict(reply=statistics.median((r.get('usage') or {}).get('completion_tokens') or 0
                                           for (k, vv, s), r in reps.items() if vv == v and k in keys),
                   thinking=statistics.median(r.get('reasoning_tokens') or 0
                                              for (k, vv, s), r in reps.items() if vv == v and k in keys))
           for v in 'AB'}
    return res, tok, len(keys)


def fig_verify(cache, out, rows):
    res, tok, n = verify_numbers(cache)
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(13, 4.6), gridspec_kw=dict(width_ratios=[1.3, 0.8, 1.2]))
    names = list(res)
    x = range(len(names))
    for off, v, col, lab in ((-0.18, 'A', C_NONE, 'without note'), (0.18, 'B', C_QWEN, 'with note')):
        for i, nm in enumerate(names):
            m, lo, hi = res[nm][v]
            a1.bar(i + off, m, width=0.34, color=col, edgecolor=SURF, label=lab if i == 0 else None)
            a1.errorbar(i + off, m, yerr=[[m - lo], [hi - m]], color=INK, lw=1, capsize=3)
            a1.text(i + off, hi + 0.02, f'{m:.0%}', ha='center', fontsize=9, color=INK)
    a1.set_xticks(list(x))
    a1.set_xticklabels(names, fontsize=9)
    a1.set_ylim(0, 1.08)
    a1.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    a1.set_title(f'Replay: Qwen\'s reply to the confirmation\n({n} takeovers × 4 samples each, 95 % CIs)')
    a1.legend(fontsize=8.5, loc='upper right')
    tidy(a1)
    # reply length
    for i, what in enumerate(('reply', 'thinking')):
        for off, v, col in ((-0.18, 'A', C_NONE), (0.18, 'B', C_QWEN)):
            a2.bar(i + off, tok[v][what], width=0.34, color=col, edgecolor=SURF)
            a2.text(i + off, tok[v][what] + 30, f'{tok[v][what]:,.0f}', ha='center', fontsize=9, color=INK)
    a2.set_xticks([0, 1])
    a2.set_xticklabels(['whole reply', 'thinking'], fontsize=9)
    a2.set_title('Replay: median tokens\nper reply')
    tidy(a2)
    # live runs: at the done-claim takeover, did Qwen run a command before confirming?
    live = []
    for r in rows:
        dc = r['tk']['by_trigger'].get('done_claim')
        if dc:
            worked = dc['n'] - dc['teacher_confirmed_without_a_command']
            note = (r['ro'].get('teacher_format') or {}).get('verify_notes', 0) > 0
            live.append((r['label'].replace('\n', ' '), worked, dc['n'], note))
    for i, (lab, k, m, note) in enumerate(live):
        p, lo, hi = wilson(k, m)
        a3.bar(i, p, width=0.6, color=C_QWEN if note else C_NONE, edgecolor=SURF)
        a3.errorbar(i, p, yerr=[[p - lo], [hi - p]], color=INK, lw=1, capsize=3)
        a3.text(i, hi + 0.03, f'{k}/{m}', ha='center', fontsize=9, color=INK)
    a3.set_xticks(range(len(live)))
    a3.set_xticklabels([l[0].replace('Context budget', 'Ctx budget') for l in live], fontsize=8.5, rotation=0)
    a3.set_ylim(0, 1.15)
    a3.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    a3.set_title('Live relay runs: Qwen runs a command before\nconfirming 09-21\'s done claim (orange = note on)')
    tidy(a3)
    b = res['runs commands first']
    last = live[-1]
    fig.suptitle(f"The verify note makes Qwen check before confirming: {b['A'][0]:.0%} → {b['B'][0]:.0%} in replay, "
                 f"{last[1]} of {last[2]} live", x=0.01, ha='left', fontsize=13, fontweight='bold', color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.92), w_pad=2.5)
    save(fig, out, 'fig5_verify_note.png')
    return res, tok


def live_runs(cache):
    ls = load(cache, 'live_status.json') or {}
    now = ls.get('now_epoch', 0)
    out = []
    for r in ls.get('runs', []):
        start = dt.datetime.fromisoformat(r['start']).replace(tzinfo=CEST) if r.get('start', 'N/A')[:1].isdigit() else None
        lim = [int(p) for p in re.split('[-:]', r['limit'])]
        lim_s = sum(v * m for v, m in zip(lim[::-1], (1, 60, 3600, 86400)))
        el = [int(p) for p in re.split('[-:]', r['elapsed'])]
        el_s = sum(v * m for v, m in zip(el[::-1], (1, 60, 3600, 86400)))
        nh = r['node_h'] if r.get('node_h') is not None else r['nodes'] * el_s / 3600
        hard = start + dt.timedelta(seconds=lim_s) if start else None
        eta = None
        if r.get('harbor_start') and r['finished'] >= 30 and r.get('planned'):
            hs = dt.datetime.fromisoformat(r['harbor_start']).timestamp()
            rate = r['finished'] / max(now - hs, 1)
            eta = dt.datetime.fromtimestamp(hs + r['planned'] / rate, PT)
        exc = r.get('exceptions', {})
        herr = sum(v for k, v in exc.items() if k not in ('none', 'ContextLengthExceededError', 'AgentTimeoutError'))
        out.append(dict(r, node_h_now=nh, hard=hard.astimezone(PT) if hard else None, eta=eta,
                        scored=r['finished'] - herr))
    return out, now


def spend_items(cache):
    items = []
    for label, run, date in SPEND_RUNS:
        m = meta(cache, run)
        if not m:
            continue
        d = os.path.join(cache, 'runs', run)
        status = 'stopped' if os.path.exists(os.path.join(d, 'ABORT')) else \
            'superseded' if os.path.exists(os.path.join(d, 'SUPERSEDED')) else 'used'
        items.append((label, float(m['node_hours']), status, date))
    for label, nh, status, date, _src in NOTES_SPEND:
        items.append((label, nh, status, date))
    live, _ = live_runs(cache)
    for r in live:
        items.append((LIVE_LABEL.get(r['run'], r['run']) + ' (so far)', r['node_h_now'], 'running', '09-26'))
    return items


def fig_spend(cache, out):
    items = spend_items(cache)
    items.sort(key=lambda t: t[1])
    style = {'used': dict(color='#52514e', label='result used'),
             'superseded': dict(color='#b9b8b1', label='superseded (replaced by a later run)'),
             'stopped': dict(color=C_FAIL_MARK, label='stopped early or failed to start'),
             'running': dict(color=C_RELAY, label='running now (so far; outline = ceiling)')}
    fig, ax = plt.subplots(figsize=(12, 0.9 + 0.33 * len(items)))
    ceil_by_label = {LIVE_LABEL[k] + ' (so far)': v for k, v in LIVE_CEILING.items()}
    for y, (label, nh, st, date) in enumerate(items):
        if label in ceil_by_label:
            ax.barh(y, ceil_by_label[label], color='none', edgecolor=C_RELAY, lw=1.2, ls='--', height=0.66)
        ax.barh(y, nh, color=style[st]['color'], height=0.66, edgecolor=SURF)
        extra = f' of ≤ {ceil_by_label[label]:g}' if label in ceil_by_label else ''
        ax.text(max(nh, ceil_by_label.get(label, 0)) + 0.15, y, f'{nh:.2f}{extra}', va='center', fontsize=8.5,
                color=INK)
    ax.set_yticks(range(len(items)))
    ax.set_yticklabels([f'{lab}  ·  {d}' for lab, _, _, d in items], fontsize=8.5)
    ax.set_xlabel('node-hours (GH200 nodes)')
    used = {s for _, _, s, _ in items}
    ax.legend(handles=[Patch(color=style[s]['color'], label=style[s]['label']) for s in style if s in used],
              loc='lower right', fontsize=8.5)
    tidy(ax, xgrid=True)
    ax.grid(axis='y', visible=False)
    total = sum(t[1] for t in items)
    waste = sum(t[1] for t in items if t[2] in ('stopped', 'superseded'))
    fig.suptitle(f'{total:.1f} node-hours since 09-24; {waste:.1f} went to runs that were stopped, failed to '
                 'start or were superseded', x=0.01, ha='left', fontsize=13, fontweight='bold', color=INK)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.45 / (0.9 + 0.33 * len(items))))
    save(fig, out, 'fig6_spend.png')
    return total, waste


def fig_checkq(out):
    q = NOTES_CHECKQ
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.9), gridspec_kw=dict(width_ratios=[1.15, 1]))
    segs = [('wrong', 'fixed_passed', 'caught, fixed, passed', C_QWEN, None, 'white'),
            ('wrong', 'wrong_fix', 'caught, fixed the wrong way (failed)', '#8a80d0', None, 'white'),
            ('wrong', 'missed', 'missed: confirmed, failed', CAUSE['false done'], None, 'white'),
            ('right', 'confirmed', 'confirmed, passed', C_QWEN_LIGHT, None, INK),
            ('right', 'broken', 'broke it', CAUSE['context overflow'], None, 'white')]
    ys = {'wrong': 1, 'right': 0}
    left = {'wrong': 0, 'right': 0}
    for grp, k, lab, col, hatch, tc in segs:
        v = q[grp][k]
        if v:
            a1.barh(ys[grp], v, left=left[grp], color=col, hatch=hatch, edgecolor=SURF, lw=2, height=0.6)
            a1.text(left[grp] + v / 2, ys[grp], f'{v}', ha='center', va='center', fontsize=10, color=tc,
                    fontweight='bold')
        left[grp] += v
    nw, nr = sum(q['wrong'].values()), sum(q['right'].values())
    a1.text(nr + 0.3, 0, f"0 broken", va='center', fontsize=9, color=INK)
    a1.set_yticks([1, 0])
    a1.set_yticklabels([f"09-21's claim wrong\n({nw} of {nw + nr})", f"09-21's claim right\n({nr} of {nw + nr})"],
                       fontsize=9.5)
    a1.set_xlim(0, max(nw, nr) + 2.5)
    a1.set_xlabel('done-claim takeovers (with the verify note, context budget 2, hand-labelled)')
    a1.legend(handles=[Patch(color=c, label=l) for _, _, l, c, _, _ in segs[:4]], loc='upper center',
              bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=8.5)
    pa, cr = q['pass_after'], q['claim_right']
    bn, bw = q['blind']['no_note'], q['blind']['note']
    a1.set_title(f"Checking 09-21's claim: pass {cr[0]}/{cr[1]} if confirmed as-is → {pa[0]}/{pa[1]} after Qwen's "
                 f"check;\nblind confirmations {bn[0]}/{bn[1]} without the note → {bw[0]}/{bw[1]} with it", fontsize=10)
    tidy(a1, xgrid=True)
    a1.grid(axis='y', visible=False)
    rows = q['confirmed_then_failed']
    ys = list(range(len(rows)))[::-1]
    for y, (lab, k, n, note, who, approx) in zip(ys, rows):
        p, lo, hi = wilson(k, n)
        a2.barh(y, p, height=0.6, color=C_QWEN if note else C_NONE, edgecolor=SURF)
        a2.errorbar(p, y, xerr=[[p - lo], [hi - p]], color=INK, lw=1, capsize=3)
        a2.text(hi + 0.015, y, f'{k}/{n} = {p:.0%}' + ('*' if approx else ''), va='center', fontsize=9, color=INK)
    a2.set_yticks(ys)
    a2.set_yticklabels([r[0] for r in rows], fontsize=8.5)
    for tl, r in zip(a2.get_yticklabels(), rows):
        tl.set_fontweight('bold' if r[4] == 'student' else 'normal')
    a2.axhline(ys[1] - 0.5, color=INK2, lw=0.8, ls=':')
    a2.text(0.6, ys[1] - 0.5, "above: checking 09-21's claim\nbelow: Qwen checking its own", fontsize=8,
            color=INK2, ha='right', va='center')
    a2.set_xlim(0, 0.62)
    a2.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    a2.set_title('Confirmed as done, then failed the tests (95 % CIs)\norange = note on; * = automatic labels, '
                 'approximate', fontsize=10)
    tidy(a2, xgrid=True)
    a2.grid(axis='y', visible=False)
    fig.suptitle("Qwen catches 09-21's wrong claims, not its own: 8 of 10 caught, 0 of 16 right claims broken; "
                 'the note does not lower its own false done', x=0.01, ha='left', fontsize=12.5, fontweight='bold',
                 color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93), w_pad=3)
    save(fig, out, 'fig7_check_quality.png')


FIGS = ['fig1_checks.png', 'fig2_pass.png', 'fig3_work.png', 'fig4_failures.png', 'fig5_verify_note.png',
        'fig6_spend.png', 'fig7_check_quality.png']


def make_figs(cache, out):
    rows = check_rows(cache)
    fig_checks(cache, out, rows)
    fig_pass(cache, out, rows)
    fig_work(cache, out, rows)
    fig_fail(cache, out)
    fig_verify(cache, out, rows)
    fig_spend(cache, out)
    fig_checkq(out)


def hm(t):
    return t.strftime('%H:%M') if t else '?'


def make_readme(cache, out, base):
    rows = check_rows(cache)
    last = rows[-1]
    ho = load(cache, 'runs', HELDOUT[0], 'readout.json')[HELDOUT[1]]['outcomes']
    pp = last['paired']
    n = pp['tasks']
    cb = last['ro']['context_budget']
    items = spend_items(cache)
    total = sum(t[1] for t in items)
    waste = sum(t[1] for t in items if t[2] in ('stopped', 'superseded'))
    res, tok, nk = verify_numbers(cache)
    v = dict(ovf=last['ovf'], relay_p=(pp['both_pass'] + pp['relay_only']) / n,
             qwen_p=(pp['both_pass'] + pp['control_only']) / n, n_pair=n, diff=100 * pp['relay_minus_control'],
             dlo=100 * pp['ci95_bootstrap'][0], dhi=100 * pp['ci95_bootstrap'][1], s_p=ho['pass_rate'], fmt=last['fmt'],
             share=last['share'], hard=cb['hard_ends'], rec_cb=cb['recovery_after_context_budget']['recovery'],
             rec_dc=cb['recovery_after_other_takeovers']['recovery'], total=total, waste=waste)
    b1 = load(cache, 'runs', FAIL_RUNS[0][1], 'readout.json')[FAIL_RUNS[0][2]]['outcomes']
    b1_to = b1['failure_causes'].get('timeout', 0) / b1['trials']
    live, now = live_runs(cache)
    stamp = dt.datetime.fromtimestamp(now or dt.datetime.now().timestamp(), PT)
    L = ['# Teacher relay: status', '', f'**{VERDICT.format(**v)}**', '']
    L += [f'- {b.format(**v)}' for b in BULLETS]
    caps = [
        f"The four gated numbers across the five relay checks on the pilot's 100 tasks. Only overflow still misses "
        f"its line ({last['ovf_n'][0]}/{last['ovf_n'][1]} scored episodes).",
        '09-21 alone is on the 300 held-out tasks; Qwen alone and relay are the same tasks, paired. Right: the gap '
        'closed as the kit was fixed.',
        "Left: executed turns by owner per check. Right: takeover triggers in the last check, with Qwen's pass rate "
        'after each.',
        f'Baseline 1 lost {b1_to:.0%} of episodes to queue waits; per-GPU serving removed them. Live runs join this '
        'chart when their readout lands.',
        f"Replay of {nk} real done-claim takeovers, with and without the note, plus what the live relay runs did. "
        f"The note costs longer replies ({tok['A']['reply']:,.0f} → {tok['B']['reply']:,.0f} tokens) and a few more "
        'format errors.',
        'Node-hours per run since 09-24, from run.meta (MSA, bench, IF, failed starts and the replay from the notes).',
        'Hand-labelled audit of the done-claim takeovers against the hidden tests. Misses are hidden conventions Qwen '
        "never questions; checking its own claim repeats its own assumptions.",
    ]
    for f, c in zip(FIGS, caps):
        L += ['', f'![{f}]({base}{f})', f'*{c}*']
    L += ['', '## Now running / next', '', '| What | Status | When (PT) |', '|---|---|---|']
    for r in live:
        what = f"{LIVE_WHAT.get(r['run'], r['run'])} · job {r['job']}"
        if r['finished']:
            status = f"{r['finished']:,} of {r['planned']:,} episodes done"
            if r['scored'] >= 20:
                ovf = r['exceptions'].get('ContextLengthExceededError', 0)
                status += f", pass {r['passes'] / r['scored']:.0%} so far, {ovf} overflows"
            status += f", {r['node_h_now']:.1f} node-h"
        else:
            status = f"servers starting, {r['node_h_now']:.1f} node-h"
        sr = re.search(r'stop rule \S+: (ok|evaluation failed|\d+ scored)', r.get('stop_rule') or '')
        if sr:
            status += '; stop rule ' + ('waiting for 300 scored' if sr.group(1)[0].isdigit() else sr.group(1))
        when = f"hard stop {hm(r['hard'])}"
        if r['eta'] and r['hard'] and r['eta'] > r['hard']:
            when += f"; at the current rate it would need until ≈ {hm(r['eta'])}, so the cap binds"
        elif r['eta']:
            when = f"done ≈ {hm(r['eta'])} at the current rate; " + when
        L.append(f'| {what} | {status} | {when} |')
    if not live:
        L.append('| (nothing running) | | |')
    for what, when in NEXT:
        L.append(f'| Next: {what} | waiting | {when} |')
    L += ['', f'Updated {stamp:%Y-%m-%d %H:%M} PT.', '']
    with open(os.path.join(out, 'README.md'), 'w') as f:
        f.write('\n'.join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('what', choices=['figs', 'readme'])
    ap.add_argument('--cache', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--fig-base', default='')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.what == 'figs':
        make_figs(a.cache, a.out)
    else:
        make_readme(a.cache, a.out, a.fig_base)


if __name__ == '__main__':
    main()
