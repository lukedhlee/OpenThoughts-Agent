"""Figures for the relay SFT gist (2026-09-28). Reads the eval outputs on Jupiter; writes fig1_results.png, fig2_why.png."""
import json, sys, os
sys.path.insert(0, '/e/project1/transfernetx/lee27/code/ota-sft-v2/data/relay/sft')
sys.path.insert(0, '/e/project1/transfernetx/lee27/code/ota-sft-v2/data/relay/pilot')
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from paired_eval import tb2_outcomes, rate
import glob
sys.path.insert(0, '/e/project1/transfernetx/lee27/code/ota-sft-v2/data/relay/pilot')
import readout
OUT = os.path.dirname(os.path.abspath(__file__))
J = '/e/data1/mmlaion/lee27/experiments/tb2_jobs'
res = json.load(open('/e/fscratch/reformo/lee27/experiments/relay/sft_v2/results.json'))
def allrate(dirs):
    n = k = 0
    for d in dirs:
        for f in glob.glob(d + '/*/result.json'):
            r = json.load(open(f)); n += 1
            k += ((((r.get('verifier_result') or {}).get('rewards') or {}).get('reward')) or 0) >= 1
    lo, hi = readout.wilson(k, n)
    return dict(n=n, passes=k, rate=k / n, ci95=[lo, hi])
H = '/e/data1/mmlaion/lee27/experiments/relay_pilot_jobs'
def sharded(p, m):
    o = {}
    for s in (0, 1): o.update(tb2_outcomes(f'{J}/{p}_6516_{m}_s{s}_20260928')[0])
    return rate(o)
def shard4(p):
    o = {}
    for s in range(4): o.update(tb2_outcomes(f'{J}/{p}_6516_MIX_s{s}_20260928')[0])
    return rate(o)
data = {'Held-out CalibForge (300)': {m: allrate([f'{H}/heldout6516_{m}_20260928_student_only']) for m in ('0921', 'A', 'B', 'C')},
        'TB2.1 (88)': {**{m: allrate([f'{J}/tb21_6516_{m}_20260928']) for m in ('0921', 'A', 'B', 'C')},
                       'MIX': allrate([f'{J}/tb21_6516_MIX_s{s}_20260928' for s in range(4)])},
        'SWE-bench Verified (100)': {**{m: allrate([f'{J}/swe_6516_{m}_s{s}_20260928' for s in (0, 1)]) for m in ('0921', 'A', 'B')},
                                     'MIX': allrate([f'{J}/swe_6516_MIX_s{s}_20260928' for s in range(4)])},
        'OpenThoughts-TBLite (100)': {m: allrate([f'{J}/tblite_6516_{m}_s{s}_20260928' for s in (0, 1)]) for m in ('A', 'B')}}
models = [('0921', '09-21 (before SFT)', '#9b9a94'), ('A', 'A  relay, 09-21 turns masked', '#2a78d6'),
          ('B', 'B  Qwen alone', '#eb6834'), ('C', 'C  relay + autofix loss', '#1baf7a'),
          ('MIX', 'MIX  relay + Qwen-alone rows', '#eda100')]
INK, MUTED, SURF = '#0b0b0b', '#52514e', '#fcfcfb'
plt.rcParams.update({'font.size': 10, 'axes.edgecolor': '#d8d7d2', 'axes.labelcolor': MUTED, 'xtick.color': MUTED, 'ytick.color': MUTED})
fig, ax = plt.subplots(figsize=(11, 4.2), facecolor=SURF); ax.set_facecolor(SURF)
w = 0.16
seen = set()
for i, (bench, d) in enumerate(data.items()):
    for j, (k, lab, col) in enumerate(models):
        if k not in d or not d[k].get('n'): continue
        r = d[k]; x = i + (j - 2) * (w + 0.02)
        ax.bar(x, r['rate'] * 100, w, color=col, label=None if k in seen else lab, zorder=2); seen.add(k)
        ax.plot([x, x], [r['ci95'][0] * 100, r['ci95'][1] * 100], color=INK, lw=1, zorder=3)
        ax.text(x, r['ci95'][1] * 100 + 1, f"{r['rate']*100:.0f}%\n{r['passes']}/{r['n']}", ha='center', va='bottom', fontsize=7.5, color=INK, linespacing=1.1)
ax.set_xticks(range(len(data))); ax.set_xticklabels(list(data), color=INK)
ax.set_ylabel('tasks solved (%), one try per task'); ax.set_ylim(0, 52)
ax.grid(axis='y', color='#ecebe7', zorder=0); ax.spines[['top', 'right']].set_visible(False)
ax.legend(frameon=False, loc='upper left', fontsize=9)
fig.tight_layout(); fig.savefig(f'{OUT}/fig1_results.png', dpi=160, facecolor=SURF)
# fig 2: why pass rates stay low (TB2.1)
rej = {'0921': 89.4, 'A': 15.2, 'B': 20.3, 'C': 14.5}          # harness_errors.py, TB2.1 turns Terminus-2 rejects
out = {'0921': 55.5, 'A': 129.9, 'B': 119.7, 'C': 145.9}         # timeout_anatomy.py, median output tokens per TB2.1 task (k)
tmo = {'0921': 29, 'A': 60, 'B': 62, 'C': 68}                    # agent timeouts of 88
fig, axs = plt.subplots(1, 3, figsize=(10, 2.8), facecolor=SURF)
for ax, (title, d, fmt) in zip(axs, [('Replies the harness rejects (% of turns)', rej, '{:.0f}%'),
                                     ('Tokens written per task (thousands, median)', out, '{:.0f}k'),
                                     ('Hit the 30-min clock (of 88 tasks)', tmo, '{:.0f}')]):
    ax.set_facecolor(SURF)
    ys = list(range(4))[::-1]
    for y, (k, lab, col) in zip(ys, models[:4]):
        ax.barh(y, d[k], 0.62, color=col, zorder=2); ax.text(d[k], y, ' ' + fmt.format(d[k]), va='center', fontsize=9, color=INK)
    ax.set_yticks(ys); ax.set_yticklabels([m[1].split('  ')[0] if m[0] != '0921' else '09-21' for m in models[:4]], color=INK)
    ax.set_title(title, fontsize=9.5, color=INK, loc='left'); ax.set_xlim(0, max(d.values()) * 1.3)
    ax.spines[['top', 'right']].set_visible(False); ax.grid(axis='x', color='#ecebe7', zorder=0); ax.tick_params(axis='x', labelsize=8)
fig.tight_layout(); fig.savefig(f'{OUT}/fig2_why.png', dpi=160, facecolor=SURF)
print('ok')
