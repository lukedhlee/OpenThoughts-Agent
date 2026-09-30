"""TB2.1 diagnostics vs the pre-registered runs of the same model, paired on tasks both scored.
Also the outcome mix (context overflow / agent timeout / other fail / pass) and turns per trial."""
import json, glob, collections, os, sys, random
sys.path.insert(0, '/e/project1/transfernetx/lee27/code/ota-sft-v2/data/relay/pilot')
sys.path.insert(0, '/e/project1/transfernetx/lee27/code/ota-sft-v2/data/relay/sft')
from paired_eval import tb2_outcomes, paired, rate  # noqa
J = '/e/data1/mmlaion/lee27/experiments/tb2_jobs'
def mix(dirs):
    c = collections.Counter(); turns = []
    for d in dirs:
        for f in glob.glob(f'{d}/*/result.json'):
            r = json.load(open(f)); e = (r.get('exception_info') or {}).get('exception_type') or 'none'
            rew = ((r.get('verifier_result') or {}).get('rewards') or {}).get('reward')
            if rew is None: c['unscored'] += 1; continue
            c['pass' if rew >= 1 else {'ContextLengthExceededError': 'overflow', 'AgentTimeoutError': 'timeout'}.get(e, 'fail')] += 1
            n = ((r.get('agent_result') or {}).get('metadata') or {}).get('n_episodes')
            if n: turns.append(n)
    turns.sort(); return dict(c), (turns[len(turns)//2] if turns else None)
def outs(dirs):
    o = {}
    for d in dirs: o.update(tb2_outcomes(d)[0])
    return o
runs = {'A': [f'{J}/tb21_6516_A_20260928'], 'B': [f'{J}/tb21_6516_B_20260928'],
        'A+cut': [f'{J}/tb21diag_cutA_s0_20260928', f'{J}/tb21diag_cutA_s1_20260928'],
        'B+cut': [f'{J}/tb21diag_cutB_s0_20260928', f'{J}/tb21diag_cutB_s1_20260928'],
        'A@128k': [f'{J}/tb21diag_ctx128_s0_20260928', f'{J}/tb21diag_ctx128_s1_20260928']}
O = {k: outs(v) for k, v in runs.items()}
print('| run | pass rate [95 % CI] (n) | pass / overflow / timeout / other fail | median turns |'); print('|---|---|---|---|')
for k, v in runs.items():
    r = rate(O[k]); m, t = mix(v)
    print(f"| {k} | {r['rate']:.3f} [{r['ci95'][0]:.3f}, {r['ci95'][1]:.3f}] ({r['n']}) | {m.get('pass',0)} / {m.get('overflow',0)} / {m.get('timeout',0)} / {m.get('fail',0)} | {t} |")
print(); print('| paired (tasks both scored) | diff [95 % CI] | X-only / Y-only passes |'); print('|---|---|---|')
for x, y in (('A+cut', 'A'), ('B+cut', 'B'), ('A@128k', 'A'), ('A+cut', 'B+cut')):
    p = paired(O[x], O[y]); print(f"| {x} − {y} ({p['tasks']}) | {p['diff']:+.3f} [{p['ci95'][0]:+.3f}, {p['ci95'][1]:+.3f}] | {p['x_only']} / {p['y_only']} |")
