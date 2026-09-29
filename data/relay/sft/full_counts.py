"""Every eval run of the relay SFT comparison: trials run, scored, passed, agent timeouts, context overflow, other fails,
infra losses (no verdict). One trial per task; the last attempt per trial dir counts (after any recovery pass)."""
import json, glob, collections
J = '/e/data1/mmlaion/lee27/experiments/tb2_jobs'; H = '/e/data1/mmlaion/lee27/experiments/relay_pilot_jobs'
runs = {'held-out 300': {m: [f'{H}/heldout6516_{m}_20260928_student_only'] for m in ('0921', 'A', 'B', 'C')},
        'TB2.1 (88)': {**{m: [f'{J}/tb21_6516_{m}_20260928'] for m in ('0921', 'A', 'B', 'C')},
                       'MIX': [f'{J}/tb21_6516_MIX_s{s}_20260928' for s in range(4)]},
        'SWE-bench (100)': {**{m: [f'{J}/swe_6516_{m}_s{s}_20260928' for s in (0, 1)] for m in ('0921', 'A', 'B')},
                            'MIX': [f'{J}/swe_6516_MIX_s{s}_20260928' for s in range(4)]},
        'TB-Lite (100)': {m: [f'{J}/tblite_6516_{m}_s{s}_20260928' for s in (0, 1)] for m in ('A', 'B')}}
print('| benchmark | model | trials | scored | passed | pass rate | hit 30-min clock | context full | other fail | infra (no verdict) |')
print('|---|---|---|---|---|---|---|---|---|---|')
for b, ms in runs.items():
    for m, dirs in ms.items():
        c = collections.Counter()
        for d in dirs:
            for f in glob.glob(d + '/*/result.json'):
                r = json.load(open(f)); c['trials'] += 1
                e = (r.get('exception_info') or {}).get('exception_type')
                rew = ((r.get('verifier_result') or {}).get('rewards') or {}).get('reward')
                if rew is None: c['infra'] += 1; continue
                c['scored'] += 1
                if rew >= 1: c['pass'] += 1
                elif e == 'AgentTimeoutError': c['timeout'] += 1
                elif e == 'ContextLengthExceededError': c['ctx'] += 1
                else: c['fail'] += 1
        print(f"| {b} | {m} | {c['trials']} | {c['scored']} | {c['pass']} | {c['pass']/max(1,c['scored']):.1%} | {c['timeout']} | {c['ctx']} | {c['fail']} | {c['infra']} |")
