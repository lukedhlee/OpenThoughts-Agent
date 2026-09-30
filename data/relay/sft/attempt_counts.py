"""Per run: tasks, total attempts (harbor's automatic retries on infra errors + recovery-pass reruns, each an attempts/NNN
dir), tasks attempted more than once, and the final outcome (passed / scored / no verdict)."""
import json, glob, os, collections
J = '/e/data1/mmlaion/lee27/experiments/tb2_jobs'; H = '/e/data1/mmlaion/lee27/experiments/relay_pilot_jobs'
runs = [('held-out', m, [f'{H}/heldout6516_{m}_20260928_student_only']) for m in ('0921', 'A', 'B', 'C')] + \
       [('TB2.1', m, [f'{J}/tb21_6516_{m}_20260928']) for m in ('0921', 'A', 'B', 'C')] + [('TB2.1', 'MIX', [f'{J}/tb21_6516_MIX_s{s}_20260928' for s in range(4)])] + \
       [('SWE', m, [f'{J}/swe_6516_{m}_s{s}_20260928' for s in (0, 1)]) for m in ('0921', 'A', 'B')] + [('SWE', 'MIX', [f'{J}/swe_6516_MIX_s{s}_20260928' for s in range(4)])] + \
       [('TB-Lite', m, [f'{J}/tblite_6516_{m}_s{s}_20260928' for s in (0, 1)]) for m in ('A', 'B')]
print('| bench | model | tasks | attempts | tasks tried >1× | passed | scored | no verdict |'); print('|---|---|---|---|---|---|---|---|')
for b, m, dirs in runs:
    t = att = multi = p = sc = nv = 0
    for d in dirs:
        for f in glob.glob(d + '/*/result.json'):
            t += 1
            n = len(glob.glob(os.path.dirname(f) + '/attempts/*/'))
            att += max(n, 1); multi += n > 1
            r = json.load(open(f)); rew = ((r.get('verifier_result') or {}).get('rewards') or {}).get('reward')
            if rew is None: nv += 1
            else: sc += 1; p += rew >= 1
    print(f'| {b} | {m} | {t} | {att} | {multi} | {p} | {sc} | {nv} |')
