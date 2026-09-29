"""Per model: how often Terminus-2 rejects or warns on a reply (parse errors / warnings in the next observation), and
how many trials end in a harness/infra exception. Held-out and TB2.1 runs."""
import json, glob, collections, sys
H = '/e/data1/mmlaion/lee27/experiments/relay_pilot_jobs/heldout6516_%s_20260928_student_only'
T = '/e/data1/mmlaion/lee27/experiments/tb2_jobs/tb21_6516_%s_20260928'
MODEL = {'ContextLengthExceededError', 'AgentTimeoutError', 'TurnCapExhaustedError', 'NonZeroAgentExitCodeError'}
for bench, pat in (('held-out', H), ('TB2.1', T)):
    print(f'== {bench}')
    for m in ['0921', 'A', 'B', 'C']:
        turns = err = warn = 0; trials = infra = 0; kinds = collections.Counter()
        for f in glob.glob(pat % m + '/*/result.json'):
            r = json.load(open(f)); trials += 1
            e = (r.get('exception_info') or {}).get('exception_type')
            if e and e not in MODEL: infra += 1; kinds[e] += 1
            a = r.get('trial_uri', '').replace('file://', '')
            try: t = json.load(open(a + '/agent/trajectory.json'))
            except Exception: continue
            for s in t['steps']:
                if not s.get('metrics'): continue
                turns += 1
                obs = json.dumps(s.get('observation') or {})
                if 'had parsing errors' in obs or 'ERROR:' in obs[:400]: err += 1
                elif 'had warnings' in obs: warn += 1
        print(f'  {m:>4}: parse errors {err}/{turns} turns ({err/max(1,turns):.1%}), warnings {warn/max(1,turns):.1%} | harness/infra exceptions {infra}/{trials} ({infra/max(1,trials):.1%}) {dict(kinds.most_common(3))}')
