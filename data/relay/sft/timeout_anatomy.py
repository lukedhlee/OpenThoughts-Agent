"""TB2.1 runs: where an agent's 1,800 s goes. Per trial: model-wait share, output tokens, turns, per-stream decode speed,
thinking share; split by outcome."""
import json, glob, datetime as dt, collections
J = '/e/data1/mmlaion/lee27/experiments/tb2_jobs/tb21_6516_%s_20260928'
def q(xs, p): xs = sorted(x for x in xs if x is not None); return xs[int(p * (len(xs) - 1))] if xs else None
def ts(s): return dt.datetime.fromisoformat(s.replace('Z', '+00:00'))
for m in ['0921', 'A', 'B', 'C']:
    rows = []
    for f in glob.glob(J % m + '/*/result.json'):
        r = json.load(open(f)); a = r.get('trial_uri', '').replace('file://', '')
        ae = r.get('agent_execution') or {}
        try: wall = (ts(ae['finished_at']) - ts(ae['started_at'])).total_seconds()
        except Exception: continue
        md = (r.get('agent_result') or {}).get('metadata') or {}
        times = md.get('api_request_times_msec') or []
        try: t = json.load(open(a + '/agent/trajectory.json'))
        except Exception: t = {'steps': []}
        steps = [s for s in t['steps'] if s.get('metrics')]
        comp = [s['metrics'].get('completion_tokens') or 0 for s in steps]
        think = sum(len(s['message'].split('<|end_think|>')[0]) for s in steps if '<|end_think|>' in s['message'])
        allc = sum(len(s['message']) for s in steps)
        spd = [c / (ms / 1000) for c, ms in zip(comp, times) if c > 1000 and ms > 0]
        e = (r.get('exception_info') or {}).get('exception_type') or 'none'
        rows.append(dict(e=e, wall=wall, wait=sum(times) / 1000 / wall if wall else None, out=(r.get('agent_result') or {}).get('n_output_tokens'),
                         turns=len(times), spd=q(spd, .5), think=think / allc if allc else None))
    to = [x for x in rows if x['e'] == 'AgentTimeoutError']
    print(f"== {m}: {len(rows)} trials, timeouts {len(to)}")
    for lab, s in (('all', rows), ('timeouts', to)):
        print(f"   {lab}: model-wait share {q([x['wait'] for x in s],.5):.2f} | output tok/trial {q([x['out'] for x in s],.5)} | turns {q([x['turns'] for x in s],.5)} | decode tok/s/stream {q([x['spd'] for x in s],.5):.0f} | think share {q([x['think'] for x in s],.5):.2f}")
