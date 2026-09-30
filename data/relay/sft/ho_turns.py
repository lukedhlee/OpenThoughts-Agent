import json,glob,collections,statistics as st
R='/e/data1/mmlaion/lee27/experiments/relay_pilot_jobs/heldout6516_%s_20260928_student_only'
def q(xs,p): xs=sorted(xs); return xs[int(p*(len(xs)-1))] if xs else None
for m in ['0921','A','B','C']:
    rows=[]
    for f in glob.glob(R%m+'/*/result.json'):
        r=json.load(open(f)); a=r.get('trial_uri','').replace('file://','')
        e=(r.get('exception_info') or {}).get('exception_type') or 'none'
        rew=((r.get('verifier_result') or {}).get('rewards') or {}).get('reward')
        try: t=json.load(open(a+'/agent/trajectory.json'))
        except Exception: continue
        steps=[s for s in t['steps'] if s.get('metrics')]
        comp=[s['metrics'].get('completion_tokens') or 0 for s in steps]
        prm=[s['metrics'].get('prompt_tokens') or 0 for s in steps]
        # history growth after a turn vs that turn's completion: is thinking re-fed?
        grow=[(prm[i+1]-prm[i], comp[i]) for i in range(len(prm)-1) if comp[i]>4000]
        times=((r.get('agent_result') or {}).get('metadata') or {}).get('api_request_times_msec') or []
        ae=r.get('agent_execution') or {}
        import datetime as dt
        try: wall=(dt.datetime.fromisoformat(ae['finished_at'].replace('Z','+00:00'))-dt.datetime.fromisoformat(ae['started_at'].replace('Z','+00:00'))).total_seconds()
        except Exception: wall=None
        rows.append(dict(e=e,rew=rew,turns=len(steps),comp=sum(comp),maxc=max(comp or [0]),n16=sum(c>=16000 for c in comp),n8=sum(c>=8000 for c in comp),last=max(prm or [0]),grow=grow,api=sum(times)/1000,wall=wall,
                         thinkfrac=sum(len(s['message'].split('<|end_think|>')[0]) for s in steps if '<|end_think|>' in s['message'])/max(1,sum(len(s['message']) for s in steps))))
    n=len(rows)
    print(f"== {m} n={n}")
    print(f"  turns median {q([r['turns'] for r in rows],.5)} p90 {q([r['turns'] for r in rows],.9)} | output tok/trial median {q([r['comp'] for r in rows],.5)} | max turn output median {q([r['maxc'] for r in rows],.5)} p90 {q([r['maxc'] for r in rows],.9)}")
    print(f"  trials with a turn >=8k out: {sum(r['n8']>0 for r in rows)/n:.2f}  >=16k: {sum(r['n16']>0 for r in rows)/n:.2f} | think share of output chars median {q([r['thinkfrac'] for r in rows],.5):.2f}")
    g=[x for r in rows for x in r['grow']]; 
    if g: print(f"  after a >4k-token turn, next prompt grows by median {q([a for a,b in g],.5)} for turn size median {q([b for a,b in g],.5)} (n={len(g)})")
    api=[r['api']/r['wall'] for r in rows if r['wall']]; print(f"  share of agent wall time spent waiting on the model: median {q(api,.5):.2f}")
    for k in ['ContextLengthExceededError','AgentTimeoutError']:
        s=[r for r in rows if r['e']==k]
        if s: print(f"  {k}: n={len(s)} turns median {q([r['turns'] for r in s],.5)}, max-turn output median {q([r['maxc'] for r in s],.5)}, has >=16k turn {sum(r['n16']>0 for r in s)/len(s):.2f}, model-wait share {q([r['api']/r['wall'] for r in s if r['wall']],.5):.2f}")
print("---- decode speed and passes")
for m in ['0921','A','B','C']:
    sp=[];P=[];over=0;nt=0
    for f in glob.glob(R%m+'/*/result.json'):
        r=json.load(open(f)); a=r.get('trial_uri','').replace('file://','')
        try: t=json.load(open(a+'/agent/trajectory.json'))
        except Exception: continue
        times=((r.get('agent_result') or {}).get('metadata') or {}).get('api_request_times_msec') or []
        steps=[s for s in t['steps'] if s.get('metrics')]
        for s,ms in zip(steps,times):
            c=s['metrics'].get('completion_tokens') or 0; nt+=1; over+=c>16384
            if c>2000 and ms>0: sp.append(c/(ms/1000))
        rew=((r.get('verifier_result') or {}).get('rewards') or {}).get('reward')
        if (rew or 0)>=1: P.append((len(steps),sum(s['metrics'].get('completion_tokens') or 0 for s in steps),max([s['metrics'].get('prompt_tokens') or 0 for s in steps] or [0])))
    print(f"{m}: per-stream decode tok/s on >2k turns median {q(sp,.5):.0f} | turns over 16,384 out: {over}/{nt} | passes n={len(P)} turns median {q([p[0] for p in P],.5)} output median {q([p[1] for p in P],.5)} peak prompt median {q([p[2] for p in P],.5)}")
