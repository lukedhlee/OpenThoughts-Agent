"""Per-turn API latency + decode-rate implied from a probe trace tree (result.json + rollout_details)."""
import glob, json, sys, statistics
tj = sys.argv[1]
paths = glob.glob(f"{tj}/eval_sessions/*/*/attempts/*/result.json")
api=[]; out=[]; rate=[]; prompt=[]; trials=0; turns=[]
for p in paths:
    try: d=json.load(open(p))
    except Exception: continue
    ar=d.get("agent_result") or {}; md=ar.get("metadata") or {}
    t=md.get("api_request_times_msec") or []
    rd=(ar.get("rollout_details") or [None])[0] or {}
    comp=rd.get("completion_token_ids") or rd.get("response_ids") or rd.get("completion_ids") or []
    pr=rd.get("prompt_token_ids") or []
    if not t: continue
    trials+=1; turns.append(len(t))
    for i,ms in enumerate(t):
        api.append(ms/1000)
        if i < len(comp) and isinstance(comp[i],list):
            L=len(comp[i]); out.append(L)
            if ms>0: rate.append(L/(ms/1000))
        if i < len(pr) and isinstance(pr[i],list): prompt.append(len(pr[i]))
def q(x,f): 
    x=sorted(x); return x[int(f*(len(x)-1))]
print(f"trials={trials} turns_total={len(api)} turns/trial p50={statistics.median(turns)}")
for name,x in [("api s/turn",api),("completion tok/turn",out),("prompt tok/turn",prompt),("implied tok/s per turn (out/api)",rate)]:
    if x: print(f"{name:34s} p10={q(x,.1):8.1f} p50={q(x,.5):8.1f} p90={q(x,.9):8.1f} p99={q(x,.99):8.1f} max={max(x):8.1f} mean={statistics.mean(x):8.1f}")
if out:
    # bucket implied rate by completion length
    buckets=[(0,200),(200,500),(500,1000),(1000,2000),(2000,4000),(4000,99999)]
    pairs=[(L,ms) for L,ms in zip(out,api)]
    for lo,hi in buckets:
        sel=[(L,s) for L,s in pairs if lo<=L<hi]
        if sel: print(f"  completion {lo:5d}-{hi:5d}: n={len(sel):5d} api p50={q([s for _,s in sel],.5):6.1f}s  tok/s p50={q([L/s for L,s in sel if s>0],.5):6.1f}")
