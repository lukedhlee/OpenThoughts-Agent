import json, glob, csv, collections, statistics, os, sys, random
ARM=sys.argv[1] if len(sys.argv)>1 else "E"
import os as _os
S=_os.environ.get("P2O_SESSION","/e/fscratch/reformo/lee27/experiments/p2o6all_s0/p2o6all_s0/trace_jobs/eval_sessions/p2o6all_s0_eval_step0")
dev={r["task"]:r for r in csv.DictReader(open("/e/fscratch/reformo/lee27/experiments/p2o/dev120.tsv"),delimiter="\t")}
def find_reward(o,depth=0):
    if depth>6: return None
    if isinstance(o,dict):
        for k in ("reward","rewards"):
            if k in o:
                v=o[k]
                if isinstance(v,(int,float)): return float(v)
                if isinstance(v,dict):
                    for vv in v.values():
                        if isinstance(vv,(int,float)): return float(vv)
        for k,v in o.items():
            if k in ("config","trajectory","messages","steps"): continue
            r=find_reward(v,depth+1)
            if r is not None: return r
    return None
per=collections.defaultdict(list); ctx=collections.Counter(); ntr=collections.Counter()
for f in glob.glob(S+"/*/result.json"):
    try: d=json.load(open(f))
    except Exception: continue
    name=d.get("task_name") or os.path.basename(os.path.dirname(f)).split("__")[0]
    task,arm=name.rsplit("-p",1)
    e=d.get("exception_info") or {}
    et=(e.get("exception_type") if isinstance(e,dict) else "") or ""
    r=find_reward(d)
    if r is None: continue
    ntr[arm]+=1
    if "ContextLength" in et: ctx[arm]+=1
    per[(task,arm)].append(r)
rng=random.Random(0)
strata=["zero","hard","medium","success"]
def summarize(a):
    ds=[];rows=[]
    for (t,ar),v in per.items():
        if ar!=a or (t,"ctl") not in per: continue
        d_=statistics.mean(v)-statistics.mean(per[(t,"ctl")]); ds.append(d_); rows.append((dev.get(t,{}).get("stratum","?"),t,d_,v,per[(t,"ctl")]))
    n=len(ds); m=statistics.mean(ds)
    bs=sorted(sum(rng.choice(ds) for _ in range(n))/n for _ in range(4000)); lo,hi=bs[100],bs[3899]
    w=sum(d>0 for d in ds); l=sum(d<0 for d in ds); tie=n-w-l
    print(f"{a} vs ctl: pairs={n} mean_delta={m:+.3f} 95%CI=[{lo:+.3f},{hi:+.3f}] wins={w} losses={l} ties={tie}")
    print(f"  trials: {a}={ntr[a]} ctl={ntr['ctl']} | mean reward {a}={statistics.mean([x for (t,ar),v in per.items() if ar==a for x in v]):.3f} ctl={statistics.mean([x for (t,ar),v in per.items() if ar=='ctl' for x in v]):.3f} | ctx-death share {a}={ctx[a]/max(1,ntr[a]):.2f} ctl={ctx['ctl']/max(1,ntr['ctl']):.2f}")
    for s in strata:
        rs=[r for r in rows if r[0]==s]
        if not rs: continue
        dd=[r[2] for r in rs]; w=sum(d>0 for d in dd); l=sum(d<0 for d in dd)
        pa=statistics.mean([statistics.mean(r[3]) for r in rs]); pc=statistics.mean([statistics.mean(r[4]) for r in rs])
        print(f"  {s:<8} pairs={len(dd):>3} {a}={pa:.3f} ctl={pc:.3f} delta={statistics.mean(dd):+.3f} wins={w} losses={l}")
    unl=[r[1] for r in rows if r[0]=="zero" and max(r[3])>0 and max(r[4])==0]
    lost=[r[1] for r in rows if max(r[4])>0 and max(r[3])==0]
    print(f"  never-solved tasks passed under {a} but not ctl: {len(unl)} {unl[:8]}")
    print(f"  tasks passed under ctl but never under {a}: {len(lost)}")
summarize(ARM)
