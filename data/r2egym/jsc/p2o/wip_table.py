import json, glob, csv, collections, statistics, os, sys
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
shown=False
per=collections.defaultdict(list)   # (task,arm) -> rewards
exc=collections.Counter(); n_noreward=0
for f in glob.glob(S+"/*/result.json"):
    try: d=json.load(open(f))
    except Exception: continue
    name=d.get("task_name") or os.path.basename(os.path.dirname(f)).split("__")[0]
    task,arm=name.rsplit("-p",1)
    if not shown:
        print("top keys:",[k for k in d if k!="config"][:20]); shown=True
    r=find_reward(d)
    e=d.get("exception_info") or d.get("exception") or {}
    et=(e.get("exception_type") if isinstance(e,dict) else None) or ""
    if et: exc[(arm,et.split(".")[-1])]+=1
    if r is None: n_noreward+=1; continue
    per[(task,arm)].append(r)
arms=["ctl","A","B","C","D","E"]
print("trials with reward:",sum(len(v) for v in per.values()),"| no reward:",n_noreward)
print("exceptions:",dict(sorted(exc.items())))
strata=["zero","hard","medium","success"]
print("\nARM   n_trials  mean_reward  n_tasks   " + "  ".join(f"{s:>7}" for s in strata))
for a in arms:
    tr=[x for (t,ar),v in per.items() if ar==a for x in v]
    tasks=[t for (t,ar) in per if ar==a]
    by={s:[] for s in strata}
    for (t,ar),v in per.items():
        if ar!=a: continue
        s=dev.get(t,{}).get("stratum","?")
        if s in by: by[s].append(statistics.mean(v))
    print(f"{a:<5} {len(tr):>8} {statistics.mean(tr) if tr else float('nan'):>11.3f} {len(tasks):>8}   " + "  ".join(f"{(statistics.mean(by[s]) if by[s] else float('nan')):>7.3f}" for s in strata))
print("\nPAIRED vs ctl (tasks scored in both arms; per-task mean reward):")
print("ARM   n_pairs  delta   wins  losses   " + "  ".join(f"{s:>7}" for s in strata))
for a in arms[1:]:
    ds=[]; w=l=0; bys={s:[] for s in strata}
    for (t,ar),v in per.items():
        if ar!=a or (t,"ctl") not in per: continue
        d_=statistics.mean(v)-statistics.mean(per[(t,"ctl")]); ds.append(d_)
        w+=d_>0; l+=d_<0
        s=dev.get(t,{}).get("stratum","?")
        if s in bys: bys[s].append(d_)
    print(f"{a:<5} {len(ds):>7} {statistics.mean(ds) if ds else float('nan'):>+7.3f} {w:>5} {l:>7}   " + "  ".join(f"{(statistics.mean(bys[s]) if bys[s] else float('nan')):>+7.3f}" for s in strata))
