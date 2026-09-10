import json, glob, os, collections
S="/e/fscratch/reformo/lee27/experiments/snowball_probe_val/snowball_probe_val/trace_jobs/eval_sessions/snowball_probe_val_eval_step0"
ds=sorted(glob.glob(S+"/*"))
print("task dirs:", len(ds), "example:", os.path.basename(ds[0]), os.path.basename(ds[-1]))
tasks=collections.Counter(os.path.basename(d).split("__")[0] for d in ds); print("unique tasks:", len(tasks), "attempt-dirs per task:", collections.Counter(tasks.values()))
a=sorted(glob.glob(ds[7]+"/attempts/*")); print("attempts under a task dir:", [os.path.basename(x) for x in a])
att=a[0]
for root,dirs,files in os.walk(att):
    lvl=root[len(att):].count(os.sep)
    if lvl<=2: print("  "*lvl+os.path.basename(root)+"/", [f+"(%d)"%os.path.getsize(os.path.join(root,f)) for f in files][:12])
r=json.load(open(att+"/result.json")); print("result.json keys:", list(r.keys()))
print("exception_info:", r.get("exception_info")); print("verifier_result keys:", list((r.get("verifier_result") or {}).keys()))
print("agent_result:", json.dumps(r.get("agent_result"))[:600])
c=json.load(open(att+"/config.json")); print("config.json:", json.dumps(c)[:1500])
t=json.load(open(att+"/agent/trajectory.json")); print("traj top keys:", list(t.keys())); 
for k in t:
    if k!="steps": print("  ",k,":",json.dumps(t[k])[:300])
st=t["steps"]; print("n steps", len(st), collections.Counter(s.get("source") for s in st))
for i,s in enumerate(st[:4]):
    print("--- step",i,"source",s.get("source"),"keys",list(s.keys()))
    for k,v in s.items():
        if k in("message","observation"): print("   ",k,": len",len(v if isinstance(v,str) else json.dumps(v)), repr((v if isinstance(v,str) else json.dumps(v))[:200]))
        elif k=="metrics": print("    metrics keys:", {kk:(len(vv) if isinstance(vv,list) else vv) for kk,vv in v.items()})
        else: print("   ",k,":",repr(v)[:200])
