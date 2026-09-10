import json, glob, os, re
S="/e/fscratch/reformo/lee27/experiments/snowball_probe_val/snowball_probe_val/trace_jobs/eval_sessions/snowball_probe_val_eval_step0"
hits=[]
for rp in glob.glob(S+"/*/attempts/*/verifier/reward.txt"):
    if open(rp).read().strip().startswith("1"):
        att=os.path.dirname(os.path.dirname(rp))
        if os.path.exists(att+"/exception.txt"): hits.append(att)
print("reward.txt==1 with exception.txt:", len(hits))
for att in hits[:3]:
    r=json.load(open(att+"/result.json")); v=r.get("verifier_result") or {}
    print(os.path.basename(att.split("/attempts/")[0]), "| verifier reward:", (v.get("rewards") or {}).get("reward"), "| exception:", (r.get("exception_info") or {}).get("exception_type"), "| agent_result n_output:", (r.get("agent_result") or {}).get("n_output_tokens"))
# WANDB eval line: mean avg_score
L=open("/e/fscratch/reformo/lee27/experiments/snowball_probe_val/logs/snowball_probe_val_1629776.out").read().splitlines()
for l in L:
    if "WANDB_MIRROR kind=eval step=0" in l:
        d=json.loads(l.split("metrics=",1)[1]); av=[v for k,v in d.items() if k.endswith("/avg_score")]; p8=[v for k,v in d.items() if k.endswith("/pass_at_8")]
        print("wandb step0: n tasks", len(av), "mean avg_score %.4f"%(sum(av)/len(av)), "mean pass@8 %.4f"%(sum(p8)/len(p8)), "| other keys:", [k for k in d if not k.startswith("eval/_e_")][:10]); break
