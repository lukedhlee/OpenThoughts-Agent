import json, glob
S="/e/fscratch/reformo/lee27/experiments/snowball_probe_val/snowball_probe_val/trace_jobs/eval_sessions/snowball_probe_val_eval_step0"
p=sorted(glob.glob(S+"/*/attempts/000/agent/trajectory.json"))[7]
t=json.load(open(p)); st=t["steps"]; m=st[0]["message"]
print("first user msg chars", len(m)); print(m[:600]); print("....."); print(m[-1500:])
ag=[s for s in st if s.get("source")=="agent"]
print("=== agent step 0 message head:", repr(ag[0]["message"][:300]))
print("=== agent step 0 message tail:", repr(ag[0]["message"][-400:]))
o=json.loads(ag[0]["observation"]); print("=== obs keys", list(o.keys()), [list(r.keys()) for r in o["results"]]); print(repr(o["results"][0]["content"][:300]))
print("=== last agent step observation (never fed):", repr(ag[-1]["observation"][:300]))
print("=== prompt/completion per step:", [(s["metrics"]["prompt_tokens"], s["metrics"]["completion_tokens"]) for s in ag])
import os; att=os.path.dirname(os.path.dirname(p)); print(open(att+"/exception.txt").read()[:300]); print("reward.txt:", open(att+"/verifier/reward.txt").read() if os.path.exists(att+"/verifier/reward.txt") else None)
print("trial.log:", open(att+"/trial.log").read()[:300])
