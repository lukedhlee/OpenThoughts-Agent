#!/bin/bash
set -euo pipefail
C=/e/project1/transfernetx/lee27/code/snowball; E=/e/fscratch/reformo/lee27/experiments; T=/e/fscratch/reformo/lee27/tasks
W=/e/project1/transfernetx/lee27/code/OpenThoughts-Agent
python3 - <<'PY'
import glob, json, os, shutil, collections
TJ="/e/fscratch/reformo/lee27/experiments/snowball_probe_val_b60k/snowball_probe_val_b60k/trace_jobs"
src="/e/fscratch/reformo/lee27/tasks/r2egym-raw-v3-val"
scored=collections.Counter()
for p in glob.glob(f"{TJ}/eval_sessions/*/*/attempts/*/result.json"):
    try: d=json.load(open(p))
    except Exception: continue
    v=d.get("verifier_result"); r=(v.get("rewards") or {}).get("reward") if isinstance(v,dict) else None
    if r is None: continue
    t=(d.get("task_name") or os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(p))))).split("__")[0]
    scored[t]+=1
tasks=sorted(d for d in os.listdir(src) if d.startswith("r2egym-") and os.path.isdir(os.path.join(src,d)))
todo=[t for t in tasks if scored[t]<8]
print("tasks",len(tasks),"fully_sampled_so_far",len(tasks)-len(todo),"todo",len(todo))
K=4
for i in range(K):
    dst=f"/e/fscratch/reformo/lee27/tasks/r2egym-raw-v3-val-b60k-rest4-s{i}"
    if os.path.exists(dst): shutil.rmtree(dst)
    os.makedirs(dst)
    mine=todo[i::K]
    for t in mine: shutil.copytree(os.path.join(src,t), os.path.join(dst,t))
    print(dst,len(mine))
open("/e/fscratch/reformo/lee27/experiments/snowball_probe_val_b60k_rest4_todo.txt","w").write("\n".join(todo)+"\n")
PY
for i in 0 1 2 3; do
  SB=$(python3 $C/make_snowball_probe.py --name snowball_probe_val_b60k_p$i --val-dir $T/r2egym-raw-v3-val-b60k-rest4-s$i --k 8 --conc 128 --max-in 61440 --max-out 4096 --max-model-len 65536 --eval-timeout 1800 --wall 04:00:00 --nodes 8 --engines 4)
  grep -c "skip_special_tokens\|chat_template_content_format" $E/snowball_probe_val_b60k_p$i/configs/*.json || true
  cd $W && J=$(sbatch --parsable "$SB") && echo "p$i JOB=$J"
  tmux new -d -s val_watch_b60k_p$i "bash $C/val_watch.sh snowball_probe_val_b60k_p$i $J b60k_p$i"
done
tmux ls | grep b60k_p
