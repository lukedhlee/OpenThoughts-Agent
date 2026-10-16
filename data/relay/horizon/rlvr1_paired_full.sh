#!/usr/bin/env bash
# Gist-style Terminus-2 teacher-only vs relay comparison, from RLVR1 step 12.
set -euo pipefail
OTA=$(cd "$(dirname "$0")/../../.." && pwd)
X=$SCRATCH/experiments/relay_rlvr1_t2
TREE=$SCRATCH/relay/calibforge_tree
mkdir -p "$X"
python3 - "$TREE/TASKS.txt" "$OTA/data/relay/pilot/calibforge_heldout300.txt" "$X/manifest.json" "$OTA" <<'PY'
import hashlib,json,random,subprocess,sys
from pathlib import Path
tasks=Path(sys.argv[1]).read_text().splitlines()
held=set(Path(sys.argv[2]).read_text().splitlines())
assert len(tasks)==len(set(tasks))==2613 and not set(tasks)&held
random.Random(12).shuffle(tasks)
shards=[]; coverage=[]
for i,(offset,n) in enumerate([(0,654),(654,653),(1307,653),(1960,653)]):
    part=tasks[offset:offset+n]; coverage.extend(part)
    shards.append(dict(name=f'rlvr1_t2_pair_s{i}_20261012',offset=offset,n_tasks=n,
                       task_sha256=hashlib.sha256(('\n'.join(part)+'\n').encode()).hexdigest(),
                       nodes=3,student_nodes=1,concurrency_per_arm=30,attempts_per_arm=2,cap_node_hours=24))
assert coverage==tasks
m=dict(base='open-athena/Grug-67B-A2B-Antidoom-RLVR1-Step12-2026.10.02',
       revision='88f4d8b8c2a6083b9d107836b2a3a028e2465a8c',teacher='Qwen/Qwen3.8-27B',
       harness='terminus-2',arms=['control','relay_repair'],tasks=2613,
       attempts_per_arm=5226,total_attempts=10452,heldout_overlap=0,shards=shards,
       nodes=12,cap_node_hours=96,
       ota_commit=subprocess.check_output(['git','-C',sys.argv[4],'rev-parse','HEAD'],text=True).strip(),
       selection='Identical QA; exact (task, pass/fail) multiset; at most 2 rows/task; retain eligible verified failures',
       training=dict(lr=0.0003,passes=3,sequence_length=65536,batch=16,seed=0,freeze_router_bias=True),
       evaluation=dict(models=['base','vanilla','relay'],benchmarks=['TB2.1','SWE-Verified-r100','TBLite'],
                       runs_per_model_benchmark=3,context=65536,output=16384,task_budget_seconds=1800))
Path(sys.argv[3]).write_text(json.dumps(m,indent=2)+'\n')
print('PAIRED_FULL_COVERAGE_OK: 2613 unique tasks; 5226 attempts per arm; heldout overlap 0')
PY
for i in 0 1 2 3; do
    session=relay_rlvr1_t2_s$i
    ! tmux has-session -t "$session" 2>/dev/null || { echo "$session already exists" >&2; exit 1; }
    case $i in 0) offset=0; count=654;; 1) offset=654; count=653;; 2) offset=1307; count=653;; 3) offset=1960; count=653;; esac
    name=rlvr1_t2_pair_s${i}_20261012
    [ ! -e "$X/runs/$name" ] && [ ! -e "$X/$name.job" ]
    tmux new-session -d -s "$session" \
      "EXPERIMENT_ROOT=$X AGENT_MODE=terminus2 ARMS='control relay_repair' RUN_NAME=$name TASK_OFFSET=$offset TASK_COUNT=$count N_ATTEMPTS=2 CONC=30 SERVE_NODES=3 STUDENT_NODES=1 CAP_NODE_H=24 SERVE_WALL=08:00:00 MAX_OUTPUT=16384 bash $OTA/data/relay/horizon/rlvr1_pilot.sh > $X/$name.bootstrap.log 2>&1"
    echo "Started $session: $count tasks x2 per arm, three nodes, cap 24 node-hours"
    sleep 2
done
