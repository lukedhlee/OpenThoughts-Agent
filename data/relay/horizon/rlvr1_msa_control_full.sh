#!/usr/bin/env bash
# Complete the MSA2 experiment with fresh teacher-only controls on the same train pool.
set -euo pipefail
OTA=$(cd "$(dirname "$0")/../../.." && pwd)
X=/scratch/11584/$USER/experiments/relay_rlvr1_msa_control
TREE=/scratch/11584/$USER/relay/calibforge_tree
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
    shards.append(dict(name=f'rlvr1_msa_control_s{i}_20261012',offset=offset,n_tasks=n,
                       task_sha256=hashlib.sha256(('\n'.join(part)+'\n').encode()).hexdigest(),
                       nodes=2,student_nodes=0,concurrency=40,attempts=2,cap_node_hours=12))
assert coverage==tasks
m=dict(base='open-athena/Grug-67B-A2B-Antidoom-RLVR1-Step12-2026.10.02',
       revision='88f4d8b8c2a6083b9d107836b2a3a028e2465a8c',teacher='Qwen/Qwen3.8-27B',
       harness='mini-swe-agent-host / mini-swe-agent 2.4.6 tool mode',arm='control',
       tasks=2613,attempts=5226,heldout_overlap=0,shards=shards,nodes=8,cap_node_hours=48,
       ota_commit=subprocess.check_output(['git','-C',sys.argv[4],'rev-parse','HEAD'],text=True).strip(),
       relay_experiment='/scratch/11584/lukedhlee/experiments/relay_rlvr1',
       matching='Same (task, outcome) multiset and count, at most two per task, identical QA and submit filter',
       primary_sft_arms=['vanilla_teacher_only','relay_teacher_supervision'],
       training=dict(init='init-rlvr1-step12-native0',lr=0.0003,passes=3,batch=16,context=65536,seed=0,
                     freeze_router_bias=True,mask_student_turns=True,fixed_no_refeed_windows=True),
       evaluation=dict(models=['untuned_base','vanilla','relay'],harness='MSA2 no-refeed tool mode',
                       benchmarks=['TB2.1','SWE-Verified-r100','TBLite'],runs_per_model_benchmark=3,
                       context=65536,output=16384,task_budget_seconds=1800))
Path(sys.argv[3]).write_text(json.dumps(m,indent=2)+'\n')
print('CONTROL_FULL_COVERAGE_OK: same 2613 train tasks as relay, 5226 attempts, heldout overlap 0')
PY
for i in 0 1 2 3; do
    session=relay_rlvr1_msa_control_s$i
    ! tmux has-session -t "$session" 2>/dev/null || { echo "$session already exists" >&2; exit 1; }
    case $i in 0) offset=0; count=654;; 1) offset=654; count=653;; 2) offset=1307; count=653;; 3) offset=1960; count=653;; esac
    name=rlvr1_msa_control_s${i}_20261012
    [ ! -e "$X/runs/$name" ] && [ ! -e "$X/$name.job" ]
    tmux new-session -d -s "$session" \
      "EXPERIMENT_ROOT=$X AGENT_MODE=msa ARMS=control RUN_NAME=$name TASK_OFFSET=$offset TASK_COUNT=$count N_ATTEMPTS=2 CONC=40 SERVE_NODES=2 STUDENT_NODES=0 CAP_NODE_H=12 SERVE_WALL=06:00:00 bash $OTA/data/relay/horizon/rlvr1_pilot.sh > $X/$name.bootstrap.log 2>&1"
    echo "Started $session: $count tasks x2, teacher-only, two nodes, cap 12 node-hours"
    sleep 2
done
