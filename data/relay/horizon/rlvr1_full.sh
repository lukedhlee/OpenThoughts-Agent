#!/usr/bin/env bash
# Cover the remaining CalibForge train pool, preserving the existing first 128 tasks.
set -euo pipefail
OTA=$(cd "$(dirname "$0")/../../.." && pwd)
X=/scratch/11584/$USER/experiments/relay_rlvr1
TREE=/scratch/11584/$USER/relay/calibforge_tree
mkdir -p "$X"
python3 - "$TREE/TASKS.txt" "$X/tasks128.txt" "$OTA/data/relay/pilot/calibforge_heldout300.txt" "$X/full_manifest.json" <<'PY'
import hashlib,json,random,sys
from pathlib import Path
tasks=Path(sys.argv[1]).read_text().splitlines()
random.Random(12).shuffle(tasks)
pilot=Path(sys.argv[2]).read_text().splitlines()
held=set(Path(sys.argv[3]).read_text().splitlines())
assert len(tasks)==2613 and len(set(tasks))==2613
assert pilot==tasks[:128] and not set(tasks)&held
plan=[(128,622),(750,621),(1371,621),(1992,621)]
covered=list(pilot)
shards=[]
for i,(offset,n) in enumerate(plan):
    part=tasks[offset:offset+n]; covered.extend(part)
    shards.append(dict(name=f'rlvr1_cf_full_s{i}_20261012',offset=offset,n_tasks=n,
                       task_sha256=hashlib.sha256(('\n'.join(part)+'\n').encode()).hexdigest(),
                       nodes=3,student_nodes=1,concurrency=40,attempts=2,cap_node_hours=18))
assert covered==tasks and len(set(covered))==2613
manifest=dict(model='open-athena/Grug-67B-A2B-Antidoom-RLVR1-Step12-2026.10.02',
              revision='88f4d8b8c2a6083b9d107836b2a3a028e2465a8c',tasks=2613,attempts=5226,
              heldout_overlap=0,reuse_run='rlvr1_cf128_20261012',shards=shards,
              new_nodes=12,new_cap_node_hours=72,ota_commit=None)
Path(sys.argv[4]).write_text(json.dumps(manifest,indent=2)+'\n')
print('FULL_COVERAGE_OK: 128 existing + 2485 new, 2613 unique tasks, 5226 attempts, no heldout overlap')
PY
for i in 0 1 2 3; do
    session=relay_rlvr1_full_s$i
    ! tmux has-session -t "$session" 2>/dev/null || { echo "$session already exists" >&2; exit 1; }
    case $i in 0) offset=128; count=622;; 1) offset=750; count=621;; 2) offset=1371; count=621;; 3) offset=1992; count=621;; esac
    name=rlvr1_cf_full_s${i}_20261012
    [ ! -e "$X/runs/$name" ] && [ ! -e "$X/$name.job" ]
    tmux new-session -d -s "$session" \
      "RUN_NAME=$name TASK_OFFSET=$offset TASK_COUNT=$count N_ATTEMPTS=2 CONC=40 SERVE_NODES=3 STUDENT_NODES=1 CAP_NODE_H=18 SERVE_WALL=06:00:00 bash $OTA/data/relay/horizon/rlvr1_pilot.sh > $X/$name.bootstrap.log 2>&1"
    echo "Started $session: $count tasks x2, three nodes, cap 18 node-hours"
    sleep 2
done
