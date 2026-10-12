#!/usr/bin/env bash
# Fresh guided CalibForge relay pilot from the pinned Antidoom RLVR1 step-12 export.
set -euo pipefail
OTA=$(cd "$(dirname "$0")/../../.." && pwd)
SB=$HOME/snowball
S=/scratch/11584/$USER
X=$S/experiments/relay_rlvr1
PY=$SB/envs/snowball/bin/python
MODEL=$S/models/grug-antidoom-rlvr1-step12
REV=88f4d8b8c2a6083b9d107836b2a3a028e2465a8c
NAME=${RUN_NAME:-rlvr1_cf_pilot_20261012}
mkdir -p "$X"
exec > >(tee -a "$X/$NAME.launch.log") 2>&1
trap 'echo "FAILED at line $LINENO" > "$X/$NAME.status"' ERR
echo 'Downloading pinned RLVR1 checkpoint' > "$X/$NAME.status"
# The initial download may already be running; do not compete for the same cache locks.
while tmux has-session -t relay_rlvr1_download 2>/dev/null; do sleep 20; done
"$SB/envs/snowball/bin/hf" download open-athena/Grug-67B-A2B-Antidoom-RLVR1-Step12-2026.10.02 \
  --revision "$REV" --local-dir "$MODEL"
"$PY" - "$MODEL" "$S/models/grug-datakit-sft-20260921" <<'PY'
import json,sys
from pathlib import Path
from tokenizers import Tokenizer
m,b=map(Path,sys.argv[1:])
cfg=json.loads((m/'config.json').read_text())
assert cfg['model_type']=='grug_moe' and cfg['qk_mult']==1.75
assert (m/'chat_template.jinja').is_file()
v=Tokenizer.from_file(str(m/'tokenizer.json')).get_vocab()
old=Tokenizer.from_file(str(b/'tokenizer.json')).get_vocab()
for token in ['<|begin_of_text|>','<|eot_id|>','<|start_header_id|>','<|end_header_id|>',
              '<|start_think|>','<|end_think|>']:
    assert token in v and v[token]==old[token], (token,v.get(token),old.get(token))
assert v == old, 'Tokenizer vocabulary differs from the validated relay student'
idx=json.loads((m/'model.safetensors.index.json').read_text())
shards=set(idx['weight_map'].values())
assert all((m/s).is_file() and (m/s).stat().st_size>0 for s in shards)
print('MODEL_PREFLIGHT_OK',len(shards),'shards, compatible special tokens, native chat template')
PY
export PYTHONPATH=$SB/harbor-msa-relay/src
export RES_OPTIONS='timeout:1 attempts:2'
export KEYF=$HOME/.config/otagent/daytona_eval.env
export TREE=$S/relay/calibforge_tree
echo 'Restoring missing CalibForge snapshots (create only)' > "$X/$NAME.status"
"$PY" "$OTA/data/relay/horizon/rlvr1_snapshots.py" "$TREE" "$KEYF" "$X/snapshot_logs"
# Keep the original full staged tree (router task metadata), selecting a deterministic subset.
"$PY" - "$TREE/TASKS.txt" "$OTA/data/relay/pilot/calibforge_heldout300.txt" "$X/tasks32.txt" <<'PY'
import random,sys
from pathlib import Path
tasks=Path(sys.argv[1]).read_text().splitlines()
held=set(Path(sys.argv[2]).read_text().splitlines())
assert len(tasks)==len(set(tasks)) and not set(tasks)&held
random.Random(12).shuffle(tasks)
Path(sys.argv[3]).write_text('\n'.join(tasks[:32])+'\n')
print('TASK_PREFLIGHT_OK',len(tasks),'train tasks, 32 selected, heldout disjoint')
PY
export STUDENT_MODEL=$MODEL
export RELAY_PILOT_DIR=$OTA/data/relay/pilot
export RELAY_EXP_DIR=$X E=$X JOBS_ROOT=$X/jobs
export AGENT=msa HARBOR_SRC=$SB/harbor-msa-relay/src HARBOR_SHA=96fd911b
export MSA_DIR=$SB/envs/msa-2.4.6
export NODES=2 CAP_NODE_H=6 IN_SERVE=1
export TASK_LIST=$X/tasks32.txt N_ATTEMPTS=1 CONC=8
export CTX_BUDGET=40000 TIME_TAKEOVER=0.6 ASSIST=1 ASSIST_CTX=20000,35000
export ASSIST_ERR=4 ASSIST_GAP=3 ASSIST_MAX=6
export TEACHER_HINT_TEXT='Work economically: the context window is limited and the episode ends when it is full. Keep command output short (pipe through head or tail, use grep -n and then sed -n on a line range; never cat binary or very large files). Change files with targeted edits (sed, or a short python patch) instead of rewriting whole files. Commands time out after 30 seconds, so run long jobs in the background with nohup and poll their output. Once the requirements are met and checked, submit rather than re-checking.'
export GATE_LAT=120 GATE_MIN=15 SHUFFLE_SEED=list STAGGER_SEC=0
export EARLY_MIN=10 MIN_EARLY_TURNS=4 STOP_AFTER=16 STOP_EVERY=120 STOP_HERR=0.25 STOP_OVF=1.01
export TAKEOVER_MIN=0 TAKEOVER_MAX=1 TAKEOVER_AFTER=16
export TUNNEL_PORTS=18180,18181 BRIDGE_PORT=18947 PORT0=22750
echo 'Submitting two-node pilot; hard ceiling 6 node-hours' > "$X/$NAME.status"
JOB=${PILOT_JOB:-}
if [ -z "$JOB" ]; then
  JOB=$(bash "$OTA/data/relay/horizon/serve_submit.sh" 2 1 03:00:00 "$NAME")
else
  [ -n "$(squeue -h -j "$JOB" -u "$USER")" ]
fi
[[ $JOB =~ ^[0-9]+$ ]]
echo "$JOB" > "$X/$NAME.job"
if ! bash "$OTA/data/relay/horizon/launch_driver.sh" "$JOB" "$NAME"; then
  scancel "$JOB"
  exit 1
fi
echo "Pilot submitted: job $JOB; 32 tasks; max 6 node-hours; run $X/runs/$NAME" | tee "$X/$NAME.status"
# Keep this tmux session alive while the driver and reverse tunnels run. Horizon may
# reap a login command's descendants when its final shell exits.
while [ -n "$(squeue -h -j "$JOB")" ]; do sleep 30; done
"$PY" "$OTA/data/relay/msa/gate.py" "$X/runs/$NAME" > "$X/$NAME.gate.json"
echo "Pilot finished: job $JOB; gate $X/$NAME.gate.json" > "$X/$NAME.status"
