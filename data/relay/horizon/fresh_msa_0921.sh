#!/usr/bin/env bash
# Fresh paired MSA2 tool-mode data on tasks absent from the old matched pair.
set -euo pipefail
OTA=$(cd "$(dirname "$0")/../../.." && pwd)
SB=$HOME/snowball
S=/scratch/11584/$USER
R=${FRESH_MSA_ROOT:-$S/experiments/msa2-fresh-0921-20261023}
NAME=msa2_fresh_cf64_20261023
PY=$SB/envs/snowball/bin/python
HB=$R/code/harbor
MR=$S/experiments/cf2-mixed-sft-20261023/code/marin
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export CC=gcc CXX=g++
unset LD_PRELOAD
cd "$OTA"
export PYTHONPATH=$HB/src:$OTA
export TREE=$S/relay/calibforge_tree
if [ "${1:-driver}" = prepare ]; then
  : "${SLURM_JOB_ID:?selection must run in a compute allocation}"
  exec "$PY" "$OTA/data/relay/horizon/prepare_fresh_msa.py" --root "$R" --tree "$TREE" \
    --heldout "$OTA/data/relay/pilot/calibforge_heldout300.txt" \
    --old-selection "$S/experiments/msa0921_clean_pair/matched_sft/selected"
fi
mkdir -p "$R/bin" "$R/logs"
exec 9>"$R/controller.lock"
flock -n 9 || { echo "An existing controller owns $R" >&2; exit 1; }
trap 'echo "FAILED line $LINENO; inspect controller.out and logs" > "$R/STATUS"' ERR
test -z "$(git status --porcelain --untracked-files=no)"
test "$(git -C "$HB" rev-parse HEAD)" = faa9a756a4a3ac1d8063feaad450f1d4574acf48
test -f "$R/runtime_gate.json"
"$PY" -B - "$R/runtime_gate.json" "$(git rev-parse HEAD)" <<'PY'
import json,sys
x=json.load(open(sys.argv[1]));assert x['passed'] and x['ota_commit']==sys.argv[2]
assert x['snapshot_operations']=='none' and x['context_fit_unit_gate']
PY
ln -sfn "$MR/experiments/june_tpu_67b_a2b/moe/replay_sbatch.py" "$R/bin/sbatch"
export FP32_REPLAY_ROOT=$R PATH=$R/bin:$PATH
submit() {
  local count=$1 output rc nodes; shift
  while :; do
    nodes=$(squeue -u "$USER" -h -o %D | awk '{n+=$1} END {print n+0}')
    if [ $((nodes + count)) -gt 100 ]; then sleep 30; continue; fi
    set +e; output=$(sbatch "$@" 2>&1); rc=$?; set -e
    if [ "$rc" = 75 ]; then sleep 30; continue; fi
    [ "$rc" = 0 ] || { echo "$output" >&2; return "$rc"; }
    echo "$output" | sed -nE 's/Submitted batch job ([0-9]+).*/\1/p' | tail -1
    return
  done
}
echo PREPARING_64_FRESH_TASKS > "$R/STATUS"
if [ ! -f "$R/manifest.json" ]; then
  if [ ! -f "$R/prepare.job" ]; then
    printf -v command 'bash %q prepare' "$OTA/data/relay/horizon/fresh_msa_0921.sh"
    job=$(submit 1 -A CCR24067 -p debug -N 1 -t 00:20:00 --ntasks-per-node=1 --cpus-per-task=4 \
      -J msa2_fresh_select -o "$R/logs/prepare.%j.log" --export=ALL --wrap="$command")
    [ -n "$job" ]; echo "$job" > "$R/prepare.job"
  fi
  job=$(cat "$R/prepare.job")
  while :; do
    state=$(sacct -j "$job" -X -n -P --format=State,ExitCode | head -1)
    [ "$state" != 'COMPLETED|0:0' ] || break
    squeue -u "$USER" -h -o %i | grep -qx "$job" || { echo "Selection failed: $state" >&2; exit 1; }
    sleep 30
  done
  grep -q FRESH_MSA_SELECTION_OK "$R/logs/prepare.$job.log"
fi
"$PY" "$OTA/data/relay/horizon/validate_fresh_msa.py" --root "$R" --harbor "$HB" --ota "$OTA"
export KEYF=$HOME/.config/otagent/daytona_eval.env
"$PY" "$OTA/data/calibforge/daytona/snapshot_census.py" --key-file "$KEYF" --tree "$TREE" > "$R/snapshot_census.txt"
"$PY" -B - "$R/snapshot_census.txt" <<'PY'
import re,sys
s=open(sys.argv[1]).read();m=re.search(r'tree hashes present in the org: (\d+) of (\d+)',s)
owned=[x.split(' | ') for x in s.splitlines() if x.startswith('owned ')]
assert m and m.group(1)==m.group(2)=='3' and owned and all(x[2].lower()=='active' for x in owned)
print('THREE_EXISTING_SNAPSHOTS_ACTIVE')
PY
export STUDENT_MODEL=$S/models/grug-datakit-sft-20260921
export RELAY_PILOT_DIR=$OTA/data/relay/pilot RELAY_EXP_DIR=$R E=$R JOBS_ROOT=$R/jobs
export AGENT=msa HARBOR_SRC=$HB/src HARBOR_SHA=faa9a756 MSA_DIR=$SB/envs/msa-2.4.6
export MSA_JOB_TEMPLATE=relay_msa_fresh_0921.yaml RELAY_EXTRA_PYTHONPATH=$OTA MSA_INTERLEAVED=true
export ARMS="control relay_repair" N_ATTEMPTS=1 CONC=8 NTASKS=2613 TASK_LIST=$R/tasks.txt
export MAX_INPUT=65536 MAX_OUTPUT=16384 TEACHER_MAX_TOKENS=16384
export CLOCK=repair CTX_BUDGET=40000 ROW_MAX=65536 ROW_RESERVE=8192
export ASSIST=1 ASSIST_CTX=20000,35000 ASSIST_ERR=4 ASSIST_GAP=3 ASSIST_MAX=6 TIME_TAKEOVER=0.6
export TEACHER_GUARD=1 TEACHER_RESAMPLES=2 VERIFY_NOTE=1
export TEACHER_HINT_TEXT='Work economically: the context window is limited and the episode ends when it is full. Keep command output short (pipe through head or tail, use grep -n and then sed -n on a line range; never cat binary or very large files). Change files with targeted edits (sed, or a short python patch) instead of rewriting whole files. Commands time out after 30 seconds, so run long jobs in the background with nohup and poll their output. Once the requirements are met and checked, submit rather than re-checking.'
export SHUFFLE_SEED=list STAGGER_SEC=0 EARLY_MIN=10 MIN_EARLY_TURNS=4
export GATE_MIN=15 GATE_LAT=120 STOP_AFTER=16 STOP_EVERY=120 STOP_HERR=0.25 STOP_OVF=1.01
export TAKEOVER_MIN=0 TAKEOVER_MAX=1 TAKEOVER_AFTER=16
export NODES=3 CAP_NODE_H=24 IN_SERVE=1
export TUNNEL_PORTS=18280,18281 BRIDGE_PORT=19147 PORT0=23750
echo SUBMITTING_3_NODE_MSA2_SERVE > "$R/STATUS"
if [ ! -f "$R/serve.job" ]; then
  while :; do
    while [ "$(squeue -u "$USER" -h -o %D | awk '{n+=$1} END {print n+0}')" -gt 97 ]; do sleep 30; done
    set +e
    job=$(bash "$OTA/data/relay/horizon/serve_submit.sh" 3 1 08:00:00 msa2_fresh_0921)
    rc=$?
    set -e
    if [ "$rc" = 75 ]; then sleep 30; continue; fi
    [ "$rc" = 0 ] && [[ "$job" =~ ^[0-9]+$ ]]
    break
  done
  echo "$job" > "$R/serve.job"
fi
job=$(cat "$R/serve.job")
test ! -e "$R/runs/$NAME"
test ! -e "$R/driver.launched"
bash "$OTA/data/relay/horizon/launch_driver.sh" "$job" "$NAME"
echo "$job" > "$R/driver.launched"
echo "SERVE_AND_DRIVER_LAUNCHED job=$job tasks64 arms2 attempts1 slots128" > "$R/STATUS"
while squeue -u "$USER" -h -o %i | grep -qx "$job"; do
  if [ -f "$R/runs/$NAME/readout.json" ] && grep -q RUN_DONE "$R/runs/$NAME/driver.log"; then break; fi
  sleep 30
done
test ! -f "$R/runs/$NAME/ABORT"
test -f "$R/runs/$NAME/readout.json"
for arm in control relay_repair; do
  "$PY" "$OTA/data/relay/msa/gate.py" "$R/runs/$NAME" --arm "$arm" > "$R/$arm.gate.json"
done
echo "GENERATION_SETTLED: inspect readout and both gate files before training or scaling" > "$R/STATUS"
