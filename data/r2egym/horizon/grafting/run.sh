#!/bin/bash
# Login-side controller: guarded compute build, then the existing matched evals.
set -euo pipefail
unset LD_PRELOAD
R=/scratch/11584/$USER
export GRAFT_ROOT=${GRAFT_ROOT:-$R/experiments/snowball-grafting-20261030}
export GRAFT_CODE=${GRAFT_CODE:?Set to the committed isolated checkout}
export GRAFT_REVISION=$(git -C "$GRAFT_CODE" rev-parse HEAD)
export FP32_REPLAY_ROOT=$R/experiments/paired-sft-rl-20261026/mixed/rl
export PATH=$FP32_REPLAY_ROOT/bin:$PATH
mkdir -p "$GRAFT_ROOT/logs"
exec 9>"$GRAFT_ROOT/controller.lock"
flock -n 9 || { echo 'Controller already active'; exit 1; }
say() { echo "[$(date -u +%FT%TZ)] $*" | tee "$GRAFT_ROOT/STATUS"; }
trap 'say "FAILED line=$LINENO; inspect controller.log and logs/"' ERR
test -z "$(git -C "$GRAFT_CODE" status --porcelain --untracked-files=no)"
if [ ! -f "$GRAFT_ROOT/build.job" ]; then
  while :; do
    set +e
    out=$(sbatch --export=ALL -o "$GRAFT_ROOT/logs/build.%j.log" "$GRAFT_CODE/data/r2egym/horizon/grafting/build.sbatch" 2>&1)
    rc=$?
    set -e
    if [ "$rc" = 75 ]; then say "Waiting for booking headroom"; sleep 60; continue; fi
    [ "$rc" = 0 ] || { echo "$out"; exit "$rc"; }
    job=$(echo "$out" | sed -nE 's/Submitted batch job ([0-9]+).*/\1/p' | tail -1)
    [[ "$job" =~ ^[0-9]+$ ]]
    echo "$job" > "$GRAFT_ROOT/build.job"
    say "BUILD_SUBMITTED job=$job revision=$GRAFT_REVISION"
    break
  done
fi
job=$(cat "$GRAFT_ROOT/build.job")
while squeue -h -j "$job" -t PD,R,CF,S,RQ,RS | grep -q .; do sleep 30; done
for attempt in $(seq 1 12); do
  sacct -j "$job" -X -n -o JobID,State,ExitCode > "$GRAFT_ROOT/build.sacct"
  grep -Eq 'COMPLETED|FAILED|TIMEOUT|CANCELLED|OUT_OF_MEMORY|NODE_FAIL|PREEMPTED' "$GRAFT_ROOT/build.sacct" && break
  sleep 10
done
grep -q 'COMPLETED.*0:0' "$GRAFT_ROOT/build.sacct"
test -f "$GRAFT_ROOT/models/build_complete.json"
say "BUILD_VERIFIED; waiting for eval capacity"
while :; do
  n=$(python3 "$HOME/snowball/ota-fp32-20261020/data/r2egym/horizon/sft/daytona_started.py" 2>/dev/null || echo 9999)
  if [[ "$n" =~ ^[0-9]+$ ]] && [ "$n" -le 260 ]; then break; fi
  say "Waiting for eval room: started=$n need<=640 limit=900"
  sleep 60
done
export MODELS="graft25=$GRAFT_ROOT/models/graft025 graft50=$GRAFT_ROOT/models/graft050 graft75=$GRAFT_ROOT/models/graft075 graft100=$GRAFT_ROOT/models/graft100"
export DAY=graft_20261030
say "EVAL_START: alpha 0.25/0.5/0.75/1; 2 reps; TB2.1/SWE/TB-lite"
bash "$FP32_REPLAY_ROOT/tools/main_eval.sh" > "$GRAFT_ROOT/logs/eval.log" 2>&1
say "EVAL_QUEUE_ENDED; inspect coverage and rerun infrastructure losses before scientific readout"
