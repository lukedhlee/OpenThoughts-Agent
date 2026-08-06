#!/bin/bash
# Parallel model-free environment gate: pristine vs oracle, one task per node.
#
# The sequential version ran one srun at a time; at up to 40 min per run that is
# ~8 h for 6 tasks. We have 32 nodes in fleet 15500584 and 32 staged tasks, so
# pin task i to node i and run its two modes back-to-back there. Wall clock =
# time of the slowest single task, not the sum.
#
# GATE: pristine 0.0 AND oracle 1.0 on the SAME task == the environment measures
# code state. No model involved, so it cannot be confounded by agent behaviour.
set -uo pipefail
R=/p/scratch/synthlaion/lee27

# FLEET has NO default on purpose. It used to default to 15500584; that job is dead,
# and a stale default silently sends every srun at a nonexistent allocation.
FLEET=${FLEET:?set FLEET=<current sandbox fleet jobid> (squeue -u lee27 on JURECA)}

# Staging root + output paths are parameterised together so a new sweep cannot
# overwrite the validated 32-task results.
STAGE_ROOT=${EG_STAGE_ROOT:-$R/envgate}
TAG=${EG_TAG:-$(basename "$STAGE_ROOT")}
OUT=$R/envgate_results_$TAG.jsonl
PARTS=$R/envgate_parts_$TAG
LOG=$R/envgate_par_$TAG.log

# Concurrency. The fleet runs 16 workers/node, so one-task-per-node left ~15/16 of
# the allocation idle. CONC is how many tasks run at once; nodes are still assigned
# round-robin so the load spreads evenly.
CONC=${EG_CONC:-64}

export EG_STAGE_ROOT="$STAGE_ROOT"
exec >>"$LOG" 2>&1

mkdir -p $PARTS
echo "=== parallel sweep $(date '+%F %T') fleet=$FLEET stage=$STAGE_ROOT conc=$CONC ==="

mapfile -t TASKS < <(ls $STAGE_ROOT)
mapfile -t NODES < <(scontrol show hostnames "$(squeue -j $FLEET -h -o '%N')")
echo "tasks=${#TASKS[@]} nodes=${#NODES[@]}"
if [[ ${#NODES[@]} -eq 0 ]]; then
  echo "FATAL: fleet $FLEET has no nodes -- is it RUNNING?"; exit 1
fi

i=0
for T in "${TASKS[@]}"; do
  # Idempotent: skip tasks already recorded, so a re-run resumes instead of redoing.
  if [[ -s $PARTS/$T.pristine.json && -s $PARTS/$T.oracle.json ]]; then
    echo "  skip $T (already recorded)"; i=$((i+1)); continue
  fi
  N=${NODES[$(( i % ${#NODES[@]} ))]}
  (
    for M in pristine oracle; do
      srun --jobid=$FLEET --overlap -N1 -n1 -w "$N" --time=40 \
        bash $R/envgate.sh "$T" "$M" 2>/dev/null | grep "^{" >> $PARTS/$T.$M.json
    done
  ) &
  echo "  launched $T on $N"
  i=$((i+1))
  while [[ $(jobs -rp | wc -l) -ge $CONC ]]; do sleep 2; done
done

wait
cat $PARTS/*.json > $OUT 2>/dev/null
echo "=== SWEEP COMPLETE $(date '+%F %T') ==="
wc -l $OUT
