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
FLEET=${FLEET:-15500584}
OUT=$R/envgate_results.jsonl
PARTS=$R/envgate_parts
LOG=$R/envgate_par.log
exec >>"$LOG" 2>&1

rm -rf $PARTS; mkdir -p $PARTS
echo "=== parallel sweep $(date '+%F %T') fleet=$FLEET ==="

mapfile -t TASKS < <(ls $R/envgate)
mapfile -t NODES < <(scontrol show hostnames "$(squeue -j $FLEET -h -o '%N')")
echo "tasks=${#TASKS[@]} nodes=${#NODES[@]}"

i=0
for T in "${TASKS[@]}"; do
  N=${NODES[$(( i % ${#NODES[@]} ))]}
  (
    for M in pristine oracle; do
      srun --jobid=$FLEET --overlap -N1 -n1 -w "$N" --time=40 \
        bash $R/envgate.sh "$T" "$M" 2>/dev/null | grep "^{" >> $PARTS/$T.$M.json
    done
  ) &
  echo "  launched $T on $N"
  i=$((i+1))
done

wait
cat $PARTS/*.json > $OUT 2>/dev/null
echo "=== SWEEP COMPLETE $(date '+%F %T') ==="
wc -l $OUT
