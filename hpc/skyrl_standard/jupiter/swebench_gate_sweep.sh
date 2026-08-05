#!/bin/bash
# Fan out the SWE-bench nop-vs-oracle gate: one task per node, both modes on
# that node back-to-back. Same shape as envgate_par.sh.
set -uo pipefail
R=/p/scratch/synthlaion/lee27
FLEET=${FLEET:-15500584}
LIST=${1:-$R/pilot.txt}
OUT=$R/swebench_gate_results.jsonl
PARTS=$R/swebench_gate_parts
LOG=$R/swebench_gate_sweep.log
exec >>"$LOG" 2>&1

rm -rf $PARTS; mkdir -p $PARTS
echo "=== swebench gate sweep $(date '+%F %T') list=$LIST ==="
mapfile -t TASKS < "$LIST"
mapfile -t NODES < <(scontrol show hostnames "$(squeue -j $FLEET -h -o '%N')")
echo "tasks=${#TASKS[@]} nodes=${#NODES[@]}"

i=0
for T in "${TASKS[@]}"; do
  [[ -n "$T" ]] || continue
  N=${NODES[$(( i % ${#NODES[@]} ))]}
  (
    for M in nop oracle; do
      srun --jobid=$FLEET --overlap -N1 -n1 -w "$N" --time=45 \
        bash $R/swebench_gate.sh "$T" "$M" 2>/dev/null | grep "^{" >> $PARTS/$T.$M.json
    done
  ) &
  echo "  launched $T on $N"
  i=$((i+1))
done
wait
cat $PARTS/*.json > $OUT 2>/dev/null
echo "=== SWEEP COMPLETE $(date '+%F %T') === $(wc -l < $OUT) records"
