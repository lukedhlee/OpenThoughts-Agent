#!/bin/bash
# iterlog_dump.sh <jobid> <node> [N=12] — on one engine node of a running job, print the last N per-iteration engine log lines
# of every EngineCore rank (needs the run launched with enable_logging_iteration_details=true) plus each rank's stats lines.
J=$1; W=$2; N=${3:-12}
timeout 150 srun --jobid=$J --overlap -N1 -n1 -w $W bash -s "$N" <<'INNER'
N=$1; cd $(find /tmp/ray -maxdepth 5 -type d -name logs | head -1) || exit 1
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader | tr "\n" ";"; echo
for r in 0 1 2 3; do f=$(grep -l "EngineCore_DP$r" worker-*.out | head -1); echo "##### DP$r $f"
  grep -n -E "iteration|dummy|ctx_requests|Iteration|cudagraph" $f | tail -n $N | cut -c1-220
  echo "  stats lines: $(grep -c 'Avg prompt' $f); last: $(grep 'Avg prompt' $f | tail -n 1 | cut -c1-150)"; done
INNER
