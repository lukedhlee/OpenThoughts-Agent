#!/bin/bash
# engine_health.sh <jobid> — one nvidia-smi sweep over every node of a running RL job (srun --overlap, parallel).
# Prints one line per node: util of the 4 GPUs + a tag: POLICY (all GPUs < 20 GB), ENGINE-OK (engine node with all 4 busy or all 4
# idle), ENGINE-HUNG? (engine node with exactly one GPU at 0 % while the others are at 100 % — the DP4xEP4 collective-hang signature seen
# on snowball_overfit32_a 1647672, 2026-09-03 18:35 CEST). Idle engines between waves show 0/0/0/0 — only the 3x100/1x0 pattern is a hang.
J=$1; nodes=$(scontrol show hostnames $(squeue -h -j $J -o %N)); [ -n "$nodes" ] || { echo "no nodes for $J"; exit 1; }
for w in $nodes; do ( r=$(timeout 60 srun --jobid=$J --overlap -N1 -n1 -w $w nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d " " | tr "\n" ";"); 
  [ -n "$r" ] || { echo "$w NO-ANSWER"; exit 0; }
  u=$(echo "$r" | tr ";" "\n" | grep . | cut -d, -f1 | tr "\n" " "); m=$(echo "$r" | tr ";" "\n" | grep . | cut -d, -f2 | sort -n | tail -1)
  z=$(echo $u | tr " " "\n" | grep -c "^0$"); h=$(echo $u | tr " " "\n" | grep -c "^100$")
  if [ "$m" -lt 20000 ]; then tag=POLICY; elif [ "$z" -eq 1 ] && [ "$h" -eq 3 ]; then tag="ENGINE-HUNG?"; else tag=ENGINE-OK; fi
  echo "$w util=[$u] maxmem=${m}MiB $tag" ) & done; wait
