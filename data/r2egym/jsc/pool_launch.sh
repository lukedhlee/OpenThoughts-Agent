#!/bin/bash
# pool_launch.sh <prefix> [shards=8] [stagger_sec=180]  — submit shard sbatches from the OTA root, staggered.
PREFIX=$1; N=${2:-8}; ST=${3:-180}
cd /e/project1/transfernetx/lee27/code/OpenThoughts-Agent || exit 1
for i in $(seq 0 $((N-1))); do
  sb=/e/fscratch/reformo/lee27/experiments/${PREFIX}_s$i/sbatch/${PREFIX}_s${i}_rl.sbatch
  out=$(sbatch $sb) && echo "$(date +%H:%M:%S) ${PREFIX}_s$i -> $out"
  [ $i -lt $((N-1)) ] && sleep $ST
done
