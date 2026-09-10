#!/bin/bash
# envcheck_node.sh <jobid> <node> — print NCCL/Triton env of the first EngineCore process on a node (verifies sbatch env propagation)
timeout 90 srun --jobid=$1 --overlap -N1 -n1 -w $2 bash -s <<'INNER'
pid=$(ps -eo pid,args | grep 'VLLM::EngineCore' | grep -v grep | head -1 | awk '{print $1}')
echo "engines=$(ps -eo args | grep -c 'VLLM::EngineCore') first_pid=${pid:-none}"
[ -n "$pid" ] && tr '\0' '\n' < /proc/$pid/environ | grep -E 'NCCL_GRAPH_MIXING|NCCL_PXN_DISABLE|^TRITON_CACHE_DIR|VLLM_PORT=' 
INNER
