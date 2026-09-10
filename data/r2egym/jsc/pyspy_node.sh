#!/bin/bash
# pyspy_node.sh <jobid> <node> <outdir> — py-spy dump every vLLM engine process on one node of a running job (srun --overlap).
J=$1; W=$2; O=$3; PS=/e/project1/transfernetx/lee27/code/envs/snowball/bin/py-spy; ts=$(date +%H%M%S)
mkdir -p "$O"
timeout 300 srun --jobid=$J --overlap -N1 -n1 -w $W bash -s <<'INNER' > "$O/${ts}_$W.txt" 2>&1
PS=/e/project1/transfernetx/lee27/code/envs/snowball/bin/py-spy
echo '### nvidia-smi'; nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader; nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader
for pid in $(ps -eo pid,args | grep -E 'VLLM::|AsyncVLLMInferenceEngine' | grep -v grep | awk '{print $1}'); do
  echo; echo "=== PID $pid: $(ps -p $pid -o pcpu=,stat=,args= | cut -c1-80) ==="
  timeout 30 $PS dump --pid $pid --nonblocking 2>&1 | head -60
done
echo; echo '### native (DP0 + DP1 EngineCore)'
for pid in $(ps -eo pid,args | grep -E 'VLLM::EngineCore_DP[01]' | grep -v grep | awk '{print $1}'); do
  echo "=== NATIVE PID $pid: $(ps -p $pid -o args= | cut -c1-40) ==="; timeout 40 $PS dump --pid $pid --nonblocking --native 2>&1 | head -80
done
INNER
echo "wrote $O/${ts}_$W.txt ($(wc -l < "$O/${ts}_$W.txt") lines)"
