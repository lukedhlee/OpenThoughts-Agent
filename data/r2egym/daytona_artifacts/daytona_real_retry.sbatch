#!/bin/bash
#SBATCH --job-name=transport-daytona-real
#SBATCH --partition=booster
#SBATCH --account=reformo
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --time=00:30:00
#SBATCH --signal=B:USR1@180
#SBATCH --output=/e/fscratch/reformo/lee27/experiments/rollout_transport_pair_20260915/%x_%j.out
set -euo pipefail
umask 077
ulimit -c 0
ulimit -n 65536
module load GCC/14.3.0
module load nvidia-compilers/25.9-CUDA-13
X=/e/fscratch/reformo/lee27/experiments/rollout_transport_pair_20260915
V=/e/project1/transfernetx/lee27/code/envs/snowball-v2
M=/e/fscratch/reformo/lee27/models/snowball-s3-nemotron-terminal-step1888
RUN="$X/run_$SLURM_JOB_ID"
mkdir -p "$RUN"
export PATH="$V/bin:$PATH"
export PYTHONPATH=/e/project1/transfernetx/lee27/code/src/marin_vllm_eagle3:/e/fscratch/reformo/lee27/code/harbor-pacing/src:/e/fscratch/reformo/lee27/experiments/tokenization_transport_20260915/deps
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True VLLM_ALLREDUCE_USE_SYMM_MEM=0 TORCH_CUDNN_SDPA_ENABLED=0 VLLM_MQ_MAX_CHUNKS=240 VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_SERVER_DEV_MODE=1
CACHE=/e/fscratch/reformo/lee27/cache
export VLLM_CACHE_ROOT=$CACHE/vllm XDG_CACHE_HOME=$CACHE/xdg TRITON_CACHE_DIR=$CACHE/triton TORCHINDUCTOR_CACHE_DIR=$CACHE/inductor FLASHINFER_WORKSPACE_BASE=$CACHE/flashinfer TMPDIR=$CACHE/tmp
set -a
source /e/fscratch/reformo/lee27/keys/socks5_currease.env
source /e/fscratch/reformo/lee27/keys/daytona_eval.env
set +a
export OPENAI_API_KEY=EMPTY
export HARBOR_DAYTONA_CREATE_RATE=5 HARBOR_DAYTONA_CREATE_SHARES=1
export HTTPS_PROXY=http://127.0.0.1:18946 HTTP_PROXY=http://127.0.0.1:18946
export NO_PROXY=127.0.0.1,localhost,10.128.1.2
export APPTAINER_BRIDGE_URL=http://10.128.1.2:9936 BRIDGE_EXEC_TIMEOUT=1200
server='' gateway='' child=''
trap '[[ -z "$server" ]] || kill "$server" 2>/dev/null; [[ -z "$gateway" ]] || kill "$gateway" 2>/dev/null' EXIT
trap '[[ -z "$child" ]] || kill -INT "$child" 2>/dev/null' USR1 TERM INT
"$V/bin/python" -u "$X/async_socks_connect_proxy.py" --ready "$RUN/gateway-ready" > "$RUN/gateway.log" 2>&1 &
gateway=$!
for i in $(seq 1 60); do [[ -f "$RUN/gateway-ready" ]] && break; sleep 1; done
[[ -f "$RUN/gateway-ready" ]] || exit 1
"$V/bin/python" -c 'import json; p="/e/fscratch/reformo/lee27/experiments/daytona_fixed_ramp_20260915/run_1821426/preflight/preflight_result.json"; assert json.load(open(p))["stats"]["n_errored_trials"] == 0' 
"$V/bin/python" -u -m vllm.entrypoints.openai.api_server --model "$M" --served-model-name snowball --host 127.0.0.1 --port 8000 \
  --max-model-len 65536 --gpu-memory-utilization .85 --max-num-seqs 32 --max-num-batched-tokens 16384 \
  --max-cudagraph-capture-size 64 --enable-prefix-caching --enable-chunked-prefill --async-scheduling \
  --tensor-parallel-size 1 --data-parallel-size 4 --enable-expert-parallel \
  --chat-template "$M/chat_template.jinja" --no-enable-log-requests \
  --hf-overrides '{"max_position_embeddings":65536,"max_seq_len":65536}' \
  --speculative-config '{"method":"eagle3","model":"/e/data1/mmlaion/lee27/eagle3/probe_adapt_20260911/checkpoints/3","num_speculative_tokens":3}' \
  > "$RUN/server.log" 2>&1 &
server=$!
UP=0
for i in $(seq 1 90); do
  if curl -fsS --max-time 3 http://127.0.0.1:8000/health >/dev/null 2>&1; then UP=1; break; fi
  kill -0 "$server" 2>/dev/null || { tail -50 "$RUN/server.log"; exit 1; }
  sleep 10
done
[[ "$UP" == 1 ]] || { tail -50 "$RUN/server.log"; exit 1; }
for backend in daytona; do
  curl -fsS -X POST http://127.0.0.1:8000/reset_prefix_cache > "$RUN/cache_reset_${backend}.txt"
  "$V/bin/python" -u "$X/paired_rollout_probe.py" --backend "$backend" --tasks "$X/tasks.json" --out "$RUN" --name "${backend}_16" > "$RUN/${backend}.log" 2>&1 &
  child=$!
  wait "$child"
  child=''
done
