#!/bin/bash
# qwen_bench.sbatch — serving throughput for the relay-rollout teacher (Qwen3.8-27B) on one Jupiter GH200 node.
#
# Question: how many agent turns per minute (and so episodes per node-hour) one node sustains for Terminus-2-shaped
# multi-turn load, and whether Qwen's built-in MTP draft helps at that concurrency. Four data-parallel replicas
# (TP1 x DP4; 27B dense fits one GH200 with room for KV), each layout in LAYOUTS started in turn, then a natural
# (non-forced) replay of real Terminus-2 conversations at each concurrency in REPLAY_CONCS. The model generates freely,
# thinking included, so completion lengths are Qwen's own. GAP stands in for sandbox execution between turns.
# Design note: ai_memory/active/snowball-sft/research/2026-09-24_relay_rollouts_design.md
#
#   REPLAY=<replay.jsonl> sbatch qwen_bench.sbatch
#
#SBATCH --job-name=qwen38_bench
#SBATCH --time=01:15:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:4
#SBATCH -p booster
#SBATCH --account laionize
#SBATCH --output=/e/fscratch/reformo/lee27/experiments/relay/bench/logs/%x_%j.out
#SBATCH --exclude=jpbo-031-[01-48],jpbo-011-[01-48],jpbo-038-38,jpbo-004-46,jpbo-065-17,jpbo-074-22,jpbo-074-40,jpbo-048-41,jpbo-091-05,jpbo-044-0[1-5]
set -uo pipefail
module load GCC/14.3.0
module load nvidia-compilers/25.9-CUDA-13
C=/e/project1/transfernetx/lee27/code
PY=$C/envs/snowball-v2/bin/python
export PYTHONPATH=$C/src/marin_vllm_eagle3${PYTHONPATH:+:$PYTHONPATH}   # same vLLM as serve_snowball.sbatch, so a co-located run uses one build
# vLLM's qwen3_5.py imports qwen3_vl -> transformers' Qwen2-VL image processor -> torchvision, which snowball-v2 lacks.
# torchvision 0.26.0+cu130 (matches torch 2.11) sits in a side dir (pip --no-deps --target) so the shared env is untouched.
export PYTHONPATH=$PYTHONPATH:$C/envs/relay-extra
export PATH=$C/envs/snowball-v2/bin:$PATH
M=${MODEL:-/e/data1/mmlaion/lee27/models/Qwen3.8-27B}
B=${BENCH_DIR:-$C/relay/bench}
E=/e/fscratch/reformo/lee27/experiments/relay/bench
OUT=$E/results/$SLURM_JOB_ID; mkdir -p $OUT $E/logs
REPLAY=${REPLAY:?set REPLAY to a replay jsonl}
LAYOUTS=${LAYOUTS:-mtp2,nomtp}
REPLAY_CONCS=${REPLAY_CONCS:-64,128,256}
BUDGET=${BUDGET:-360}
GAP=${GAP:-5}
MAXLEN=${MAXLEN:-65536}
MAXTOK=${MAXTOK:-8192}
MAXSEQS=${MAXSEQS:-96}
GPUMEM=${GPUMEM:-0.90}

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_ALLREDUCE_USE_SYMM_MEM=0
CACHE=/e/fscratch/reformo/lee27/cache; mkdir -p $CACHE/vllm $CACHE/xdg $CACHE/triton $CACHE/inductor $CACHE/flashinfer $CACHE/tmp
export VLLM_CACHE_ROOT=$CACHE/vllm XDG_CACHE_HOME=$CACHE/xdg TRITON_CACHE_DIR=$CACHE/triton TORCHINDUCTOR_CACHE_DIR=$CACHE/inductor FLASHINFER_WORKSPACE_BASE=$CACHE/flashinfer TMPDIR=$CACHE/tmp

[ -f "$M/config.json" ] || { echo "no model at $M"; exit 1; }
[ -f "$REPLAY" ] || { echo "no replay at $REPLAY"; exit 1; }
echo "node=$(hostname) job=$SLURM_JOB_ID model=$M layouts=$LAYOUTS concs=$REPLAY_CONCS budget=$BUDGET gap=$GAP maxlen=$MAXLEN maxtok=$MAXTOK replay=$REPLAY ($(date -Is))"
nvidia-smi --query-gpu=name,memory.total --format=csv
$PY -c "import vllm,torch;print('vllm',vllm.__file__,vllm.__version__,'torch',torch.__version__)"
$PY -c "import torchvision, vllm.model_executor.models.qwen3_5; print('torchvision', torchvision.__version__, 'qwen3_5 import ok')" || { echo "qwen3_5 import failed"; exit 1; }

run_layout() {
  local L=$1 SPEC=()
  case $L in
    mtp1)  SPEC=(--speculative-config '{"method":"mtp","num_speculative_tokens":1}');;
    mtp2)  SPEC=(--speculative-config '{"method":"mtp","num_speculative_tokens":2}');;
    nomtp) SPEC=();;
    *) echo "unknown layout $L"; return;;
  esac
  local LO=$OUT/$L; mkdir -p $LO
  echo "=== LAYOUT $L ($(date -Is))"
  $PY -m vllm.entrypoints.openai.api_server --model $M --served-model-name qwen38 --port 8000 \
    --tensor-parallel-size 1 --data-parallel-size 4 \
    --max-model-len $MAXLEN --gpu-memory-utilization $GPUMEM --max-num-seqs $MAXSEQS \
    --enable-prefix-caching --enable-chunked-prefill --no-enable-log-requests "${SPEC[@]}" > $LO/server.log 2>&1 &
  local SPID=$! UP=0
  for i in $(seq 1 120); do
    curl -sf localhost:8000/health >/dev/null && { echo "UP after $((i*10))s"; UP=1; break; }
    kill -0 $SPID 2>/dev/null || { echo "server died"; grep -iE -m8 "error|Traceback|ValueError" $LO/server.log | cut -c1-240; break; }
    sleep 10
  done
  if [ "$UP" != "1" ]; then echo "SERVER_FAILED $L"; tail -40 $LO/server.log | cut -c1-240; kill $SPID 2>/dev/null; wait $SPID 2>/dev/null; return; fi
  grep -m1 -oE "GPU KV cache size: [0-9,]+ tokens" $LO/server.log
  grep -m1 -oE "Maximum concurrency for [0-9,]+ tokens per request: [0-9.]+x" $LO/server.log
  for c in ${REPLAY_CONCS//,/ }; do
    $PY $B/bench_client2.py --mode replay --label ${L}_c$c --model qwen38 --replay $REPLAY --conc $c --n-traj $((c*3)) \
      --time-budget $BUDGET --gap $GAP --max-tokens $MAXTOK --max-model-len $MAXLEN --no-logprobs \
      --out $LO/replay_natural_c$c.json 2>&1 | grep -E "^RESULT|error|wrote"
  done
  # spec-decode acceptance and engine load over the run
  grep -E "SpecDecoding metrics" $LO/server.log | tail -3 | cut -c1-260
  grep -E "Avg prompt throughput" $LO/server.log | tail -400 > $LO/engine_stats.log
  $PY - $LO/engine_stats.log <<'PY'
import re,sys
gen=[];run=[];hit=[];kv=[]
for l in open(sys.argv[1]):
    m=re.search(r"generation throughput: ([\d.]+) tokens/s, Running: (\d+) reqs.*?KV cache usage: ([\d.]+)%, Prefix cache hit rate: ([\d.]+)%",l)
    if not m: continue
    g,r,k,h=map(float,m.groups())
    if r==0: continue
    gen.append(g);run.append(r);kv.append(k);hit.append(h)
q=lambda x,f: sorted(x)[int(f*(len(x)-1))] if x else None
print(f"engine samples={len(gen)} running p50={q(run,.5)} gen_tps/replica p50={q(gen,.5)} kv% p90={q(kv,.9)} prefix_hit p50={q(hit,.5)}")
PY
  kill $SPID 2>/dev/null; wait $SPID 2>/dev/null; pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null; sleep 15
}
for L in ${LAYOUTS//,/ }; do run_layout $L; done
echo "=== SUMMARY"; grep -h "^RESULT" $E/logs/${SLURM_JOB_NAME}_$SLURM_JOB_ID.out
echo "=== DONE $(date -Is)"
