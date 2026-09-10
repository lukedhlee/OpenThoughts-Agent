#!/bin/bash
#SBATCH --time=01:15:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --output=/e/project1/transfernetx/lee27/code/snowball/bench/logs/%x_%j.out
#SBATCH --job-name=snowball_bench
#SBATCH -p booster
#SBATCH --account reformo
#SBATCH --gres=gpu:4
#SBATCH --exclude=jpbo-031-[01-48],jpbo-011-[01-48],jpbo-038-38,jpbo-004-46,jpbo-065-17,jpbo-074-22,jpbo-074-40,jpbo-048-41,jpbo-091-05,jpbo-044-0[1-5]
# Snowball serving benchmark: one GH200 node, TP1 x DP4 x EP4, api_server with the RL engine's flags
# (max_num_seqs 32 / max_num_batched_tokens 16384 / gpu-mem 0.85 / async scheduling / cudagraph cap 64),
# then bench_client.py phases. Override via --export=ALL,ARM=...,SARGS="...",ENVX="K=V K=V",...
set -uo pipefail
[ -n "${ARMFILE:-}" ] && { echo "sourcing $ARMFILE"; source "$ARMFILE"; }
module load GCC/14.3.0
module load nvidia-compilers/25.9-CUDA-13
C=/e/project1/transfernetx/lee27/code
PY=${SNOWBALL_PY:-$C/envs/snowball/bin/python}
export PATH=$C/envs/snowball/bin:$PATH   # ninja for FlashInfer JIT (production activates the venv)
M=/e/fscratch/reformo/lee27/models/snowball-s3-nemotron-terminal-step1888
B=$C/snowball/bench
ARM=${ARM:-base}
OUT=$B/results/${ARM}_$SLURM_JOB_ID; mkdir -p $OUT $B/logs
SARGS=${SARGS:-}
ENVX=${ENVX:-}
MAXLEN=${MAXLEN:-32768}
MAXSEQS=${MAXSEQS:-32}
MAXBT=${MAXBT:-16384}
MAXCG=${MAXCG:-64}
GPUMEM=${GPUMEM:-0.85}
PHASES=${PHASES:-sweep,prefill,replay_forced,replay_natural}
REPLAY=${REPLAY:-$B/replay_b28k.jsonl}
REPLAY_CONC=${REPLAY_CONC:-64}
REPLAY_BUDGET=${REPLAY_BUDGET:-420}
CONC_SWEEP=${CONC_SWEEP:-4,16,44,64,128}
PREFIXES=${PREFIXES:-2000,12000,24000}
PROFILE=${PROFILE:-0}
CLIENT_ARGS=${CLIENT_ARGS:-}

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1
export VLLM_ALLREDUCE_USE_SYMM_MEM=0
export TORCH_CUDNN_SDPA_ENABLED=0
export VLLM_MQ_MAX_CHUNKS=240
export VLLM_USE_FLASHINFER_SAMPLER=0   # production parity: RL engines log "FlashInfer top-p/top-k sampling disabled"
CACHE=/e/fscratch/reformo/lee27/cache; mkdir -p $CACHE/vllm $CACHE/xdg $CACHE/triton $CACHE/inductor $CACHE/flashinfer $CACHE/tmp
export VLLM_CACHE_ROOT=$CACHE/vllm XDG_CACHE_HOME=$CACHE/xdg TRITON_CACHE_DIR=$CACHE/triton TORCHINDUCTOR_CACHE_DIR=$CACHE/inductor FLASHINFER_WORKSPACE_BASE=$CACHE/flashinfer TMPDIR=$CACHE/tmp
for kv in $ENVX; do export "$kv"; done

echo "node=$(hostname) job=$SLURM_JOB_ID arm=$ARM maxlen=$MAXLEN maxseqs=$MAXSEQS maxbt=$MAXBT maxcg=$MAXCG gpumem=$GPUMEM sargs='$SARGS' envx='$ENVX' phases=$PHASES ($(date -Is))"
nvidia-smi --query-gpu=name,memory.total,driver_version,clocks.max.sm,clocks.max.mem --format=csv
$PY -c "import vllm,torch;print('vllm',vllm.__version__,'torch',torch.__version__,torch.version.cuda)"
env | grep -E "^VLLM_|^TRITON_|^FLASHINFER|^NCCL_" | sort > $OUT/env.txt

HFOV=""; [ "$MAXLEN" -gt 32768 ] && HFOV="--hf-overrides {\"max_position_embeddings\":$MAXLEN,\"max_seq_len\":$MAXLEN}"
PROFARGS=""
if [ "$PROFILE" = "1" ]; then
  mkdir -p $OUT/prof
  PROFARGS="--profiler-config {\"profiler\":\"torch\",\"torch_profiler_dir\":\"$OUT/prof\",\"torch_profiler_with_stack\":false,\"ignore_frontend\":true,\"warmup_iterations\":10,\"active_iterations\":${PROF_ITERS:-40},\"max_iterations\":${PROF_MAX:-80}}"
fi

$PY -m vllm.entrypoints.openai.api_server --model $M --served-model-name snowball --port 8000 \
  --max-model-len $MAXLEN --gpu-memory-utilization $GPUMEM --max-num-seqs $MAXSEQS --max-num-batched-tokens $MAXBT \
  --max-cudagraph-capture-size $MAXCG --enable-prefix-caching --enable-chunked-prefill --async-scheduling \
  --tensor-parallel-size 1 --data-parallel-size 4 --enable-expert-parallel \
  --chat-template $M/chat_template.jinja --no-enable-log-requests $HFOV $PROFARGS $SARGS > $OUT/server.log 2>&1 &
SPID=$!
UP=0
for i in $(seq 1 150); do
  curl -sf localhost:8000/health >/dev/null && { echo "UP after $((i*10))s"; UP=1; break; }
  kill -0 $SPID 2>/dev/null || { echo "server died"; grep -iE -m8 "error|Traceback|NotImplemented|ValueError" $OUT/server.log | cut -c1-240; break; }
  sleep 10
done
if [ "$UP" != "1" ]; then echo "SERVER_FAILED"; tail -60 $OUT/server.log | cut -c1-240; kill $SPID 2>/dev/null; exit 1; fi
grep -m1 -oE "Using [A-Z_]+ attention backend" $OUT/server.log; grep -m1 -oE "Using [A-Za-z ]+ Unquantized MoE backend" $OUT/server.log
grep -m1 -oE "all2all=[a-z_]+" $OUT/server.log; grep -m1 -oE "GPU KV cache size: [0-9,]+ tokens" $OUT/server.log
grep -m1 -iE "default MoE config|Using configuration from" $OUT/server.log | cut -c1-200
grep -m1 -iE "speculative|ngram" $OUT/server.log | cut -c1-200 || true

run_phase() {
  local ph=$1
  echo "=== PHASE $ph ($(date -Is))"
  case $ph in
    sweep)
      for p in ${PREFIXES//,/ }; do
        $PY $B/bench_client2.py --mode sweep --label $ARM --prefix-tokens $p --out-tokens 256 --conc $CONC_SWEEP --max-model-len $MAXLEN --out $OUT/sweep_p$p.json $CLIENT_ARGS 2>&1 | grep -vE "^\s*$"
      done;;
    prefill)
      $PY $B/bench_client2.py --mode prefill --label $ARM --prefix-tokens 12000 --conc 16,64 --max-model-len $MAXLEN --out $OUT/prefill.json $CLIENT_ARGS 2>&1;;
    replay_forced)
      $PY $B/bench_client2.py --mode replay --forced --label $ARM --replay $REPLAY --conc $REPLAY_CONC --n-traj $((REPLAY_CONC*3)) --time-budget $REPLAY_BUDGET --max-model-len $MAXLEN --out $OUT/replay_forced_c$REPLAY_CONC.json $CLIENT_ARGS 2>&1;;
    replay_natural)
      $PY $B/bench_client2.py --mode replay --label $ARM --replay $REPLAY --conc $REPLAY_CONC --n-traj $((REPLAY_CONC*3)) --time-budget $REPLAY_BUDGET --max-model-len $MAXLEN --out $OUT/replay_natural_c$REPLAY_CONC.json $CLIENT_ARGS 2>&1;;
    replay_conc)
      REPLAY_CONCS=${REPLAY_CONCS:-16,32,64,128}
      for c in ${REPLAY_CONCS//,/ }; do
        $PY $B/bench_client2.py --mode replay --forced --label $ARM --replay $REPLAY --conc $c --n-traj $((c*3)) --time-budget ${REPLAY_BUDGET} --max-model-len $MAXLEN --out $OUT/replay_forced_c$c.json $CLIENT_ARGS 2>&1
      done;;
    profile)
      echo "starting profiler"; curl -s -X POST localhost:8000/start_profile; echo
      $PY $B/bench_client2.py --mode replay --forced --label ${ARM}_prof --replay $REPLAY --conc $REPLAY_CONC --n-traj $((REPLAY_CONC*2)) --time-budget ${PROF_BUDGET:-45} --steady-after 10 --max-model-len $MAXLEN --out $OUT/replay_profiled.json $CLIENT_ARGS 2>&1
      echo "stopping profiler"; curl -s -X POST localhost:8000/stop_profile; echo; sleep 30
      ls -la $OUT/prof | head; $PY $B/prof_summary.py $OUT/prof 2>&1 | tee $OUT/prof_summary.txt;;
    *) echo "unknown phase $ph";;
  esac
}
for ph in ${PHASES//,/ }; do run_phase $ph; done
# engine-side stats snapshot from the server log
grep -E "Avg prompt throughput" $OUT/server.log | tail -400 > $OUT/engine_stats.log
echo "=== engine stats (last 400 lines): running/gen-tps/prefix-hit percentiles"
$PY - $OUT/engine_stats.log <<'PY'
import re,sys,statistics
gen=[];run=[];hit=[];pr=[]
for l in open(sys.argv[1]):
    m=re.search(r"prompt throughput: ([\d.]+) tokens/s, Avg generation throughput: ([\d.]+) tokens/s, Running: (\d+) reqs, Waiting: (\d+) reqs, GPU KV cache usage: ([\d.]+)%, Prefix cache hit rate: ([\d.]+)%",l)
    if not m: continue
    p,g,r,w,k,h=map(float,m.groups())
    if r==0: continue
    pr.append(p);gen.append(g);run.append(r);hit.append(h)
q=lambda x,f: sorted(x)[int(f*(len(x)-1))] if x else None
print(f"samples={len(gen)} running p50={q(run,.5)} p90={q(run,.9)} gen_tps p50={q(gen,.5)} p90={q(gen,.9)} prompt_tps p50={q(pr,.5)} prefix_hit p50={q(hit,.5)}")
PY
kill $SPID 2>/dev/null; wait $SPID 2>/dev/null; pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null
echo "=== SUMMARY $ARM"; grep -h "^RESULT" $B/logs/${SLURM_JOB_NAME}_$SLURM_JOB_ID.out
echo "=== DONE $(date -Is)"
