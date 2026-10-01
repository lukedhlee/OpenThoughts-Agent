#!/bin/bash
# vllm_decomp.sbatch — tis_decomp, vLLM side, one Horizon node, three serves in a row (~0.5 node-h):
#   1. Stage-3 step 1888, DP4-EP (gate 2's serve + --enable-return-routed-experts): the 96 gate-2 turns scored
#      4 times, concurrency 4 twice (gate 2's setting) and one request at a time twice (no co-batching)
#   2. Stage-3, four single-GPU servers (TP1, no DP/EP), one request at a time each: the layout comparison
#   3. H9 (relay_acont step 999), DP4-EP: its own 96 screen turns, concurrency 4 twice + one at a time once
# Each rep saves prompt logprobs (gate-2 JSONL) and the routed experts per turn (.npy [P+C, 26, 4]).
# H9 set (login node, stdlib): python3 data/r2egym/horizon/gate2_score.py build --trials
#   $S/experiments/rl/screen_h9/trials/eval_sessions/screen_h9_eval_step0 --out $E/h9_set.jsonl --n 96
#   --edges 4096,16384,32768,49152 --seed 0   (sha256 d557600b..., 42,442 completion tokens)
#   sbatch ~/OpenThoughts-Agent-rl/data/r2egym/horizon/rl/tis_decomp/vllm_decomp.sbatch
#SBATCH -J tisd_vllm
#SBATCH -A CCR24067
#SBATCH -p debug
#SBATCH -N 1
#SBATCH -t 00:55:00
#SBATCH -o /scratch/11584/lukedhlee/experiments/rl/tis_decomp/logs/%x.%j.out
unset LD_PRELOAD   # first command: TACC's XALT preload prints an NVML warning into every $(...) capture
set -uo pipefail
ROOT=$HOME/snowball; S=/scratch/11584/$USER
HERE=${HERE:-$HOME/OpenThoughts-Agent-rl/data/r2egym/horizon/rl/tis_decomp}
PY=$ROOT/envs/snowball/bin/python
export PYTHONPATH=$ROOT/src/marin_vllm_eagle3${PYTHONPATH:+:$PYTHONPATH}   # the EAGLE-3 overlay shadows the venv's vllm
export PATH=$ROOT/envs/snowball/bin:$PATH
E=$S/experiments/rl/tis_decomp; O=$E/vllm; mkdir -p $O $E/logs
S3=${S3:-$S/models/snowball-s3-nemotron-terminal-step1888}
H9=${H9:-$S/snowball-sft/experiments/snowball-relay-sft/relay_acont/lr3e-4-sched3/export-step999-hf-bf16}
S3SET=${S3SET:-$S/horizon_gates/gate2_set.jsonl}; H9SET=${H9SET:-$E/h9_set.jsonl}
MAXLEN=65536; GEN='{"temperature":1.0,"top_p":1.0,"top_k":-1}'
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8
export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ALLREDUCE_USE_SYMM_MEM=0
export CC=gcc CXX=g++
C=$S/cache; L=/tmp/tisd_$SLURM_JOB_ID; mkdir -p $L
export XDG_CACHE_HOME=$C/xdg FLASHINFER_WORKSPACE_BASE=$C/flashinfer TMPDIR=$L
J=$SLURM_JOB_ID
echo "node=$(hostname) job=$J s3=$(readlink -f $S3) h9=$H9"
sha256sum $S3SET $H9SET
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | head -1
$PY -c "import vllm,torch;print('vllm',vllm.__file__,vllm.__version__,'torch',torch.__version__)"

PGIDS=(); declare -A PIDOF
start() {  # start NAME MODEL PORT GPUS LAYOUT_ARGS... ; node-local compile caches per server
  local name=$1 model=$2 port=$3 gpus=$4; shift 4
  local chat=(); [ -f "$model/chat_template.jinja" ] && chat=(--chat-template "$model/chat_template.jinja")
  mkdir -p $L/$name
  CUDA_VISIBLE_DEVICES=$gpus VLLM_CACHE_ROOT=$L/$name/vllm TRITON_CACHE_DIR=$L/$name/triton \
  TORCHINDUCTOR_CACHE_DIR=$L/$name/inductor setsid $PY -m vllm.entrypoints.openai.api_server --model "$model" \
      --served-model-name snowball --port $port --max-model-len $MAXLEN \
      --hf-overrides "{\"max_position_embeddings\": $MAXLEN, \"max_seq_len\": $MAXLEN}" \
      --override-generation-config "$GEN" --enable-return-routed-experts "${chat[@]}" "$@" \
      > $E/logs/server_${name}.$J.log 2>&1 &
  PGIDS+=($!); PIDOF[$name]=$!
  echo "started $name pid/pgid $! port $port gpus $gpus: $*"
}
wait_up() {  # wait_up PORT NAME
  for i in $(seq 1 120); do
    curl -sf localhost:$1/health >/dev/null && { echo "$2 UP after $((i*10))s"; return 0; }
    kill -0 ${PIDOF[$2]} 2>/dev/null || { echo "$2 died"; tail -60 $E/logs/server_$2.$J.log; return 1; }
    sleep 10
  done
  echo "$2 did not come up"; tail -40 $E/logs/server_$2.$J.log; return 1
}
stop_all() {  # SIGTERM each server's own process group, then wait for the GPUs to drain
  for g in "${PGIDS[@]}"; do kill -TERM -- -$g 2>/dev/null; done
  for i in $(seq 1 30); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
    [ "${used:-0}" -lt 2000 ] && break; sleep 4
  done
  for g in "${PGIDS[@]}"; do kill -KILL -- -$g 2>/dev/null; done
  PGIDS=(); sleep 3
  nvidia-smi --query-gpu=memory.used --format=csv,noheader | tr '\n' ' '; echo
}
trap stop_all EXIT
score() {  # score SET LABEL REP CONC URLS
  $PY $HERE/vllm_score.py --set $1 --label $2 --rep $3 --conc $4 --urls $5 \
      --out $O/$2.r$3.jsonl --route-dir $O/route_$2.r$3
}
DP4EP=(--tensor-parallel-size 1 --data-parallel-size 4 --enable-expert-parallel --max-num-seqs 32 --gpu-memory-utilization 0.80)
U0=http://localhost:8000

echo "=== phase 1: Stage-3 DP4-EP $(date -Is)"
start s3_dp4ep $S3 8000 0,1,2,3 "${DP4EP[@]}"
if wait_up 8000 s3_dp4ep; then
  grep -m4 -i -E "attention backend|Unquantized MoE backend|max_num_batched_tokens|RoutedExpertsManager" $E/logs/server_s3_dp4ep.$J.log
  score $S3SET s3_dp4ep_c4 0 4 $U0
  score $S3SET s3_dp4ep_c1 0 1 $U0
  score $S3SET s3_dp4ep_c4 1 4 $U0
  score $S3SET s3_dp4ep_c1 1 1 $U0
fi
stop_all

echo "=== phase 2: Stage-3 single-GPU x4 $(date -Is)"
for g in 0 1 2 3; do
  start s3_tp1_g$g $S3 $((8001 + g)) $g --tensor-parallel-size 1 --max-num-seqs 8 --gpu-memory-utilization 0.92
done
ok=1; for g in 0 1 2 3; do wait_up $((8001 + g)) s3_tp1_g$g || ok=0; done
if [ $ok = 1 ]; then
  grep -m4 -i -E "attention backend|Unquantized MoE backend|max_num_batched_tokens|KV cache" $E/logs/server_s3_tp1_g0.$J.log
  score $S3SET s3_tp1_c1 0 1 http://localhost:8001,http://localhost:8002,http://localhost:8003,http://localhost:8004
fi
stop_all

echo "=== phase 3: H9 DP4-EP $(date -Is)"
start h9_dp4ep $H9 8000 0,1,2,3 "${DP4EP[@]}"
if wait_up 8000 h9_dp4ep; then
  score $H9SET h9_dp4ep_c4 0 4 $U0
  score $H9SET h9_dp4ep_c1 0 1 $U0
  score $H9SET h9_dp4ep_c4 1 4 $U0
fi
stop_all
trap - EXIT
du -sh $O; ls $O
echo "=== VLLM_DECOMP_DONE $(date -Is)"
