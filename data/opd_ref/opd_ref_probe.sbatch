#!/bin/bash
# opd_ref_probe.sbatch — the scoring-only CompassOPD reference test (opd_ref_probe.py) on one Jupiter GH200 node:
# prep on CPU, then every model scores every prompt token of the same Snowball rollouts (vLLM prompt_logprobs=0),
# then analyze. Stages whose outputs exist are skipped, so a resubmission resumes.
#   teacher  Qwen3.8-27B            4 shards, TP1 each
#   refs     Qwen3.5-{0.8,2,4,9}B   4 shards, TP1 each
#   student  the probed Snowball    through a vLLM server, TP1 x DP4 x EP as serve_node.sh serves it (GrugMoE refuses
#            TP>1; job 2100734), no EAGLE-3 draft
# Env copied by diff from data/relay/pilot/serve_node.sh (EAGLE-3 vLLM overlay, relay-extra torchvision for qwen3_5).
#
#   RUN=<name> SESSION=<eval session dir> STUDENT=<model dir> sbatch opd_ref_probe.sbatch
#
#SBATCH --job-name=opd_ref_probe
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:4
#SBATCH -p booster
#SBATCH --account transfernetx
#SBATCH --output=/e/data1/mmlaion/lee27/experiments/opd_ref/logs/%x_%j.out
#SBATCH --exclude=jpbo-031-[01-48],jpbo-011-[01-48],jpbo-038-38,jpbo-004-46,jpbo-065-17,jpbo-074-22,jpbo-074-40,jpbo-048-41,jpbo-091-05,jpbo-044-0[1-5]
set -uo pipefail
module load GCC/14.3.0
module load nvidia-compilers/25.9-CUDA-13
C=/e/project1/transfernetx/lee27/code
HERE=${OPD_REF_DIR:-$C/ota-opdref/data/opd_ref}
PY=$C/envs/snowball-v2/bin/python
export PYTHONPATH=$C/src/marin_vllm_eagle3:$C/envs/relay-extra
export PATH=$C/envs/snowball-v2/bin:$PATH
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ALLREDUCE_USE_SYMM_MEM=0
CACHE=/e/fscratch/reformo/lee27/cache; mkdir -p $CACHE/vllm $CACHE/xdg $CACHE/triton $CACHE/inductor $CACHE/flashinfer $CACHE/tmp
export VLLM_CACHE_ROOT=$CACHE/vllm XDG_CACHE_HOME=$CACHE/xdg TRITON_CACHE_DIR=$CACHE/triton TORCHINDUCTOR_CACHE_DIR=$CACHE/inductor FLASHINFER_WORKSPACE_BASE=$CACHE/flashinfer TMPDIR=$CACHE/tmp

M=/e/data1/mmlaion/lee27/models
RUN=${RUN:?RUN=<name>}
SESSION=${SESSION:?SESSION=<eval session dir>}
STUDENT=${STUDENT:?STUDENT=<model dir>}
TEACHER=${TEACHER:-$M/Qwen3.8-27B}
REFS=${REFS:-0.8B,2B,4B,9B}
OUT=/e/data1/mmlaion/lee27/experiments/opd_ref/$RUN; mkdir -p $OUT/logs
echo "opd_ref_probe $RUN on $(hostname) job=$SLURM_JOB_ID ($(date -Is)) session=$SESSION student=$STUDENT"
nvidia-smi --query-gpu=name,memory.total --format=csv

[ -f $OUT/episodes.pkl ] || $PY $HERE/opd_ref_probe.py prep --session $SESSION --snowball-tok $STUDENT \
  --qwen-tok $TEACHER --out $OUT || { echo "prep failed"; exit 1; }

qwen4() {  # qwen4 <name> <model dir>: four TP1 shards, one per GPU
  local name=$1 model=$2 g pids=()
  if [ "$(ls $OUT/scores/$name.*.pkl 2>/dev/null | wc -l)" = 4 ]; then echo "$name: done already"; return 0; fi
  for g in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$g VLLM_PORT=$((29500 + 100 * g)) $PY $HERE/opd_ref_probe.py score --out $OUT --name $name \
      --model $model --family qwen --shard $g --nshards 4 --max-len 98304 --batched-tokens 2048 \
      > $OUT/logs/$name.$g.log 2>&1 & pids+=($!)
  done
  local rc=0; for p in "${pids[@]}"; do wait $p || rc=1; done
  echo "$name: exit $rc ($(date -Is))"; for f in $OUT/logs/$name.*.log; do tail -n 1 $f; done
}

qwen4 teacher $TEACHER
for r in ${REFS//,/ }; do qwen4 $r $M/Qwen3.5-$r; done
if [ ! -f $OUT/scores/student.0.pkl ]; then
  for try in 1 2 3; do
    $PY -m vllm.entrypoints.openai.api_server --model $STUDENT --served-model-name snowball --port 8000 \
      --tensor-parallel-size 1 --data-parallel-size 4 --enable-expert-parallel \
      --max-model-len 65536 --hf-overrides '{"max_position_embeddings": 65536, "max_seq_len": 65536}' \
      --max-num-seqs 8 --gpu-memory-utilization 0.80 --max-num-batched-tokens 4096 \
      > $OUT/logs/student_server.$try.log 2>&1 & SPID=$!
    up=0; for i in $(seq 1 120); do
      kill -0 $SPID 2>/dev/null || break
      curl -sf --max-time 5 localhost:8000/health >/dev/null && { up=1; break; }; sleep 10
    done
    [ $up = 1 ] && break
    echo "student server not healthy (try $try)"; kill $SPID 2>/dev/null; wait $SPID 2>/dev/null; sleep 20
  done
  if [ $up = 1 ]; then
    $PY $HERE/opd_ref_probe.py score-http --out $OUT --name student --family snowball --concurrency 16 \
      > $OUT/logs/student.log 2>&1
    echo "student: exit $? ($(date -Is))"; tail -n 1 $OUT/logs/student.log
  fi
  kill $SPID 2>/dev/null; wait $SPID 2>/dev/null
fi
$PY $HERE/opd_ref_probe.py analyze --out $OUT --refs $REFS
echo "opd_ref_probe done ($(date -Is))"
