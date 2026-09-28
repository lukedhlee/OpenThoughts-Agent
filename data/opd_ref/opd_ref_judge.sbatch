#!/bin/bash
# opd_ref_judge.sbatch — teacher-as-judge scoring only (opd_ref_judge.py), one Jupiter GH200 node, 4 TP1 shards.
# Env copied from opd_ref_turn1.sbatch. Modes whose four shard files exist are skipped.
#   dense   "on track?" after every turn before the first done-claim, DENSE_STEPS (default 25 26)
#   value1  "on track?" after turn 1 only (turn-1 outputs of 8 rollouts from one state), VALUE1_STEPS (default 20 steps)
#
#   RUN=<name> OPD_REF_DIR=<code dir> sbatch opd_ref_judge.sbatch
#
#SBATCH --job-name=opd_ref_judge
#SBATCH --time=02:30:00
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
DD=${DUMPDIR:-/e/data1/mmlaion/lee27/experiments/snowball_ttband_ns_c_lr5e7_c/snowball_ttband_ns_c_lr5e7_c/exports/dumped_data}
DENSE_STEPS=${DENSE_STEPS:-25 26}
VALUE1_STEPS=${VALUE1_STEPS:-1 4 7 10 13 16 19 22 25 28 31 34 37 40 43 46 49 52 55 58}
dumps() { local s out=; for s in $*; do out=$out${out:+,}$DD/global_step_${s}_train_rollouts.jsonl; done; echo $out; }
TEACHER=${TEACHER:-$M/Qwen3.8-27B}
OUT=/e/data1/mmlaion/lee27/experiments/opd_ref/$RUN; mkdir -p $OUT/logs
echo "opd_ref_judge $RUN on $(hostname) job=$SLURM_JOB_ID ($(date -Is)) dense=[$DENSE_STEPS] value1=[$VALUE1_STEPS] code=$HERE"

judge4() {  # judge4 <mode> <dump globs> <ks> <file tag>
  local mode=$1 dump=$2 ks=$3 tag=$4 g pids=()
  if [ "$(ls $OUT/judge/$mode.*.pkl 2>/dev/null | wc -l)" = 4 ]; then echo "judge $mode: done already"; return 0; fi
  for g in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$g VLLM_PORT=$((29500 + 100 * g)) $PY $HERE/opd_ref_judge.py score --mode $mode --dump "$dump" \
      --ks $ks --model $TEACHER --out $OUT --shard $g --nshards 4 > $OUT/logs/judge_$tag.$g.log 2>&1 & pids+=($!)
  done
  local rc=0; for p in "${pids[@]}"; do wait $p || rc=1; done
  echo "judge $tag: exit $rc ($(date -Is))"; for f in $OUT/logs/judge_$tag.*.log; do tail -n 1 $f; done
}

judge4 dense "$(dumps $DENSE_STEPS)" 1 dense
judge4 value "$(dumps $VALUE1_STEPS)" 1 value1
$PY $HERE/opd_ref_judge.py analyze --out $OUT
$PY $HERE/opd_ref_judge.py dense-analyze --out $OUT
echo "opd_ref_judge done ($(date -Is))"
