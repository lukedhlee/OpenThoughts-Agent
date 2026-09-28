#!/bin/bash
# opd_ref_turn1.sbatch — same-state evidence for PivotRL x CompassOPD at scale, on one Jupiter GH200 node:
# the first turns of GRPO groups with both outcomes across many RL steps (8 candidate turns from one state each),
# scored by the teacher and Qwen3.5 references (CompassOPD), by the teacher under a swapped issue (task relevance),
# and judged Yes/No by the teacher; plus the teacher judging done-claims on the probe's rerun steps.
# Stages whose outputs exist are skipped, so a resubmission resumes. Env and qwen4 copied from opd_ref_probe.sbatch.
#
#   RUN=<name> [STEPS='1 4 ...'] [CLAIM_STEPS='25 26'] OPD_REF_DIR=<code dir> sbatch opd_ref_turn1.sbatch
#
#SBATCH --job-name=opd_ref_turn1
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
STEPS=${STEPS:-1 4 7 10 13 16 19 22 25 28 31 34 37 40 43 46 49 52 55 58}
CLAIM_STEPS=${CLAIM_STEPS:-25 26}
dumps() { local s out=; for s in $*; do out=$out${out:+,}$DD/global_step_${s}_train_rollouts.jsonl; done; echo $out; }
DUMP=$(dumps $STEPS); CLAIM_DUMP=$(dumps $CLAIM_STEPS)
STUDENT_TOK=${STUDENT_TOK:-/e/fscratch/reformo/lee27/experiments/exports/snowball_ttband_ns_c_lr5e7_c_step24/model}
TEACHER=${TEACHER:-$M/Qwen3.8-27B}
REFS=${REFS:-0.8B,2B,9B}
OUT=/e/data1/mmlaion/lee27/experiments/opd_ref/$RUN; mkdir -p $OUT/logs
echo "opd_ref_turn1 $RUN on $(hostname) job=$SLURM_JOB_ID ($(date -Is)) steps=[$STEPS] claim_steps=[$CLAIM_STEPS] code=$HERE"
nvidia-smi --query-gpu=name,memory.total --format=csv

for v in t1 t1swap; do
  [ -f $OUT/$v/episodes.pkl ] && continue
  $PY $HERE/opd_ref_branch.py prep --dump "$DUMP" --max-depth 0 --snowball-tok $STUDENT_TOK --qwen-tok $TEACHER \
    --out $OUT/$v $([ $v = t1swap ] && echo --swap-issue) > $OUT/logs/prep_$v.log 2>&1 &
done
wait
for v in t1 t1swap; do [ -f $OUT/$v/episodes.pkl ] || { echo "prep $v failed"; tail -n 5 $OUT/logs/prep_$v.log; exit 1; }; done
echo "prep done ($(date -Is))"; cat $OUT/t1/prep_summary.json

qwen4() {  # qwen4 <run dir> <name> <model dir>: four TP1 shards, one per GPU
  local dir=$1 name=$2 model=$3 g pids=()
  if [ "$(ls $dir/scores/$name.*.pkl 2>/dev/null | wc -l)" = 4 ]; then echo "$dir $name: done already"; return 0; fi
  for g in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$g VLLM_PORT=$((29500 + 100 * g)) $PY $HERE/opd_ref_probe.py score --out $dir --name $name \
      --model $model --family qwen --shard $g --nshards 4 --max-len 32768 --batched-tokens 2048 \
      > $OUT/logs/$(basename $dir).$name.$g.log 2>&1 & pids+=($!)
  done
  local rc=0; for p in "${pids[@]}"; do wait $p || rc=1; done
  echo "$(basename $dir) $name: exit $rc ($(date -Is))"; for f in $OUT/logs/$(basename $dir).$name.*.log; do tail -n 1 $f; done
}

judge4() {  # judge4 <mode> <dump globs>
  local mode=$1 dump=$2 g pids=()
  if [ "$(ls $OUT/judge/$mode.*.pkl 2>/dev/null | wc -l)" = 4 ]; then echo "judge $mode: done already"; return 0; fi
  for g in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$g VLLM_PORT=$((29500 + 100 * g)) $PY $HERE/opd_ref_judge.py score --mode $mode --dump "$dump" \
      --model $TEACHER --out $OUT --shard $g --nshards 4 > $OUT/logs/judge_$mode.$g.log 2>&1 & pids+=($!)
  done
  local rc=0; for p in "${pids[@]}"; do wait $p || rc=1; done
  echo "judge $mode: exit $rc ($(date -Is))"; for f in $OUT/logs/judge_$mode.*.log; do tail -n 1 $f; done
}

qwen4 $OUT/t1 teacher $TEACHER
judge4 turn1 "$DUMP"
qwen4 $OUT/t1swap teacher $TEACHER
for r in ${REFS//,/ }; do qwen4 $OUT/t1 $r $M/Qwen3.5-$r; done
judge4 claim "$CLAIM_DUMP"

$PY $HERE/opd_ref_turns.py --out $OUT/t1 --refs $REFS --mix-refs 2B,9B --no-student --same-state-only \
  --pmi-dir $OUT/t1swap > $OUT/logs/turns.log 2>&1; tail -n 25 $OUT/logs/turns.log
$PY $HERE/opd_ref_judge.py analyze --out $OUT
echo "opd_ref_turn1 done ($(date -Is))"
