#!/bin/bash
# trainer_decomp.sbatch — tis_decomp, trainer side, one Horizon node (~0.4 node-h): 4 processes, one GB200 each with
# the whole model (as rl_logprob_parity.sbatch), Stage-3 on the 96 gate-2 turns then H9 on its 96 screen turns; every
# variant of trainer_decomp.py; then analyze.py. Needs vllm_decomp.sbatch's outputs (routing for the replays).
#   sbatch --dependency=afterany:<vllm job> ~/OpenThoughts-Agent-rl/data/r2egym/horizon/rl/tis_decomp/trainer_decomp.sbatch
#SBATCH -J tisd_trainer
#SBATCH -A CCR24067
#SBATCH -p debug
#SBATCH -N 1
#SBATCH -t 00:50:00
#SBATCH -o /scratch/11584/lukedhlee/experiments/rl/tis_decomp/logs/%x.%j.out
unset LD_PRELOAD   # first command: TACC's XALT preload prints an NVML warning into every $(...) capture
set -uo pipefail
export CC=gcc CXX=g++ HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=${PY:-$HOME/snowball/envs/snowball/bin/python}
HERE=${HERE:-$HOME/OpenThoughts-Agent-rl/data/r2egym/horizon/rl/tis_decomp}
S=/scratch/11584/$USER; E=$S/experiments/rl/tis_decomp; G=$S/horizon_gates
S3=${S3:-$(readlink -f $S/models/snowball-s3-nemotron-terminal-step1888)}
H9=${H9:-$S/snowball-sft/experiments/snowball-relay-sft/relay_acont/lr3e-4-sched3/export-step999-hf-bf16}
OUT=$E/trainer.$SLURM_JOB_ID
LIMIT=${LIMIT:-0}; ONLY=${ONLY:-}
echo "node=$(hostname) job=$SLURM_JOB_ID s3=$S3 h9=$H9 limit=$LIMIT only=$ONLY"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
"$PY" -c "import torch, skyrl_train, flash_attn; print('torch', torch.__version__, 'skyrl_train', skyrl_train.__file__, 'flash_attn', flash_attn.__version__)"
ls $E/vllm 2>/dev/null | tr '\n' ' '; echo
for r in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$r "$PY" -u "$HERE/trainer_decomp.py" --rank $r --world 4 --out "$OUT" --vdir $E/vllm \
    --s3-model "$S3" --s3-set $G/gate2_set.jsonl --h9-model "$H9" --h9-set $E/h9_set.jsonl \
    --limit $LIMIT --only "$ONLY" > "$OUT.rank$r.log" 2>&1 &
done
wait
tail -n 4 "$OUT".rank*.log
"$PY" "$HERE/analyze.py" --vdir $E/vllm --tout "$OUT" --s3-set $G/gate2_set.jsonl --h9-set $E/h9_set.jsonl \
  --proxy "$G/rl_lp_parity.36712.part*.jsonl" --gate2 $G/gate2_horizon_dp4ep.35865.jsonl --json "$OUT.summary.json" \
  | tee "$OUT.report.txt"
echo "=== TRAINER_DECOMP_DONE $(date -Is)"
