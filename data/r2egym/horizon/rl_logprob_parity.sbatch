#!/bin/bash
# rl_logprob_parity.sbatch — gate-5 proxy on one Horizon node: MarinSkyRL trainer log-probs (4 processes, one GB200
# each, the full 67B model per GPU) on the 96 gate-2 turns, then the comparison against Horizon's and Jupiter's vLLM.
# ~0.5-1 node-h. See rl_logprob_parity.py.   sbatch data/r2egym/horizon/rl_logprob_parity.sbatch
#SBATCH -J rl_lp_parity
#SBATCH -A CCR24067
#SBATCH -p debug
#SBATCH -N 1
#SBATCH -t 01:30:00
#SBATCH -o /scratch/11584/lukedhlee/horizon_gates/logs/rl_lp_parity.%j.out
set -uo pipefail
export CC=gcc CXX=g++ HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=${PY:-$HOME/snowball/envs/snowball/bin/python}
HERE=${HERE:-$HOME/snowball/ota-train/data/r2egym/horizon}
G=/scratch/11584/lukedhlee/horizon_gates
MODEL=${MODEL:-$(ls -d /scratch/11584/lukedhlee/hf_hub/models--laion--snowball-67b-a2b-sft-s3-nemotron-terminal-step1888/snapshots/* | head -1)}
OUT=$G/rl_lp_parity.$SLURM_JOB_ID
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
"$PY" -c "import torch, skyrl_train, flash_attn; print('torch', torch.__version__, 'skyrl_train', skyrl_train.__file__, 'flash_attn', flash_attn.__version__)"
for r in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$r "$PY" -u "$HERE/rl_logprob_parity.py" score --model "$MODEL" --set $G/gate2_set.jsonl \
    --out "$OUT" --rank $r --world 4 > "$OUT.rank$r.log" 2>&1 &
done
wait
tail -n 3 "$OUT".rank*.log
"$PY" "$HERE/rl_logprob_parity.py" compare --set $G/gate2_set.jsonl --vllm $G/gate2_horizon_dp4ep.35865.jsonl --parts "$OUT.part*.jsonl"
echo RL_LP_PARITY_DONE
