#!/bin/bash
# heldout_nll.sbatch — held-out loss of one Snowball HF export on the Kimi SWE-smith held-out split.
# Serves the export with the marin vLLM fork on one Jupiter GH200 node (TP1 x DP4 x EP4, same geometry as
# tb2/serve_snowball.sbatch, no draft: prefill-only scoring), runs data/swesmith/heldout_nll.py against it,
# writes the JSON, and stops the server so the node is released. ~15 min per export.
# Prompt log-probs are a full-vocab fp32 log_softmax over every prefill chunk (8,192 tokens x 128k x 4 B = 4 GiB);
# at 0.90 utilization the KV cache leaves ~2 GiB and the worker OOMs (job 1867356). Hence 4,096-token chunks + 0.85.
#
#   MODEL=<hf export dir> NAME=<tag> sbatch heldout_nll.sbatch
#
#SBATCH --job-name=heldout_nll
#SBATCH --time=00:45:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:4
#SBATCH -p booster
#SBATCH --account laionize
#SBATCH --output=/e/data1/mmlaion/lee27/snowball-sft/logs/heldout_nll_%j.log
set -uo pipefail
module load GCC/14.3.0
module load nvidia-compilers/25.9-CUDA-13
C=/e/project1/transfernetx/lee27/code
S=/e/data1/mmlaion/lee27/snowball-sft
PY=$C/envs/snowball-v2/bin/python
export PYTHONPATH=$C/src/marin_vllm_eagle3${PYTHONPATH:+:$PYTHONPATH}
MODEL=${MODEL:?hf export dir}
NAME=${NAME:?tag for the output file}
PARQUET=${PARQUET:-$S/data/kimi_swesmith_v1/heldout-00000-of-00001.parquet}
SCRIPT=${SCRIPT:-$C/snowball/heldout_nll.py}
[ -f "$MODEL/config.json" ] || { echo "no model at $MODEL"; exit 1; }
[ -f "$MODEL/chat_template.jinja" ] || { echo "no chat_template.jinja in $MODEL"; exit 1; }
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_ALLREDUCE_USE_SYMM_MEM=0
CACHE=/e/fscratch/reformo/lee27/cache; mkdir -p $CACHE/vllm $CACHE/xdg $CACHE/triton $CACHE/inductor $CACHE/flashinfer $CACHE/tmp
export VLLM_CACHE_ROOT=$CACHE/vllm XDG_CACHE_HOME=$CACHE/xdg TRITON_CACHE_DIR=$CACHE/triton TORCHINDUCTOR_CACHE_DIR=$CACHE/inductor FLASHINFER_WORKSPACE_BASE=$CACHE/flashinfer TMPDIR=$CACHE/tmp
export PATH=$C/envs/snowball-v2/bin:$PATH
echo "HELDOUT_START node=$(hostname) job=$SLURM_JOB_ID model=$MODEL name=$NAME"
SLOG=$S/logs/heldout_serve_$SLURM_JOB_ID.log
$PY -m vllm.entrypoints.openai.api_server --model "$MODEL" --served-model-name model --port 8000 \
    --tensor-parallel-size 1 --data-parallel-size 4 --enable-expert-parallel \
    --max-model-len 34816 --hf-overrides '{"max_position_embeddings": 65536, "max_seq_len": 65536}' \
    --max-num-seqs 16 --max-num-batched-tokens 4096 --gpu-memory-utilization 0.85 --max-logprobs 1 \
    --chat-template "$MODEL/chat_template.jinja" > $SLOG 2>&1 &
SPID=$!
for i in $(seq 1 180); do
  curl -sf localhost:8000/health >/dev/null && { echo "UP after $((i*10))s"; break; }
  kill -0 $SPID 2>/dev/null || { echo "server died"; tail -60 $SLOG; exit 1; }
  sleep 10
done
curl -sf localhost:8000/health >/dev/null || { echo "timeout waiting for the server"; kill $SPID; exit 1; }
trap 'kill $SPID 2>/dev/null' TERM EXIT
# smoke: one 3-token prompt with prompt_logprobs, so a protocol problem shows before 628 requests do
curl -s localhost:8000/v1/completions -H 'Content-Type: application/json' \
  -d '{"model":"model","prompt":[128000, 9906, 1917],"max_tokens":1,"temperature":0,"prompt_logprobs":0}' \
  | $PY -c 'import sys,json; c=json.load(sys.stdin)["choices"][0]; print("smoke prompt_logprobs:", c.get("prompt_logprobs"))' || echo "smoke parse failed"
$PY "$SCRIPT" --tokenizer-dir "$MODEL" --parquet "$PARQUET" --url http://localhost:8000 --served model \
  --out "$S/logs/heldout_nll_${NAME}.json" --concurrency 16 || { echo "scorer failed"; exit 2; }
echo "HELDOUT_DONE $S/logs/heldout_nll_${NAME}.json"
