#!/bin/bash
# gate3_jupiter.sbatch — the Jupiter side of Horizon port gate 3: today's rerun of the reference acceptance (toolkit smoke
# job 1762082, 2026-09-11, 2.59) on the current serve env (envs/snowball-v2; the reference ran envs/snowball), same method
# as gate3_horizon.sbatch: tools/spec_decode/scripts/benchmark.py, toolkit serve DP4 EP, 32768, k=3, replay_b28k.jsonl,
# concurrency 16, 120 s, both arms on the public draft export (the reference's release dir). Module/env lines follow the
# reference smoke_retry1.sbatch. ~0.5 node-hours with the think probe; exits when done.
#
#   sbatch data/r2egym/horizon/gate3_jupiter.sbatch        (from a detached OTA worktree at the pushed SHA)
#SBATCH --job-name=sb_gate3
#SBATCH --time=00:50:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:4
#SBATCH -p booster
#SBATCH --account laionize
#SBATCH --output=/e/data1/mmlaion/lee27/horizon_gates/logs/%x_%j.out
#SBATCH --exclude=jpbo-031-[01-48],jpbo-011-[01-48],jpbo-038-38,jpbo-004-46,jpbo-065-17,jpbo-074-22,jpbo-074-40,jpbo-048-41,jpbo-091-05,jpbo-044-0[1-5]
set -uo pipefail
module load GCC/14.3.0
module load nvidia-compilers/25.9-CUDA-13
C=/e/project1/transfernetx/lee27/code
OTA=${OTA:-$C/ota-horizon-gates}
TK=${TOOLKIT:-$C/grugmoe-eagle3}   # github.com/lukedhlee/grugmoe-eagle3 @ 8b1558ba (tools/spec_decode is untracked in OTA)
PY=$C/envs/snowball-v2/bin/python
export PATH=$C/envs/snowball-v2/bin:$PATH
export PYTHONPATH=$C/src/marin_vllm_eagle3
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1   # the reference job's env
MODEL=${MODEL:-/e/fscratch/reformo/lee27/models/snowball-s3-nemotron-terminal-step1888}
DRAFT=${DRAFT:-/e/data1/mmlaion/lee27/eagle3/release_grug_eagle3_adapted_embedding_free_20260911}
REPLAY=${REPLAY:-$C/snowball/bench/replay_b28k.jsonl}
G=/e/data1/mmlaion/lee27/horizon_gates; mkdir -p $G/logs
echo "node=$(hostname) job=$SLURM_JOB_ID model=$MODEL draft=$DRAFT ota=$(git -C $OTA rev-parse --short HEAD)"
sha256sum $REPLAY $DRAFT/model.safetensors
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
$PY $TK/scripts/benchmark.py --target "$MODEL" --draft "$DRAFT" --baseline-draft "$DRAFT" \
    --replay $REPLAY --out $G/gate3_jupiter.$SLURM_JOB_ID --cache /e/fscratch/reformo/lee27/cache --python $PY \
    --max-model-len 32768 --seconds 120 --steady-after 30 --parity-turns 4 --repeats 1 > $G/logs/gate3_bench.$SLURM_JOB_ID.log 2>&1
RC=$?
$PY -c "import json,sys; r=json.load(open(sys.argv[1])); [print('GATE3 arm=%s acceptance=%.4f decode_tps_p50=%.1f buckets=%s' % (a['kind'], a['acceptance_length'], a['decode_tps_p50'], json.dumps(a['prompt_buckets']))) for a in r['arms']]" $G/gate3_jupiter.$SLURM_JOB_ID/results.json
# The smoke-35811 surprise ('<|start_think|>\n<|start_think|>' opening, draft on, temperature 1.0): one more server in the
# benchmark arm's exact config, then gate2_score.py think (target P of each opening token + 64 seeded samples). ~5 min.
if [ "${THINK:-1}" = 1 ]; then
  TLOG=$G/logs/gate3_think_server.$SLURM_JOB_ID.log
  setsid $PY $TK/toolkit.py serve --python $PY --target "$MODEL" --cache /e/fscratch/reformo/lee27/cache --dp 4 \
      --max-model-len 32768 --port 8000 --draft "$DRAFT" --k 3 --chat-template "$MODEL/chat_template.jinja" > $TLOG 2>&1 &
  TPID=$!
  trap 'kill -- -$TPID 2>/dev/null; pkill -u $USER -f vllm.entrypoints.openai.api_server 2>/dev/null' TERM EXIT
  for i in $(seq 1 120); do
    curl -sf localhost:8000/health >/dev/null && { echo "think server UP after $((i*10))s"; break; }
    kill -0 $TPID 2>/dev/null || { echo "think server died"; tail -30 $TLOG; break; }
    sleep 10
  done
  curl -sf localhost:8000/health >/dev/null && $PY $OTA/data/r2egym/horizon/gate2_score.py think --model target \
      --label ${SLURM_JOB_NAME}_$(hostname -s)_draft_on --out $G/think_draft_on.$SLURM_JOB_ID.json
fi
echo "=== DONE rc=$RC $(date -Is)"
