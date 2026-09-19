#!/bin/bash
# ifv2_apptainer.sbatch — serve_snowball.sbatch + the apptainer sandbox bridge and workers ON THE SAME NODE, so a
#   harness probe (the 477 held-out if-v2 tasks, ifv2_policy.yaml) needs no Daytona snapshot and no CPU fleet.
#   The bridge is harbor's stdlib HTTP server, the workers run apptainer instances from the SIF cache
#   (/e/scratch/reformo/lee27/sif; build_ifv2-dc02d52d2529.sif matches the tasks' shared Dockerfile hash).
#   Writes $ENDPOINT_DIR/<jobid>.bridge next to the endpoint file; run_tb2.sh reads both.
#   MODEL=<hf dir> sbatch ifv2_apptainer.sbatch
#
# The marin vLLM fork with the adapted EAGLE-3 draft (Luke's 2026-09-12 policy: every job uses the draft), the arm's
# geometry (TP1 x DP4 x EP4, 65k context), and marin's eval-policy sampler (#8151: temperature 0.7, top_p 1.0,
# repetition_penalty 1.1) applied server-side, so harbor's terminus-2 sends no sampling parameters of its own.
# Writes the endpoint URL to $ENDPOINT_DIR/<jobid> for run_tb2.sh on the login node, then holds the server up
# until the wall ends. One server per harbor run; the run reads the endpoint file.
#
#   MODEL=<hf-format dir> POLICY=marin|trained sbatch serve_snowball.sbatch
#
#SBATCH --job-name=ifv2_serve
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --gres=gpu:4
#SBATCH -p booster
#SBATCH --account laionize   # 2026-09-17: reformo is 18 % left and queues ~6 h on fairshare; laionize starts in minutes
#SBATCH --output=/e/fscratch/reformo/lee27/experiments/tb2/logs/%x_%j.out
#SBATCH --exclude=jpbo-031-[01-48],jpbo-011-[01-48],jpbo-038-38,jpbo-004-46,jpbo-065-17,jpbo-074-22,jpbo-074-40,jpbo-048-41,jpbo-091-05,jpbo-044-0[1-5]
set -uo pipefail
module load GCC/14.3.0
module load nvidia-compilers/25.9-CUDA-13
C=/e/project1/transfernetx/lee27/code
PY=$C/envs/snowball-v2/bin/python
export PYTHONPATH=$C/src/marin_vllm_eagle3${PYTHONPATH:+:$PYTHONPATH}   # the EAGLE-3 overlay shadows the venv's vllm
MODEL=${MODEL:-/e/fscratch/reformo/lee27/models/snowball-s3-nemotron-terminal-step1888}
DRAFT=${DRAFT:-/e/data1/mmlaion/lee27/eagle3/probe_adapt_20260911/checkpoints/3}
POLICY=${POLICY:-trained}   # v0.1 = no sampler override = vLLM defaults (temp 1.0); 'marin' is the August #8151 sampler, not the current policy
SERVED=${SERVED:-snowball}
E=/e/fscratch/reformo/lee27/experiments/tb2; ENDPOINT_DIR=$E/endpoints; LOGS=$E/logs; mkdir -p $ENDPOINT_DIR $LOGS
case "$POLICY" in
  marin)   GEN='{"temperature":0.7,"top_p":1.0,"repetition_penalty":1.1}';;   # marin eval policy #8151, shared decoding rules
  trained) GEN='{"temperature":1.0,"top_p":1.0,"top_k":-1}';;                 # the RL arm's sampler
  *) echo "POLICY must be marin or trained"; exit 2;;
esac
[ -f "$MODEL/config.json" ] || { echo "no model at $MODEL"; exit 1; }
[ -f "$DRAFT/model.safetensors" ] || { echo "no draft at $DRAFT"; exit 1; }
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_ALLREDUCE_USE_SYMM_MEM=0
CACHE=/e/fscratch/reformo/lee27/cache; mkdir -p $CACHE/vllm $CACHE/xdg $CACHE/triton $CACHE/inductor $CACHE/flashinfer $CACHE/tmp
export VLLM_CACHE_ROOT=$CACHE/vllm XDG_CACHE_HOME=$CACHE/xdg TRITON_CACHE_DIR=$CACHE/triton TORCHINDUCTOR_CACHE_DIR=$CACHE/inductor FLASHINFER_WORKSPACE_BASE=$CACHE/flashinfer TMPDIR=$CACHE/tmp
export PATH=$C/envs/snowball-v2/bin:$PATH   # ninja for FlashInfer JIT
echo "node=$(hostname) job=$SLURM_JOB_ID model=$MODEL policy=$POLICY gen=$GEN"
nvidia-smi --query-gpu=name,memory.total --format=csv
$PY -c "import vllm,torch;print('vllm',vllm.__file__,vllm.__version__,'torch',torch.__version__)"
CHAT=(); [ -f "$MODEL/chat_template.jinja" ] && CHAT=(--chat-template "$MODEL/chat_template.jinja")
SLOG=$LOGS/serve_$SLURM_JOB_ID.log
$PY -m vllm.entrypoints.openai.api_server --model "$MODEL" --served-model-name "$SERVED" --port 8000 \
    --tensor-parallel-size 1 --data-parallel-size 4 --enable-expert-parallel \
    --max-model-len 65536 --hf-overrides '{"max_position_embeddings": 65536, "max_seq_len": 65536}' \
    --max-num-seqs 32 --gpu-memory-utilization 0.90 \
    --speculative-config "{\"method\":\"eagle3\",\"model\":\"$DRAFT\",\"num_speculative_tokens\":3}" \
    --override-generation-config "$GEN" "${CHAT[@]}" > $SLOG 2>&1 &
SPID=$!
for i in $(seq 1 180); do
  curl -sf localhost:8000/health >/dev/null && { echo "UP after $((i*10))s"; break; }
  kill -0 $SPID 2>/dev/null || { echo "server died"; tail -60 $SLOG; exit 1; }
  sleep 10
done
curl -sf localhost:8000/health >/dev/null || { echo "timeout waiting for the server"; kill $SPID; exit 1; }
grep -m1 -E "Initializing a V1 LLM engine" $SLOG || true
# one real completion before anyone points harbor at it: the served name, the think markers surviving, usage counts
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d "{\"model\":\"$SERVED\",\"messages\":[{\"role\":\"user\",\"content\":\"Print hello in bash.\"}],\"max_tokens\":400,\"skip_special_tokens\":false}" \
  | $PY -c 'import json,sys; r=json.load(sys.stdin); m=r["choices"][0]["message"]; print("SMOKE finish=%s usage=%s content=%r" % (r["choices"][0]["finish_reason"], r.get("usage"), (m.get("content") or "")[:300]))' | tee -a $SLOG
URL="http://$(hostname -s).jupiter.internal:8000/v1"
echo "$URL" > $ENDPOINT_DIR/$SLURM_JOB_ID
printf 'model=%s\npolicy=%s\ngen=%s\ndraft=%s\nserved=%s\n' "$MODEL" "$POLICY" "$GEN" "$DRAFT" "$SERVED" > $ENDPOINT_DIR/$SLURM_JOB_ID.meta
echo "ENDPOINT $URL (file $ENDPOINT_DIR/$SLURM_JOB_ID)"
# --- apptainer bridge + workers on this node. The bridge comes from harbor-marin (lukedhlee/snowball-r2egym), NOT the
# v0.1 pin: upstream 7b18505a's ThreadedHTTPServer keeps Python's listen backlog of 5, so the login node's first
# connect gets refused while the dispatcher is polling (smokes 2 and 3 on 2026-09-19); the fork honours
# BRIDGE_LISTEN_BACKLOG / BRIDGE_MAX_HANDLER_THREADS. The harbor client side stays on the v0.1 pin (run_tb2.sh). ---
APT=/e/project1/transfernetx/lee27/code/harbor-marin/src/harbor/environments/apptainer
SIF=/e/scratch/reformo/lee27/sif; STG=/tmp/lee27_ifv2_staging; mkdir -p $STG
export BRIDGE_CLOSE_AFTER_RESPONSE=1 BRIDGE_MAX_HANDLER_THREADS=256 BRIDGE_LISTEN_BACKLOG=1024
BPORT=${BPORT:-9930}
# the worker must NOT inherit vLLM's cache/temp variables (they point at fscratch, which is full): apptainer uses
# TMPDIR for its own session files and the first smoke lost every tmux server between two execs
CLEAN=(env -u TMPDIR -u XDG_CACHE_HOME -u TRITON_CACHE_DIR -u TORCHINDUCTOR_CACHE_DIR -u FLASHINFER_WORKSPACE_BASE -u VLLM_CACHE_ROOT -u PYTHONPATH)
"${CLEAN[@]}" $PY -u $APT/server.py --port $BPORT --host 0.0.0.0 > $LOGS/bridge_$SLURM_JOB_ID.log 2>&1 &
BPID=$!
sleep 3
"${CLEAN[@]}" $PY -u $APT/worker.py --bridge-url http://127.0.0.1:$BPORT --sif-cache $SIF --staging-base $STG --num-workers ${NWORKERS:-40} \
    > $LOGS/worker_$SLURM_JOB_ID.log 2>&1 &
WPID=$!
BURL="http://$(hostname -s).jupiter.internal:$BPORT"
sleep 5
# readiness = a real request answered over the node's external address, not just a bound port
for i in $(seq 1 12); do curl -s --max-time 5 -X POST -H 'Content-Type: application/json' -d '{}' $BURL/env/list >/dev/null 2>&1 && break; sleep 5; done
curl -s --max-time 5 -X POST -H 'Content-Type: application/json' -d '{}' $BURL/env/list | head -c 200; echo
kill -0 $BPID || { echo "bridge died"; tail -20 $LOGS/bridge_$SLURM_JOB_ID.log; }
kill -0 $WPID || { echo "worker died"; tail -20 $LOGS/worker_$SLURM_JOB_ID.log; }
# liveness monitor: the exact second the bridge or the worker disappears goes to this log
( while sleep 5; do kill -0 $BPID 2>/dev/null || { echo "$(date +%T) BRIDGE_DIED"; break; }; done ) &
( while sleep 5; do kill -0 $WPID 2>/dev/null || { echo "$(date +%T) WORKER_DIED"; break; }; done ) &
echo "$(date +%T) bridge+worker up"
echo "$BURL" > $ENDPOINT_DIR/$SLURM_JOB_ID.bridge
echo "BRIDGE $BURL (workers ${NWORKERS:-40}, sif $SIF, staging $STG)"
trap 'trap - TERM EXIT; echo "$(date +%T) wall/term: stopping server + bridge"; kill $SPID $WPID $BPID 2>/dev/null; rm -rf $STG; rm -f $ENDPOINT_DIR/$SLURM_JOB_ID $ENDPOINT_DIR/$SLURM_JOB_ID.bridge; exit 0' TERM EXIT
wait $SPID
