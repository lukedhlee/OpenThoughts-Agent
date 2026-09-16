#!/bin/bash
# One node of the Daytona capacity run: loopback gateway + single-node vLLM (DP4, EAGLE-3) + two
# 64-seat harbor coordinators. Launched by scale_1024.sbatch through srun, one task per node.
set -uo pipefail
X=$SCALE_X
T=$SCALE_T
V=/e/project1/transfernetx/lee27/code/envs/snowball-v2
M=/e/fscratch/reformo/lee27/models/snowball-s3-nemotron-terminal-step1888
RUN=$X/run_$SLURM_JOB_ID/node_$SLURM_PROCID
mkdir -p "$RUN"
hostname > "$RUN/host"
gateway='' server='' c0='' c1=''
cleanup() { for p in $c0 $c1 $server $gateway; do [[ -n "$p" ]] && kill "$p" 2>/dev/null; done; }
trap cleanup EXIT
"$V/bin/python" -u "$X/async_socks_connect_proxy.py" --ready "$RUN/gateway-ready" > "$RUN/gateway.log" 2>&1 &
gateway=$!
for i in $(seq 1 60); do [[ -f "$RUN/gateway-ready" ]] && break; sleep 1; done
[[ -f "$RUN/gateway-ready" ]] || { echo gateway > "$RUN/FAILED"; exit 1; }
"$V/bin/python" -u -m vllm.entrypoints.openai.api_server --model "$M" --served-model-name snowball --host 127.0.0.1 --port 8000 \
  --max-model-len 65536 --gpu-memory-utilization .85 --max-num-seqs 32 --max-num-batched-tokens 16384 \
  --max-cudagraph-capture-size 64 --enable-prefix-caching --enable-chunked-prefill --async-scheduling \
  --tensor-parallel-size 1 --data-parallel-size 4 --enable-expert-parallel \
  --chat-template "$M/chat_template.jinja" --no-enable-log-requests \
  --hf-overrides '{"max_position_embeddings":65536,"max_seq_len":65536}' \
  --speculative-config '{"method":"eagle3","model":"/e/data1/mmlaion/lee27/eagle3/probe_adapt_20260911/checkpoints/3","num_speculative_tokens":3}' \
  > "$RUN/server.log" 2>&1 &
server=$!
UP=0
for i in $(seq 1 90); do
  if curl -fsS --max-time 3 http://127.0.0.1:8000/health >/dev/null 2>&1; then UP=1; break; fi
  kill -0 "$server" 2>/dev/null || break
  sleep 10
done
[[ "$UP" == 1 ]] || { echo server > "$RUN/FAILED"; tail -30 "$RUN/server.log"; exit 1; }
date +%s > "$RUN/server-up"
S=$(( (SLURM_PROCID + ${SCALE_NODE_OFFSET:-0}) * 2 ))
EXTRA=${SCALE_PROBE_FLAGS:-}
"$V/bin/python" -u "$X/scale_rollout_probe.py" --tasks "$X/slices/slice_$S.json" --task-root "$T" --out "$RUN" --name c0 $EXTRA > "$RUN/c0.log" 2>&1 &
c0=$!
"$V/bin/python" -u "$X/scale_rollout_probe.py" --tasks "$X/slices/slice_$((S + 1)).json" --task-root "$T" --out "$RUN" --name c1 $EXTRA > "$RUN/c1.log" 2>&1 &
c1=$!
wait "$c0"; r0=$?
wait "$c1"; r1=$?
c0=''; c1=''
echo "$r0 $r1" > "$RUN/DONE"
exit 0
