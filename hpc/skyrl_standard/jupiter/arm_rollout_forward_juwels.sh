#!/usr/bin/env bash
# JUWELS variant of arm_rollout_forward.sh: same contract, different fabric.
#   CM socket  ~/.ssh/cm_juwels/band  ->  lee27@juwels-cluster.fz-juelich.de (jwlogin08)
#   bind IP    10.13.0.158 (jwlogin08i) -- workers/agents on JUWELS computes hit this
#   probe      srun --overlap inside the JUWELS fleet job, via the same CM
# State file is separate from the JURECA one so cancels never cross fabrics.
#
# 2026-08-10 fixes (smoke4/smoke6 postmortems):
#   - job log path is DISCOVERED from the jobid (or passed as arg 4), no longer
#     derived from the port via the band512r_s* naming (that derivation pointed
#     at a nonexistent log and FATAL'd without installing the forward).
#   - GENERATE-mode jobs (main_tbench_generate) have NO weight sync — the
#     sync-marker wait is skipped for them (it can never pass; the meta-reload
#     window it guards does not exist on that path).
set -uo pipefail

JOB="${1:?usage: arm_rollout_forward_juwels.sh <jupiter_jobid> [port] [juwels_fleet_jobid] [job_log]}"
PORT="${2:-18300}"
S="$HOME/.ssh/cm_juwels/band"
H="lee27@juwels-cluster.fz-juelich.de"
BIND="10.13.0.158"
STATE="$HOME/.rollout_forwards_juwels"
LOG="$HOME/arm_forward_juwels_${JOB}.log"
exec >>"$LOG" 2>&1

echo "=== armed $(date '+%F %T') job=$JOB port=$PORT bind=$BIND ==="

if [[ -f "$STATE" ]]; then
  while read -r p ip; do
    [[ "$p" == "$PORT" && -n "${ip:-}" ]] || continue
    if ssh -S "$S" -O cancel -R "${BIND}:${p}:${ip}:8000" "$H" 2>/dev/null; then
      echo "  cancelled recorded forward ${p} -> ${ip}"
    fi
  done < "$STATE"
  grep -v "^${PORT} " "$STATE" > "${STATE}.tmp" 2>/dev/null || true
  mv -f "${STATE}.tmp" "$STATE" 2>/dev/null || true
fi

if ssh -o BatchMode=yes -S "$S" "$H" "ss -ltn" 2>/dev/null | grep -q "${BIND//./\\.}:${PORT}\b"; then
  echo "FATAL: ${BIND}:${PORT} is still bound and not in $STATE."
  exit 1
fi

while :; do
  ST=$(squeue -j "$JOB" -h -o "%T" 2>/dev/null)
  [[ -z "$ST" ]] && { echo "job $JOB left the queue before starting"; exit 1; }
  [[ "$ST" == "RUNNING" ]] && break
  sleep 30
done

NL=$(squeue -j "$JOB" -h -o "%N")
NODE=$(scontrol show hostnames "$NL" | head -1)
IP=$(getent hosts "$NODE" | cut -d' ' -f1)
echo "$(date +%T) RUNNING nodelist=$NL head=$NODE ip=$IP"
[[ -n "$IP" ]] || { echo "FATAL: could not resolve $NODE"; exit 1; }

# Locate the job log: arg 4 wins; otherwise discover by jobid under the
# experiments tree. Retry while the job bootstraps (log appears ~1 min in).
JOBLOG="${4:-}"
if [[ -z "$JOBLOG" ]]; then
  for t in $(seq 1 20); do
    JOBLOG=$(ls /e/fscratch/reformo/lee27/experiments/*/logs/*_${JOB}.out 2>/dev/null | head -1)
    [[ -n "$JOBLOG" ]] && break
    sleep 15
  done
fi
[[ -n "$JOBLOG" ]] || { echo "FATAL: no job log found for $JOB under experiments/*/logs"; exit 1; }
echo "  job log: $JOBLOG"

# Wait for vLLM to actually listen on the head before installing + probing;
# probing a not-yet-booted server would look identical to a dead route.
for t in $(seq 1 90); do
  if curl -s -m 3 -o /dev/null "http://${IP}:8000/v1/models"; then
    echo "$(date +%T) vLLM up on ${IP}:8000 (after ~$((t*30))s)"
    break
  fi
  if [[ "$t" == "90" ]]; then
    echo "FATAL: vLLM never came up on ${IP}:8000 within 45 min; check the RL job log"
    exit 1
  fi
  sleep 30
done

# CRITICAL ORDERING (2026-08-08 meta-crash root cause): on the TRAINER path, do
# NOT install the forward until the trainer's INITIAL weight sync has fully
# completed — a request reaching an engine inside the layerwise reload bracket
# kills that EngineCore permanently. GENERATE-mode jobs have no weight sync and
# no reload bracket; the marker never appears there, so skip the wait.
if grep -q "main_tbench_generate" "$JOBLOG" 2>/dev/null; then
  echo "$(date +%T) generate-mode job detected — no weight sync; skipping sync-marker wait"
else
  for t in $(seq 1 60); do
    if grep -q "Finished: 'sync_weights_to_inference_engines'" "$JOBLOG" 2>/dev/null; then
      echo "$(date +%T) initial weight sync complete (log marker found)"
      break
    fi
    if [[ "$t" == "60" ]]; then
      echo "FATAL: initial sync marker never appeared in $JOBLOG within 30 min"
      exit 1
    fi
    sleep 30
  done
  sleep 30  # margin past the bracket before exposing the engines
fi

if ssh -S "$S" -O forward -R "${BIND}:${PORT}:${IP}:8000" "$H" 2>&1 | sed 's/^/  forward: /'; then :; fi
echo "${PORT} ${IP}" >> "$STATE"
echo "  recorded in $STATE"

echo "--- listeners on JUWELS login (expect 9923 + ${PORT}):"
ssh -o BatchMode=yes -S "$S" "$H" 'ss -ltn | grep -E "992[0-9]|18[0-9][0-9][0-9]"'

FLEET="${3:-}"
if [[ -z "$FLEET" ]]; then
  echo "WARNING: no fleet jobid given (arg 3) -- route NOT verified end-to-end."
  echo "=== done $(date +%T) (UNVERIFIED) ==="
  exit 0
fi

echo "--- verifying route from a compute node of fleet $FLEET:"
PROBE=$(ssh -o BatchMode=yes -S "$S" "$H" \
  "srun --jobid=$FLEET --overlap -N1 -n1 --time=4 curl -s -m 15 -o /dev/null \
     -w 'HTTP=%{http_code}' http://${BIND}:${PORT}/v1/chat/completions \
     -H 'Content-Type: application/json' \
     -d '{\"model\":\"x\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":1}'" \
  2>&1 | tr -d '\r' | grep -o 'HTTP=[0-9]*' | tail -1)
echo "  probe result: ${PROBE:-<none>}"

case "$PROBE" in
  HTTP=000|"")
    echo "FATAL: route is DEAD from the compute nodes (${PROBE:-no response})."
    echo "       Exact cancel + reinstall:"
    echo "         ssh -S $S -O cancel  -R ${BIND}:${PORT}:${IP}:8000 $H"
    echo "         ssh -S $S -O forward -R ${BIND}:${PORT}:${IP}:8000 $H"
    exit 1 ;;
  *)
    echo "  ROUTE OK (any HTTP status counts; 400 model-mismatch is a PASS)"
    echo "=== done $(date +%T) VERIFIED ==="
    ;;
esac
