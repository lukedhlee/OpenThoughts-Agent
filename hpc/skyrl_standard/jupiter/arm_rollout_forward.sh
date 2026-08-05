#!/usr/bin/env bash
# Arm the rollout reverse-forward for a PENDING Jupiter job, and record what it
# installed so the NEXT run can cancel it exactly.
#
# Run ON a Jupiter login node (the JURECA ControlMaster socket lives there, and only a
# Jupiter client can reach <head>:8000). Launch it detached, e.g.
#   tmux new-session -d -s armfwd "bash arm_rollout_forward.sh 1251403 18300"
#
# WHY THE STATE FILE EXISTS
# `ssh -O cancel -R` matches on the FULL spec including the CONNECT address, not just
# the listen port. A wildcard (0.0.0.0) or a guessed IP silently no-ops and returns
# success. On 2026-08-05 that left port 18300 pointing at a dead head node three
# separate times; the symptom is an empty agent/ dir with a perfectly healthy bridge,
# and it only surfaces 35 minutes later at BRIDGE_EXEC_TIMEOUT. Job 1246853 died to it.
#
# So: every install appends "<port> <ip>" to $STATE, and startup cancels whatever is
# recorded there before installing anything new. No guessing.
set -uo pipefail

JOB="${1:?usage: arm_rollout_forward.sh <jobid> [port]}"
PORT="${2:-18300}"
S="$HOME/.ssh/cm_jureca/qwen36"
H="jureca05.fz-juelich.de"
STATE="$HOME/.rollout_forwards"
LOG="$HOME/arm_forward_${JOB}.log"
exec >>"$LOG" 2>&1

echo "=== armed $(date '+%F %T') job=$JOB port=$PORT ==="

# 1. Drop every forward we have ever recorded on this port, using its EXACT spec.
if [[ -f "$STATE" ]]; then
  while read -r p ip; do
    [[ "$p" == "$PORT" && -n "${ip:-}" ]] || continue
    if ssh -S "$S" -O cancel -R "10.14.0.46:${p}:${ip}:8000" "$H" 2>/dev/null; then
      echo "  cancelled recorded forward ${p} -> ${ip}"
    fi
  done < "$STATE"
  grep -v "^${PORT} " "$STATE" > "${STATE}.tmp" 2>/dev/null || true
  mv -f "${STATE}.tmp" "$STATE" 2>/dev/null || true
fi

# 2. Refuse to proceed if the port is still bound -- installing over it fails and we
#    would silently keep routing at the old, dead head.
if ssh -o BatchMode=yes -S "$S" "$H" "ss -ltn" 2>/dev/null | grep -q "10\.14\.0\.46:${PORT}\b"; then
  echo "FATAL: 10.14.0.46:${PORT} is still bound and not in $STATE."
  echo "       Find the owning job's head IP and cancel with the exact spec:"
  echo "         N=\$(sacct -j <jobid> -X -o NodeList%40 -n | tr -d ' ')"
  echo "         H=\$(scontrol show hostnames \"\$N\" | head -1); IP=\$(getent hosts \"\$H\" | cut -d' ' -f1)"
  echo "         ssh -S $S -O cancel -R 10.14.0.46:${PORT}:\$IP:8000 $H"
  exit 1
fi

# 3. Wait for the allocation.
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

if ssh -S "$S" -O forward -R "10.14.0.46:${PORT}:${IP}:8000" "$H" 2>&1 | sed 's/^/  forward: /'; then :; fi
echo "${PORT} ${IP}" >> "$STATE"
echo "  recorded in $STATE"

echo "--- listeners on JURECA (expect 9923 + ${PORT}):"
ssh -o BatchMode=yes -S "$S" "$H" 'ss -ltn | grep -E "992[0-9]|18[0-9][0-9][0-9]"'

# 4. VERIFY THE ROUTE END-TO-END, FROM A COMPUTE NODE. Nothing above proves it.
#
# 2026-08-06, job 1251403: this script logged a clean install at 18:25 and
# `ss` showed 10.14.0.46:18300 LISTENING -- yet every one of the 64 rollouts sat
# dead from 18:38 to 18:59 because a curl from a fleet compute node returned
# HTTP=000 in 1.5ms (refused). A cancel + reinstall of the BYTE-IDENTICAL spec
# fixed it instantly. So neither this script's own success line nor a listener
# check is evidence of a working route: `ssh -R` binds the listener regardless of
# whether the forwarded-to target is reachable, and a stale registration on the
# same port is indistinguishable by `ss`.
#
# The only trustworthy probe is a real HTTP request from a JURECA COMPUTE node --
# the same vantage point the sandboxes have. Pass the fleet jobid to enable it.
# A model-name-mismatch 400 is a PASS: it proves we reached the real vLLM server.
FLEET="${3:-}"
if [[ -z "$FLEET" ]]; then
  echo "WARNING: no fleet jobid given (arg 3) -- route NOT verified end-to-end."
  echo "         Listener presence is NOT evidence. Re-run as:"
  echo "           arm_rollout_forward.sh $JOB $PORT <fleet_jobid>"
  echo "=== done $(date +%T) (UNVERIFIED) ==="
  exit 0
fi

echo "--- verifying route from a compute node of fleet $FLEET:"
PROBE=$(ssh -o BatchMode=yes -S "$S" "$H" \
  "srun --jobid=$FLEET --overlap -N1 -n1 --time=4 curl -s -m 15 -o /dev/null \
     -w 'HTTP=%{http_code}' http://10.14.0.46:${PORT}/v1/chat/completions \
     -H 'Content-Type: application/json' \
     -d '{\"model\":\"x\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":1}'" \
  2>&1 | tr -d '\r' | grep -o 'HTTP=[0-9]*' | tail -1)
echo "  probe result: ${PROBE:-<none>}"

case "$PROBE" in
  HTTP=000|"")
    echo "FATAL: route is DEAD from the compute nodes (${PROBE:-no response}) even though"
    echo "       the listener is bound. The sandboxes would all hit BRIDGE_EXEC_TIMEOUT"
    echo "       ~35 min from now and the run would produce nothing."
    echo "       Try an exact cancel + reinstall (this fixed it on 1251403):"
    echo "         ssh -S $S -O cancel  -R 10.14.0.46:${PORT}:${IP}:8000 $H"
    echo "         ssh -S $S -O forward -R 10.14.0.46:${PORT}:${IP}:8000 $H"
    exit 1 ;;
  *)
    echo "  ROUTE OK (any HTTP status from the server counts; 400 model-mismatch is a PASS)"
    echo "=== done $(date +%T) VERIFIED ==="
    ;;
esac
