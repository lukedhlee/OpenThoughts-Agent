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
echo "=== done $(date +%T) ==="
