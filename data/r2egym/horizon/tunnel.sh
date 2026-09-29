#!/usr/bin/env bash
# tunnel.sh — give a Horizon job's nodes internet through the login node.
#
# Horizon compute nodes have no internet, and compute->login ssh asks for TACC TOTP, so the tunnel has to start on the
# login side: `ssh -R PORT node` opens a SOCKS5 proxy on the node's localhost:PORT whose traffic leaves through login1.
# Jobs then set ALL_PROXY=socks5h://127.0.0.1:PORT (curl, git, uv, cargo all honor it).
#
# Usage (on a Horizon login node, detached so it outlives your shell):
#   setsid nohup bash tunnel.sh <jobid> [PORT] > ~/tunnel.<jobid>.log 2>&1 &
# It waits for the job to start, keeps one reverse tunnel per node (re-opens a dropped one), and exits when the job ends.
set -uo pipefail
JOB=${1:?jobid}; PORT=${2:-18080}
SSH_OPTS=(-o BatchMode=yes -o StrictHostKeyChecking=no -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=3)
declare -A PID
while :; do
  state=$(squeue -h -j "$JOB" -o %T 2>/dev/null)
  case "$state" in
    RUNNING) ;;
    PENDING|CONFIGURING|"") [ -z "$state" ] && { echo "$(date -Is) job $JOB gone, exiting"; break; }; sleep 15; continue;;
    *) echo "$(date -Is) job $JOB state $state"; sleep 15; continue;;
  esac
  for n in $(scontrol show hostnames "$(squeue -h -j "$JOB" -o %N)"); do
    if [ -z "${PID[$n]:-}" ] || ! kill -0 "${PID[$n]}" 2>/dev/null; then
      ssh "${SSH_OPTS[@]}" -N -R "$PORT" "$n" & PID[$n]=$!
      echo "$(date -Is) tunnel -> $n:$PORT pid ${PID[$n]}"
    fi
  done
  sleep 20
done
for p in "${PID[@]}"; do kill "$p" 2>/dev/null; done
