#!/bin/bash
# jureca_bridge_setup.sh — reverse-forward a Jupiter apptainer bridge onto the JURECA login node's internal
# interface via the Jupiter->JURECA ControlMaster ~/.ssh/cm_jureca/qwen36. Idempotent; re-run after the master
# is re-created (a reboot of the Jupiter login node kills it; recreating it needs one TOTP from Luke).
#   jrlogin05i:9923 -> Jupiter :9922   (default; the fleet sbatch is submitted with BRIDGE_PORT=9923)
# Override with FORWARDS="remote:local ..." e.g. FORWARDS="9925:9924".
# Sibling of juwels_bridge2_setup.sh (2026-09-05); JURECA variant written 2026-09-09 after the login02 reboot.
set -u
SOCK=${SOCK:-~/.ssh/cm_jureca/qwen36}; SOCK=$(eval echo $SOCK)
HOST=${HOST:-jureca}; LOGIN=${LOGIN:-jrlogin05i}; FORWARDS=${FORWARDS:-"9923:9922"}
W="ssh -o BatchMode=yes -S $SOCK $HOST"
ssh -O check -S $SOCK $HOST 2>&1 | grep -q "Master running" || {
  echo "no master at $SOCK — Luke, on Jupiter: tmux new -A -s jureca_master ssh -N -M -S $SOCK $HOST (one TOTP)"; exit 1; }
IP=$($W "getent hosts $LOGIN" | awk '{print $1}')
[ -z "$IP" ] && { echo "$LOGIN did not resolve; interfaces:"; $W "hostname -I"; exit 1; }
echo "$LOGIN=$IP"
LISTEN=$($W "ss -ltn")
for f in $FORWARDS; do
  R=${f%%:*}; L=${f##*:}
  if echo "$LISTEN" | grep -q "$IP:$R "; then echo "forward $R->$L already present"; else
    ssh -O forward -R "${IP}:${R}:localhost:${L}" -S $SOCK $HOST && echo "forward $R->$L added"; fi
  $W "curl -s -m 8 http://$LOGIN:$R/status | head -c 160; echo \"  <- $LOGIN:$R\""
done
