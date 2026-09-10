#!/bin/bash
# juwels_bridge2_setup.sh — reverse-forward the Jupiter apptainer bridges onto the JUWELS login node's internal
# interface via the Jupiter->JUWELS ControlMaster ~/.ssh/cm_juwels/bridge. Idempotent; re-run after the master is re-created.
#   jwlogin03i:9926 -> Jupiter :9926 (Snowball RL pool, bridge #2)
#   jwlogin03i:9925 -> Jupiter :9924 (TaskTrove pool; same port JURECA uses for that bridge)
# Override with FORWARDS="remote:local ..." (default "9926:9926 9925:9924").
set -u
SOCK=${SOCK:-~/.ssh/cm_juwels/bridge}; SOCK=$(eval echo $SOCK); LOGIN=${LOGIN:-jwlogin03i}; FORWARDS=${FORWARDS:-"9926:9926 9925:9924"}
W="ssh -o BatchMode=yes -S $SOCK juwels"
ssh -O check -S $SOCK juwels 2>&1 | grep -q "Master running" || { echo "no master at $SOCK (Luke: ssh -M -S $SOCK -N -f juwels on Jupiter, one TOTP)"; exit 1; }
IP=$($W "getent hosts $LOGIN" | awk '{print $1}')
if [ -z "$IP" ]; then echo "$LOGIN did not resolve; interfaces:"; $W "hostname -I; getent hosts jwlogin03 jwlogin03i.juwels 2>/dev/null"; exit 1; fi
echo "$LOGIN=$IP"
LISTEN=$($W "ss -ltn")
for f in $FORWARDS; do
  R=${f%%:*}; L=${f##*:}
  if echo "$LISTEN" | grep -q "$IP:$R "; then echo "forward $R->$L already present"; else ssh -O forward -R "${IP}:${R}:localhost:${L}" -S $SOCK juwels && echo "forward $R->$L added"; fi
  $W "curl -s -m 8 http://$LOGIN:$R/status | head -c 120; echo \"  <- $LOGIN:$R\""
done
