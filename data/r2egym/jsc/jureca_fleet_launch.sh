#!/bin/bash
# jureca_fleet_launch.sh — submit N apptainer sandbox fleets on JURECA (dc-cpu) that register on a Jupiter bridge,
# through the Jupiter->JURECA ControlMaster. Run FROM the Jupiter login node that hosts the bridge.
#   usage: [NODES=48] [FLEETS=2] [PORT=9923] [TIME=24:00:00] [TAG=v3] bash jureca_fleet_launch.sh
# Every env the sbatch reads must be passed explicitly (--export=ALL): an `ssh … sbatch` shell carries none
# (JUWELS pool9 died in 8 s on HARBOR_SRC unset, 2026-09-05). Verify the .out header within a minute of the start.
set -u
SOCK=${SOCK:-~/.ssh/cm_jureca/qwen36}; SOCK=$(eval echo $SOCK); HOST=${HOST:-jureca}
NODES=${NODES:-48}; FLEETS=${FLEETS:-2}; PORT=${PORT:-9923}; LOGIN=${LOGIN:-jrlogin05i}
TIME=${TIME:-24:00:00}; TAG=${TAG:-v3}; WPN=${WPN:-32}
SRC=/p/project1/synthlaion/lee27/harbor/src
DIR=$SRC/harbor/environments/apptainer
W="ssh -o BatchMode=yes -S $SOCK $HOST"
ssh -O check -S $SOCK $HOST 2>&1 | grep -q "Master running" || { echo "no master at $SOCK (needs Luke's TOTP)"; exit 1; }
echo "dc-cpu availability:"; $W "sinfo -p dc-cpu -o '%.10P %.6D %.6t'"
for i in $(seq 1 $FLEETS); do
  n=$(printf "%s%s" "$TAG" "$i")
  id=$($W "cd $DIR && sbatch --nodes=$NODES --time=$TIME -J apptainer_workers_jureca_$n \
    --export=ALL,HARBOR_SRC=$SRC,WORKERS_PER_NODE=$WPN,STAGING_BASE=/dev/shm/lee27/apptainer_staging,\
APPTAINER_CACHEDIR_BASE=/dev/shm/lee27/apptainer_cache,BRIDGE_LOGIN=$LOGIN,BRIDGE_PORT=$PORT \
    jureca_workers.sbatch" | grep -o '[0-9]*')
  echo "fleet $n -> job $id (nodes=$NODES, $WPN workers/node, bridge $LOGIN:$PORT)"
done
$W "squeue -u \$USER -o '%.10i %.40j %.8T %.10M %.6D %R'"
