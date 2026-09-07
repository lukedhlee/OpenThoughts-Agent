#!/bin/bash
# launch_easy3_probes.sh [sources...] — the six keep/drop probes on the three TaskTrove synthetic sources (2026-09-06).
# Run on the Jupiter login node AFTER the environment gate (pristine 0 / oracle 1 per source) and the adversarial audits.
# For each source: submit two 32-node JUWELS fleets on bridge 9926 (one per probe, 256 seats each), then the keep and the
# drop probe via probe_history.sh on the source's task tree. Trees: /e/fscratch/reformo/lee27/tasks/tt-easy3/<source>.
set -uo pipefail
C=/e/project1/transfernetx/lee27/code/snowball; T=/e/fscratch/reformo/lee27/tasks/tt-easy3
SRCS=("$@"); [ ${#SRCS[@]} -eq 0 ] && SRCS=(curriculum-easy pymethods2test-v3 unitsyn-python-v4)
FLEET_HOURS=${FLEET_HOURS:-10}
for s in "${SRCS[@]}"; do
  [ -d "$T/$s" ] || { echo "no task tree at $T/$s"; exit 1; }
  n=$(ls "$T/$s" | wc -l); echo "== $s: $n tasks"
  short=${s//-/}
  ssh -o ControlPath=~/.ssh/cm_juwels/bridge -o BatchMode=yes juwels "cd /p/project1/synthlaion/lee27/fleet && for x in k d; do sbatch --nodes=32 --time=${FLEET_HOURS}:00:00 --export=ALL,HARBOR_SRC=/p/project1/synthlaion/lee27/harbor/src,WORKERS_PER_NODE=8,STAGING_BASE=/tmp/apptainer_staging,BRIDGE_LOGIN=jwlogin03i,BRIDGE_PORT=9926 -J apptainer_workers_juwels_${short:0:12}\$x juwels_workers.sbatch; done" 2>&1 | tail -2
  for m in keep drop; do
    bash $C/probe_history.sh snowball_easy_${short}_${m}_base "$T/$s" base $m 2>&1 | tail -2
  done
done
squeue -u $USER --format="%.9i %.40j %.5T %.8M %.4D" | grep -i snowball_easy
