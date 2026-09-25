#!/bin/bash
# stage_tree.sh [dest] (LIST=<task list>) — build a CalibForge task tree (default: the pilot's 100) on the Jupiter login node, from the public bundle.
#
# Source: laion/calibforge-daytona-layers bundle/task_tree.tar.gz (the cf1-20260925 tree: 2,457 Daytona-ready tasks,
# setup.sh fetching layers from the HF mirror), pinned by sha256. Extracts the tasks listed in relay100_tasks.txt,
# writes the router's task file (router_tasks.json: task_id, instruction, agent budget), and checks the count.
# CPU and network only; no Slurm.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
C=/e/project1/transfernetx/lee27/code
PY=${PY:-$C/envs/snowball-v2/bin/python}
DST=${1:-/e/fscratch/reformo/lee27/tasks/calibforge_relay100}
LIST=${LIST:-$HERE/relay100_tasks.txt}; N=$(grep -c . $LIST)
SHA=54e907f47434a3b731879cc1a6699aa16c24a5f9c0d0c8e522a69459852beafb
export OMP_NUM_THREADS=1
[ -e "$DST" ] && { echo "$DST exists; remove it first or pass another dest"; exit 1; }
TMP=$(mktemp -d /e/fscratch/reformo/lee27/tmp_stage.XXXXXX); trap 'rm -rf $TMP' EXIT
$PY - "$TMP" <<'PY'
import sys
from huggingface_hub import hf_hub_download
p = hf_hub_download('laion/calibforge-daytona-layers', 'bundle/task_tree.tar.gz', repo_type='dataset', local_dir=sys.argv[1])
print(p)
PY
TGZ=$TMP/bundle/task_tree.tar.gz
echo "$SHA  $TGZ" | sha256sum -c -
mkdir -p $TMP/x
sed 's#^#task_tree/#' $LIST > $TMP/members.txt
tar -xzf $TGZ -C $TMP/x -T $TMP/members.txt
mkdir -p "$(dirname $DST)"; mv $TMP/x/task_tree "$DST"
cp $LIST "$DST/TASKS.txt"
$PY $HERE/pilot_tasks.py router-json --tree "$DST" --list $LIST --out "$DST/router_tasks.json"
n=$(find "$DST" -mindepth 1 -maxdepth 1 -type d | wc -l)
[ "$n" = "$N" ] || { echo "expected $N task dirs in $DST, found $n"; exit 1; }
echo "STAGED $DST: $n tasks + router_tasks.json (bundle sha256 $SHA)"
