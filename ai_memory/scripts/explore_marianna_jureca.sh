#!/usr/bin/env bash
# Paste on JURECA login node (or: ssh jureca 'bash -s' < thisfile after ControlMaster is up)
set -euo pipefail

echo "=== who / host ==="
whoami; hostname; date

echo ""
echo "=== Marianna launch script ==="
ls -la /p/project1/laionize/marianna/dc_agent/bash-scripts/run_full_r2egym_filter_jureca.sh

echo ""
echo "=== dc_agent tree (depth 2) ==="
ls -la /p/project1/laionize/marianna/dc_agent/ 2>&1 | head -50
find /p/project1/laionize/marianna/dc_agent -maxdepth 2 -type d 2>/dev/null | head -80

echo ""
echo "=== script header (paths/exports) ==="
head -n 120 /p/project1/laionize/marianna/dc_agent/bash-scripts/run_full_r2egym_filter_jureca.sh

echo ""
echo "=== sibling bash-scripts ==="
ls -la /p/project1/laionize/marianna/dc_agent/bash-scripts/ 2>&1 | head -60

echo ""
echo "=== harbor_patched / clones? ==="
ls -la /p/project1/laionize/marianna/dc_agent/harbor_patched 2>&1 | head -20
for d in harbor SkyRL skyrl OpenThoughts-Agent dc-agent dc_agent; do
  p=$(find /p/project1/laionize/marianna -maxdepth 3 -type d -name "$d" 2>/dev/null | head -3)
  [ -n "$p" ] && echo "FOUND $d:" && echo "$p"
done

echo ""
echo "=== git remotes/branches in likely clones ==="
for repo in \
  /p/project1/laionize/marianna/dc_agent \
  /p/project1/laionize/marianna/dc_agent/harbor_patched \
  /p/project1/laionize/marianna/harbor \
  /p/project1/laionize/marianna/SkyRL
do
  if [ -d "$repo/.git" ] || [ -e "$repo/.git" ]; then
    echo "--- $repo ---"
    git -C "$repo" remote -v 2>/dev/null | head -5
    git -C "$repo" branch -vv 2>/dev/null | head -10
    git -C "$repo" rev-parse --short HEAD 2>/dev/null
  fi
done

echo ""
echo "=== Marianna conda env ==="
ls -la /p/project1/ccstdl/envs/marianna/py3.12/bin/python
/p/project1/ccstdl/envs/marianna/py3.12/bin/python -V
/p/project1/ccstdl/envs/marianna/py3.12/bin/python -c "import sys; print(sys.executable)"
/p/project1/ccstdl/envs/marianna/py3.12/bin/pip list 2>/dev/null | rg -i 'harbor|skyrl|vllm|torch|daytona|ray' || true

echo ""
echo "=== scratch / sif / staging hints from env or common paths ==="
ls -lad /p/scratch/laionize /p/scratch/synthlaion /p/scratch/ccstdl 2>&1 | head -20
ls -lad /p/scratch/*/marianna 2>/dev/null | head -20
ls -lad /p/project1/laionize/marianna/*/sif* /p/scratch/*/marianna/*sif* 2>/dev/null | head -20

echo ""
echo "=== your projects / quotas ==="
jutil user projects 2>/dev/null | head -40 || true
for p in ccstdl laionize synthlaion westai0066; do
  echo "-- quota $p --"
  jutil project dataquota -p "$p" 2>/dev/null | head -15 || true
done

echo ""
echo "=== lee27 existing dirs ==="
ls -lad /p/project1/*/lee27* /p/scratch/*/lee27* 2>/dev/null | head -40
ls -la ~ | head -30

echo DONE
