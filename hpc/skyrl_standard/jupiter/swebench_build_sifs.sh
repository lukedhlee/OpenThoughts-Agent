#!/bin/bash
# swebench_build_sifs.sh <task_list_file|--all> [n_parallel]
#
# Build SWE-bench Verified SIFs. Deliberately NOT harbor's prebuild_sifs.sh.
#
# WHY A SEPARATE SCRIPT
# prebuild_sifs.sh was written for task sets where many tasks SHARE a base image
# (SweSmith: 2500 tasks -> 38 SIFs). SWE-bench Verified is the opposite: all 500
# Dockerfiles are byte-identical except the FROM line, so bases are 1:1 with
# tasks. Under those conditions prebuild_sifs.sh does two harmful things:
#   1. it pre-pulls every FROM into its own base_<img>.sif AND then builds a
#      second SIF from the .def -- ~2x disk for zero reuse (~1-2 TB, not ~1 TB);
#   2. its .def has a %post (apt-get, pip), which needs root or --fakeroot;
#      it hides all build errors behind 2>/dev/null and silently retries.
#
# The only things those Dockerfiles add on top of the base image are:
#      RUN curl -LsSf https://astral.sh/uv/0.7.13/install.sh | sh
#      RUN mkdir -p /logs
# Neither needs to be baked in:
#   - harbor's worker puts $BRIDGE_AGENT_TOOLS/bin and .../uv_env/.venv/bin on
#     PATH inside the container, so `uv` is already available at runtime;
#   - /logs/verifier is bind-mounted, which creates /logs.
# So a plain `apptainer pull` of the base is sufficient: no %post, no fakeroot,
# no second copy. VERIFY THIS ON A SAMPLE BEFORE TRUSTING IT -- if `uv` turns
# out not to be on PATH, fall back to a %post build for the affected tasks.
#
# The output name matches what worker.py resolves:
#   build_<task_name>-<sha256(Dockerfile)[:12]>.sif
set -uo pipefail
ROOT=/p/scratch/synthlaion/lee27
TASKS=${TASKS:-$ROOT/tasks/swebench-verified}
SIF_CACHE=${SIF_CACHE:-$ROOT/swebench_sif}
export APPTAINER_CACHEDIR=${APPTAINER_CACHEDIR:-$ROOT/apptainer_cache_swebench100}
export APPTAINER_TMPDIR=${APPTAINER_TMPDIR:-$ROOT/apptainer_tmp}
mkdir -p "$SIF_CACHE" "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"

SEL="${1:?usage: swebench_build_sifs.sh <task_list_file|--all> [n_parallel]}"
NPAR="${2:-4}"

if [[ "$SEL" == "--all" ]]; then
  mapfile -t LIST < <(ls "$TASKS")
else
  mapfile -t LIST < "$SEL"
fi
echo "=== swebench SIF build: ${#LIST[@]} tasks, parallel=$NPAR, cache=$SIF_CACHE"

build_one() {
  local T="$1"
  local DF="$TASKS/$T/environment/Dockerfile"
  [[ -f "$DF" ]] || { echo "[$T] SKIP no Dockerfile"; return 1; }
  local HASH IMG SIF
  HASH=$(sha256sum "$DF" | cut -c1-12)
  IMG=$(grep -i '^FROM' "$DF" | head -1 | awk '{print $2}')
  SIF="$SIF_CACHE/build_${T}-${HASH}.sif"
  if [[ -f "$SIF" ]]; then echo "[$T] CACHED"; return 0; fi
  local T0=$SECONDS
  # Errors are NOT swallowed -- prebuild_sifs.sh's 2>/dev/null is why silent
  # build failures were invisible.
  if apptainer pull --force "$SIF.tmp" "docker://$IMG" >"$SIF_CACHE/$T.pull.log" 2>&1; then
    mv "$SIF.tmp" "$SIF"
    echo "[$T] OK $(( SECONDS - T0 ))s $(du -h "$SIF" | cut -f1)"
    rm -f "$SIF_CACHE/$T.pull.log"
    return 0
  fi
  echo "[$T] FAIL $(( SECONDS - T0 ))s -- $(tail -2 "$SIF_CACHE/$T.pull.log" | tr '\n' ' ')"
  rm -f "$SIF.tmp"
  return 1
}
export -f build_one
export TASKS SIF_CACHE

printf '%s\n' "${LIST[@]}" | xargs -P "$NPAR" -I{} bash -c 'build_one "$@"' _ {}
echo "=== done. sifs=$(ls $SIF_CACHE/build_*.sif 2>/dev/null | wc -l) disk=$(du -sh $SIF_CACHE | cut -f1)"
