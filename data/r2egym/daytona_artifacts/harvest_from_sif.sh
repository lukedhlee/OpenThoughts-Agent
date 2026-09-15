#!/bin/bash
# Harvest R2E-Gym per-task build artifacts from the JSC SIF cache without executing the image.
# Per task: extract /testbed (incl .git), /root/.local/share/uv and /r2e_tests from the squashfs partition,
# split into (a) the uv-managed Python, (b) the task's .venv, (c) the "delta" = every file install.sh
# produced or changed in /testbed outside .venv (untracked+ignored+modified vs HEAD), (d) the R2E tests.
# Usage: harvest_from_sif.sh <tsv: task<TAB>image<TAB>sifhash> <workdir>   (login node safe: -p 2, 1 SIF at a time)
set -u
TSV=$1; W=$2; C=/p/scratch/synthlaion/lee27/r2egym_sif
mkdir -p "$W/out" "$W/manifests"
export GIT_CONFIG_GLOBAL=/dev/null GIT_CONFIG_NOSYSTEM=1
while IFS=$'\t' read -r task image h; do
  sif=$(ls -L $C/*-$h.sif | head -1); out=$W/x_$task
  t0=$(date +%s)
  if [ ! -d "$out/testbed/.venv" ]; then rm -rf "$out"
    off=$(apptainer sif list "$sif" | awk -F'|' '/FS/ {gsub(/ /,"",$4); split($4,a,"-"); print a[1]}')
    unsquashfs -q -n -no-xattrs -p 2 -o "$off" -d "$out" "$sif" testbed root/.local/share/uv r2e_tests >/dev/null 2>"$W/manifests/$task.unsquash.err" || true
  fi
  t1=$(date +%s)
  tb=$out/testbed
  pyhome=$(grep -m1 '^home' $tb/.venv/pyvenv.cfg | sed 's/.*= *//'); pyver=$(grep -m1 '^version' $tb/.venv/pyvenv.cfg | sed 's/.*= *//' || true)
  [ -z "$pyver" ] && pyver=$(grep -m1 -o 'cpython-[0-9.]*' <<<"$pyhome" | head -1)
  sp=$(ls -d $tb/.venv/lib/python*/site-packages | head -1)
  numpy=$(ls $sp | grep -i -m1 -E '^numpy-[0-9]' | sed 's/.dist-info//;s/.egg-info//'); cython=$(ls $sp | grep -i -m1 -E '^cython-[0-9]' | sed 's/.dist-info//;s/.egg-info//'); st=$(ls $sp | grep -i -m1 -E '^setuptools-[0-9]' | sed 's/.dist-info//')
  head=$(git -c safe.directory='*' -C $tb rev-parse HEAD)
  # delta = untracked+ignored files (file level) + modified tracked files, excluding .venv
  (cd $tb && git -c safe.directory='*' ls-files --others -z | tr '\0' '\n' | grep -v '^\.venv/' ; git -c safe.directory='*' diff --name-only HEAD) | sort -u > "$W/manifests/$task.delta.list"
  (cd $tb && git -c safe.directory='*' ls-files --deleted) > "$W/manifests/$task.deleted.list" || true
  # sizes
  venv_b=$(du -sb $tb/.venv | cut -f1); venv_n=$(find $tb/.venv -type f | wc -l)
  delta_b=$(cd $tb && tr '\n' '\0' < "$W/manifests/$task.delta.list" | xargs -0 stat -c %s 2>/dev/null | awk '{s+=$1} END{print s+0}'); delta_n=$(wc -l < "$W/manifests/$task.delta.list")
  so_b=$(cd $tb && grep -E '\.so(\.|$)' "$W/manifests/$task.delta.list" | tr '\n' '\0' | xargs -0 stat -c %s 2>/dev/null | awk '{s+=$1} END{print s+0}'); so_n=$(grep -c -E '\.so(\.|$)' "$W/manifests/$task.delta.list" || true)
  uvdir=$(realpath "$pyhome/.." 2>/dev/null || echo "$pyhome"); uvrel=${uvdir#$out/}
  # hashes for dedupe (venv + uv python + delta)
  (cd $out && { find testbed/.venv -type f; find "$uvrel" -type f; sed 's#^#testbed/#' "$W/manifests/$task.delta.list"; } | sort -u | tr '\n' '\0' | xargs -0 -r sha256sum 2>/dev/null) > "$W/manifests/$task.sha256" || true
  t2=$(date +%s)
  # tarballs: delta (relative to /testbed), venv (relative to /testbed), tests (r2e_tests + run_tests.sh), uv python (once per version)
  (cd $tb && tar --zstd --numeric-owner -cf "$W/out/$task.delta.tar.zst" -T "$W/manifests/$task.delta.list")
  (cd $tb && tar --zstd --numeric-owner -cf "$W/out/$task.venv.tar.zst" .venv)
  (cd $out && tar --zstd --numeric-owner -cf "$W/out/$task.tests.tar.zst" r2e_tests testbed/run_tests.sh testbed/install.sh 2>/dev/null || true)
  uvtar="$W/out/uvpython_$(basename "$uvdir").tar.zst"; [ -f "$uvtar" ] || (cd $out && tar --zstd --numeric-owner -cf "$uvtar" "$uvrel")
  t3=$(date +%s)
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$task" "$image" "$head" "$pyver" "$(basename "$uvdir")" "$numpy" "$cython" "$st" "$venv_b" "$venv_n" "$delta_b" "$delta_n" "$so_b" "$so_n" "$((t1-t0))" "$((t2-t1))" "$((t3-t2))" >> "$W/manifests/summary.tsv"
  ls -l "$W/out/$task."*.tar.zst | awk '{print $5, $9}' >> "$W/manifests/tarsizes.txt"
  rm -rf "$out"
  echo "done $task extract=$((t1-t0))s hash=$((t2-t1))s tar=$((t3-t2))s py=$pyver numpy=$numpy delta=$delta_b venv=$venv_b"
done < "$TSV"
echo ALL_DONE
