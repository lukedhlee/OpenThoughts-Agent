#!/bin/bash
# Second pass over the harvested tasks: (1) slim delta tarballs = install.sh outputs minus build/, __pycache__, *.pyc,
# generated *.c (runtime needs only the in-place .so, egg-info, modified tracked files); (2) one tarball per uv-managed
# Python (root/.local/share/uv/python/<dir>), shared across every task that used that interpreter.
set -u
TSV=$1; W=$2; C=/p/scratch/synthlaion/lee27/r2egym_sif
export GIT_CONFIG_GLOBAL=/dev/null GIT_CONFIG_NOSYSTEM=1
rm -f "$W/out/uvpython_bin.tar.zst"; : > "$W/manifests/slim_sizes.txt"
while IFS=$'\t' read -r task image h; do
  sif=$(ls -L $C/*-$h.sif | head -1); out=$W/y_$task; rm -rf "$out"
  off=$(apptainer sif list "$sif" | awk -F'|' '/FS/ {gsub(/ /,"",$4); split($4,a,"-"); print a[1]}')
  unsquashfs -q -n -no-xattrs -p 2 -o "$off" -d "$out" "$sif" testbed root/.local/share/uv >/dev/null 2>&1 || true
  tb=$out/testbed
  pyhome=$(grep -m1 '^home' $tb/.venv/pyvenv.cfg | sed 's/.*= *//'); uvrel=$(dirname "${pyhome#/}"); uvname=$(basename "$uvrel")
  uvtar="$W/out/uvpython_${uvname}.tar.zst"
  if [ ! -s "$uvtar" ]; then (cd $out && tar --zstd --numeric-owner -cf "$uvtar" "$uvrel"); echo "uvpython $uvname $(du -sb $out/$uvrel | cut -f1) raw -> $(stat -c %s "$uvtar") zst" >> "$W/manifests/slim_sizes.txt"; fi
  grep -v -E '^build/|__pycache__/|\.pyc$|\.c$|\.cpp$' "$W/manifests/$task.delta.list" > "$W/manifests/$task.delta-slim.list"
  raw=$(cd $tb && tr '\n' '\0' < "$W/manifests/$task.delta-slim.list" | xargs -0 stat -c %s 2>/dev/null | awk '{s+=$1} END{print s+0}')
  (cd $tb && tar --zstd --numeric-owner -cf "$W/out/$task.delta-slim.tar.zst" -T "$W/manifests/$task.delta-slim.list")
  echo "$task slim_files=$(wc -l < "$W/manifests/$task.delta-slim.list") slim_raw=$raw slim_zst=$(stat -c %s "$W/out/$task.delta-slim.tar.zst") full_zst=$(stat -c %s "$W/out/$task.delta.tar.zst")" >> "$W/manifests/slim_sizes.txt"
  # sha manifest for the uv python (once per interpreter) for the dedupe count
  [ -s "$W/manifests/uvpython_${uvname}.sha256" ] || (cd $out && find "$uvrel" -type f -print0 | xargs -0 -r sha256sum > "$W/manifests/uvpython_${uvname}.sha256")
  rm -rf "$out"
done < "$TSV"
cat "$W/manifests/slim_sizes.txt"; echo SLIM_DONE
