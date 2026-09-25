#!/bin/bash
# CalibForge on Daytona (data/calibforge/daytona): turn the shared base sandbox into this task's image.
# The task image is the base's layers plus the layers listed below. This script fetches those layers by digest,
# checks each sha256, and stacks them onto / the way a container runtime does (whiteouts first, then contents),
# restores the dpkg entries of harbor's agent tooling, and deletes its own helper dir. Run by harbor as root
# before the agent starts (setup-files hook); a non-zero exit fails the trial before the agent runs.
#
# Layers come from CF_MIRRORS (blob base URLs, tried in order, empty by default) and then from the Docker Hub blob
# endpoint. Blob downloads are not metered pulls on Docker Hub (only manifest requests are), and no manifest is read.
set -euo pipefail
T0=$(date +%s%N)
CF=/opt/cfdelta
TASK_IMAGE="@@IMAGE@@"
REPO="${TASK_IMAGE%@*}"
CF_MIRRORS="${CF_MIRRORS:-@@MIRRORS@@}"
# digest compressed-bytes, bottom to top
LAYERS="@@LAYERS@@"
DL=$CF/dl   # inside the helper dir: nothing a task layer whites out
[ -x "$CF/curl" ] || { echo "cfdelta: $CF/curl missing: not a CalibForge base snapshot, or setup already ran" >&2; exit 3; }
rm -rf "$DL"; mkdir -p "$DL"
C=("$CF/curl" -fsSL --cacert "$CF/cacert.pem" --connect-timeout 20 --retry 6 --retry-all-errors --retry-delay 2)

TOKEN=""
fetch() {  # fetch <digest> <out>
  local d=$1 out=$2 m
  for m in $CF_MIRRORS; do
    "${C[@]}" -o "$out" "${m%/}/${d#sha256:}" 2>/dev/null && return 0
  done
  "${C[@]}" -H "Authorization: Bearer $TOKEN" -o "$out" "https://registry-1.docker.io/v2/$REPO/blobs/$d"
}

n=0; pids=()
while read -r d size; do
  [ -n "$d" ] || continue
  if [ -z "$TOKEN" ] ; then
    TOKEN=$("${C[@]}" "https://auth.docker.io/token?service=registry.docker.io&scope=repository:$REPO:pull" \
            | sed -n 's/.*"token" *: *"\([^"]*\)".*/\1/p')
    [ -n "$TOKEN" ] || { echo "cfdelta: no registry token" >&2; exit 4; }
  fi
  fetch "$d" "$DL/$n.tgz" & pids+=($!)
  n=$((n+1))
done <<< "$LAYERS"
for p in "${pids[@]}"; do wait "$p" || { echo "cfdelta: layer download failed" >&2; exit 5; }; done
T1=$(date +%s%N)

i=0; bytes=0
while read -r d size; do
  [ -n "$d" ] || continue
  f="$DL/$i.tgz"
  echo "${d#sha256:}  $f" | sha256sum -c --quiet - || { echo "cfdelta: digest mismatch on $d" >&2; exit 6; }
  bytes=$((bytes + $(stat -c %s "$f")))
  # whiteouts refer to lower layers, so they are applied before this layer's own contents
  { tar -tzf "$f" | grep -E '(^|/)\.wh\.' || true; } | while IFS= read -r p; do
    p=${p#./}; b=${p##*/}; dir=""
    [ "$p" != "$b" ] && dir=${p%/*}
    if [ "$b" = ".wh..wh..opq" ]; then
      [ -d "/$dir" ] && find "/$dir" -mindepth 1 -maxdepth 1 -exec rm -rf {} +
    else
      rm -rf "/${dir:+$dir/}${b#.wh.}"
    fi
  done
  tar -xzf "$f" -C / --numeric-owner --same-owner --same-permissions --keep-directory-symlink \
      --xattrs --xattrs-include='security.capability' --exclude='.wh.*'
  i=$((i+1))
done <<< "$LAYERS"
T2=$(date +%s%N)

# A task layer that ran apt replaced /var/lib/dpkg/status with one that predates the agent tooling; add back the
# tooling's stanzas for packages the task's status does not list (tmux, asciinema and their dependencies).
if [ -s "$CF/tooling.status" ] && [ -f /var/lib/dpkg/status ]; then
  awk 'NR==FNR { if ($1 == "Package:") have[$2] = 1; next } /^Package: / { keep = !($2 in have) } keep' \
      /var/lib/dpkg/status "$CF/tooling.status" > "$DL/add.status"
  if [ -s "$DL/add.status" ]; then
    [ -z "$(tail -c1 /var/lib/dpkg/status)" ] || echo >> /var/lib/dpkg/status
    cat "$DL/add.status" >> /var/lib/dpkg/status
  fi
fi
command -v ldconfig >/dev/null 2>&1 && ldconfig 2>/dev/null || true
rm -rf "$DL" "$CF"
T3=$(date +%s%N)
line=$(printf '{"layers": %d, "bytes": %d, "download_s": %.2f, "apply_s": %.2f, "total_s": %.2f}' \
  "$n" "$bytes" "$(( (T1-T0)/1000000 ))e-3" "$(( (T2-T1)/1000000 ))e-3" "$(( (T3-T0)/1000000 ))e-3")
echo "CFDELTA $line"
if [ -d /logs/agent ]; then echo "$line" > /logs/agent/cfdelta_setup.json 2>/dev/null || true; fi
