#!/bin/bash
# TMax on Daytona (data/tmax/daytona): turn the shared ubuntu 22.04 base sandbox into this task's image by replaying
# the task's own Dockerfile after FROM, with Docker's build semantics:
#   ENV   is added to the environment of every later step (the same values are in task.toml, for the agent's shell);
#   COPY  copies a file from /setup_files/context (the task's environment/ dir) with its original mode, owner root;
#   RUN   runs as root under /bin/sh -c from /, with only the image's environment (env -i), fails the setup on a
#         non-zero exit, and afterwards ends every process it started: a build step leaves nothing running, so a
#         service a post_install starts must not be up when the agent arrives.
# Afterwards /etc/hosts, /etc/hostname and /etc/resolv.conf get their original contents back (Docker bind-mounts them
# during a build, so a build step's edits never reach the image), and /setup_files is deleted, so the agent sees
# neither the recipe nor the fixtures' source. Run by harbor as root before the agent starts (setup-files hook); a
# non-zero exit fails the trial before the agent runs.
set -uo pipefail
T0=$(date +%s%N)
TASK="@@TASK@@"
CTX=/setup_files/context
SAVE=/setup_files/.etc   # deleted with /setup_files; nothing a task step touches
[ -d "$CTX" ] || { echo "tmaxsetup: $CTX missing: not a TMax setup dir, or setup already ran" >&2; exit 3; }
mkdir -p "$SAVE"
for f in hosts hostname resolv.conf; do [ -e /etc/$f ] && cat /etc/$f > "$SAVE/$f"; done
BASE_PIDS=" $(cd /proc && echo [0-9]*) "
ENVV=(@@BASE_ENV@@)
NSTEP=0; NKILL=0

reap() {  # end every process started since this script began (not this shell itself); builtins only, no forks
  local pass p c killed
  for pass in 1 2 3; do
    killed=0
    for p in /proc/[0-9]*; do
      p=${p#/proc/}
      case "$BASE_PIDS" in *" $p "*) continue ;; esac
      [ "$p" = "$$" ] && continue
      c=; IFS= read -r -d '' c 2>/dev/null < /proc/$p/cmdline
      [ -n "$c" ] || continue   # kernel threads and zombies have no command line
      kill -9 "$p" 2>/dev/null && killed=$((killed + 1))
    done
    NKILL=$((NKILL + killed))
    [ $killed -eq 0 ] && break
    sleep 0.2
  done
}

step() {  # step <Dockerfile RUN command>; its output goes to a log under /setup_files (shown only on failure)
  local t rc
  NSTEP=$((NSTEP + 1)); t=$(date +%s%N)
  ( cd / && exec env -i "${ENVV[@]}" /bin/sh -c "$1" ) </dev/null >"$SAVE/run$NSTEP.log" 2>&1
  rc=$?
  reap
  echo "tmaxsetup: RUN $NSTEP exited $rc after $(( ($(date +%s%N) - t) / 1000000 )) ms"
  if [ $rc -ne 0 ]; then
    echo "tmaxsetup: RUN $NSTEP failed ($rc): ${1:0:200}" >&2
    tail -c 1500 "$SAVE/run$NSTEP.log" >&2
    exit 10
  fi
}

copy() {  # copy <path in context> <destination> <octal mode>
  mkdir -p "$(dirname "$2")" && cp -f "$CTX/$1" "$2" && chown 0:0 "$2" && chmod "$3" "$2" \
    || { echo "tmaxsetup: COPY $1 -> $2 failed" >&2; exit 11; }
}

setenv() { ENVV+=("$1"); }   # env -i takes the last of repeated names, as Docker's ENV overrides

# ---- the task's Dockerfile after FROM ----
@@STEPS@@
# ------------------------------------------

for f in hosts hostname resolv.conf; do   # in place: they may be mounts
  [ -e "$SAVE/$f" ] || continue
  cmp -s "$SAVE/$f" /etc/$f || { cat "$SAVE/$f" 2>/dev/null > /etc/$f || echo "tmaxsetup: could not restore /etc/$f" >&2; }
done
cd / && rm -rf /setup_files
line=$(printf '{"task": "%s", "run_steps": %d, "killed": %d, "total_s": %.2f}' \
  "$TASK" "$NSTEP" "$NKILL" "$(( ($(date +%s%N) - T0) / 1000000 ))e-3")
echo "TMAXSETUP $line"
if [ -d /logs/agent ]; then echo "$line" > /logs/agent/tmax_setup.json 2>/dev/null || true; fi
