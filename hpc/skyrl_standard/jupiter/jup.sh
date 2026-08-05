#!/usr/bin/env bash
# jup / jrc -- the ONLY sanctioned way to run a command on Jupiter or JURECA.
#
# WHY THIS EXISTS
# Three SSH rules have been written in ai_memory/gotchas.md, twice each, and violated
# anyway -- three times on 2026-08-05 alone:
#
#   1. ALWAYS -o BatchMode=yes. Without it a refused session falls back to
#      keyboard-interactive and burns 3 TOTP tries per attempt, risking a JuDoor
#      lockout.
#   2. NEVER run Jupiter ssh calls concurrently. The ControlMaster refuses
#      concurrent sessions ("Session open refused by peer"). On 2026-08-05 three
#      parallel background loops made every call fail for ~20 minutes and produced
#      a false "our JuDoor key is rejected" diagnosis.
#   3. Long jobs go in tmux ON the cluster, never as a held ssh session. A 55-minute
#      copy held the connection slot and looked exactly like an auth outage.
#
# Prose did not stop any of it. This does: a lockfile serialises calls, BatchMode is
# not optional, and `jup_bg` is the only long-running path.
#
# Usage:
#   jup 'squeue --me'                 # serialised, BatchMode, short command
#   jrc 'squeue -u $USER'             # same, but on JURECA via the CM on Jupiter
#   jup_bg copyjob 'bash /path/x.sh'  # long job -> tmux session on Jupiter
#   jrc_bg mirror  'bash /path/x.sh'  # long job -> tmux session on JURECA
#   jup_log copyjob                   # tail that job's log
#
# Source it:  source hpc/skyrl_standard/jupiter/jup.sh

_JUP_LOCK="${TMPDIR:-/tmp}/.jup.lock"
_JUP_CM_JURECA='$HOME/.ssh/cm_jureca/qwen36'
_JUP_JURECA_HOST='jureca05.fz-juelich.de'   # only jrlogin05 owns 10.14.0.46

# Serialise every call through one lockfile. flock where available (Linux),
# mkdir-spin otherwise (macOS ships no flock).
_jup_locked() {
  if command -v flock >/dev/null 2>&1; then
    flock -w 300 9 || { echo "jup: lock timeout" >&2; return 1; }
    "$@"
  else
    local d="${_JUP_LOCK}.d" i=0
    until mkdir "$d" 2>/dev/null; do
      i=$((i+1)); [ "$i" -gt 300 ] && { echo "jup: lock timeout" >&2; return 1; }
      sleep 1
    done
    trap 'rmdir "$d" 2>/dev/null' RETURN
    "$@"
    rmdir "$d" 2>/dev/null
    trap - RETURN
  fi
}

_jup_raw() { ssh -o BatchMode=yes -o ConnectTimeout=25 jupiter "$@"; }

jup() {
  [ $# -gt 0 ] || { echo "usage: jup '<command>'" >&2; return 2; }
  if command -v flock >/dev/null 2>&1; then
    ( _jup_locked _jup_raw "$@" ) 9>"$_JUP_LOCK"
  else
    _jup_locked _jup_raw "$@"
  fi
}

# JURECA is reached through the ControlMaster that lives ON the Jupiter login node --
# NOT on your laptop. `ssh jureca` from a Mac is expected to fail; that is not an outage.
jrc() {
  [ $# -gt 0 ] || { echo "usage: jrc '<command>'" >&2; return 2; }
  jup "ssh -o BatchMode=yes -S $_JUP_CM_JURECA $_JUP_JURECA_HOST \"\$@\"" _ "$@"
}

# Long-running work: detach into tmux on the cluster so no ssh session is held.
jup_bg() {
  local name="${1:?usage: jup_bg <name> '<command>'}"; shift
  local log="\$HOME/jup_${name}.log"
  jup "tmux kill-session -t $name 2>/dev/null; tmux new-session -d -s $name \"$* > $log 2>&1\"; sleep 2; tmux ls 2>&1 | grep -E '^$name' || echo 'jup_bg: session did not start -- check $log'"
}

jrc_bg() {
  local name="${1:?usage: jrc_bg <name> '<command>'}"; shift
  local log="/p/scratch/synthlaion/lee27/jrc_${name}.log"
  jrc "tmux kill-session -t $name 2>/dev/null; tmux new-session -d -s $name \"$* > $log 2>&1\"; sleep 2; tmux ls 2>&1 | grep -E '^$name' || echo 'jrc_bg: session did not start -- check $log'"
}

jup_log() { jup "tail -n ${2:-40} \$HOME/jup_${1:?usage: jup_log <name> [lines]}.log"; }
jrc_log() { jrc "tail -n ${2:-40} /p/scratch/synthlaion/lee27/jrc_${1:?usage: jrc_log <name> [lines]}.log"; }

# Deliver a local file to the cluster without a held pipe.
jup_put() {
  local src="${1:?usage: jup_put <local> <remote>}" dst="${2:?}"
  jup "cat > $dst" < "$src" && jup "wc -c $dst"
}

jup_help() { sed -n '1,32p' "${BASH_SOURCE[0]}"; }
