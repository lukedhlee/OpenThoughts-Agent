#!/bin/bash
# jobwait.sh <jobid> [label] — poll a JURECA job every 60 s; exit when it is RUNNING or gone. Prints one line.
J=$1; L=${2:-$1}; S=$(dirname "$0")
while true; do
  st=$(bash $S/jrc.sh "squeue -h -j $J -o %T%%%R 2>/dev/null" 2>/dev/null)
  case "$st" in
    RUNNING*) echo "$(date +%H:%M:%S) $L $J RUNNING nodes=$(bash $S/jrc.sh "squeue -h -j $J -o %N")"; exit 0;;
    "") echo "$(date +%H:%M:%S) $L $J GONE (not in squeue)"; exit 1;;
    *) ;;
  esac
  sleep 60
done
