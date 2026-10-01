#!/bin/bash
# shm_prune.sh <trials_dir> [finished_age_min=10] [pressure_pct=70] [sleep_sec=60]
# Keep a tmpfs trials_dir bounded: delete trials whose attempts/*/lifecycle-result.json is older than
# finished_age_min; when tmpfs use passes pressure_pct, delete the oldest 400 finished trials as well.
# Never deletes a trial without a completion marker (it may still be running).
T=$1; AGE=${2:-10}; PCT=${3:-70}; SLEEP=${4:-60}
trial_dir() { local d; d=$(dirname "$1"); d=$(dirname "$d"); dirname "$d"; }
while true; do
  find "$T" -mindepth 4 -maxdepth 4 -name lifecycle-result.json -mmin +"$AGE" 2>/dev/null | while read -r f; do trial_dir "$f"; done | sort -u | xargs -r rm -rf
  u=$(df /dev/shm | tail -n 1 | awk '{print $5}' | tr -d '%')
  if [ "${u:-0}" -gt "$PCT" ]; then
    find "$T" -mindepth 4 -maxdepth 4 -name lifecycle-result.json -printf '%T@ %p\n' 2>/dev/null | sort -n | head -n 400 | awk '{print $2}' | while read -r f; do trial_dir "$f"; done | sort -u | xargs -r rm -rf
  fi
  sleep "$SLEEP"
done
