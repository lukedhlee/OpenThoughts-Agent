#!/bin/bash
# pyspy_sample.sh <out.log> <pid>... — every 60 s for 40 min: cpu% of the pids and the NON-idle thread stacks (py-spy dump --nonblocking).
PY=/e/project1/transfernetx/lee27/code/envs/snowball/bin/py-spy; OUT=$1; shift; PIDS="$*"
for i in $(seq 1 40); do
  { echo "##### $(date +%H:%M:%S)"; ps -o pid,pcpu,nlwp,rss,args -p ${PIDS// /,} | cut -c1-90; } >> $OUT
  for p in $PIDS; do
    echo "--- pid $p" >> $OUT
    timeout 25 $PY dump --nonblocking --pid $p 2>&1 | awk '/^Thread/{keep=($0 !~ /idle/)} /^Process/{next} keep' | head -120 >> $OUT
  done
  sleep 60
done
echo "##### done" >> $OUT
