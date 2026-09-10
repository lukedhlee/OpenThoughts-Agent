#!/bin/bash
# probe_release_after.sh <wait_probe> <held_probe> — wait until <wait_probe> is tabled and its trace tree archived, then scontrol release the held job named <held_probe>.
W=$1; S=$2; E=/e/fscratch/reformo/lee27/experiments
until [ -f $E/$W/pass8_summary.json ] && [ -f $E/$W/$W/trace_archive.tar ] && [ ! -d $E/$W/$W/trace_jobs ]; do sleep 120; done
until J=$(squeue -h -u $USER -n $S -o %i | head -1) && [ -n "$J" ]; do sleep 120; done
echo "$(date) $W archived; releasing $S ($J); inodes free $(df -i /e/fscratch/reformo/lee27 | tail -1 | awk "{print \$4}")"; scontrol release $J; echo "$(date) DONE"
