#!/bin/bash
# probe_after.sh <wait_probe> <submit_probe> — wait until <wait_probe> is tabled and its trace tree archived (inode headroom), then submit <submit_probe> and start its probe_watch.
W=$1; S=$2; E=/e/fscratch/reformo/lee27/experiments; C=/e/project1/transfernetx/lee27/code/snowball; O=/e/project1/transfernetx/lee27/code/OpenThoughts-Agent
until [ -f $E/$W/pass8_summary.json ] && [ -f $E/$W/$W/trace_archive.tar ] && [ ! -d $E/$W/$W/trace_jobs ]; do sleep 120; done
echo "$(date) $W archived; submitting $S (inodes free: $(df -i /e/fscratch/reformo/lee27 | tail -1 | awk "{print \$4}"))"
cd $O && DCFT=$PWD sbatch $E/$S/sbatch/${S}_rl.sbatch
tmux new -d -s probe_watch_$S "bash $C/probe_watch.sh $S"
echo "$(date) DONE"
