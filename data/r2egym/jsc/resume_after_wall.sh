#!/bin/bash
# resume_after_wall.sh <run> <jobid> <min_step> — wait until <jobid> has left the queue (wall/TIMEOUT), settle 3 min, then resume_arm.sh <run> <min_step>.
R=$1; J=$2; MIN=$3
while squeue -h -j $J -o %T 2>/dev/null | grep -q .; do sleep 60; done
echo "$(date) $R job $J gone; latest ckpt $(cat /e/fscratch/reformo/lee27/experiments/$R/$R/checkpoints/latest_ckpt_global_step.txt)"; sleep 180
bash /e/project1/transfernetx/lee27/code/snowball/resume_arm.sh $R $MIN; echo "$(date) DONE"
