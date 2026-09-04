#!/bin/bash
#SBATCH --partition=dc-cpu-devel
#SBATCH --account=synthlaion
#SBATCH --nodes=2
#SBATCH --time=02:00:00
#SBATCH --output=/p/scratch/synthlaion/lee27/dc_agent_eval/logs/%x_%j.out
# gate_job.sbatch — the gate sweep AS the batch job (HALF=a|b via --export). envgate_par_tt.sh is idempotent per parts dir,
# so a chain of these (--dependency=afterany) resumes until the list is exhausted; the fleet gate cancels the chain.
cd /p/scratch/synthlaion/lee27 || exit 1
echo "$(date) gate_job half=$HALF job=$SLURM_JOB_ID nodes=$SLURM_JOB_NODELIST"
FLEET=$SLURM_JOB_ID EG_STAGE_ROOT=/p/scratch/synthlaion/lee27/envgate_tt EG_TAG=tt_full_$HALF EG_SCRIPT=envgate_tt.sh \
  EG_TASKLIST=/p/scratch/synthlaion/lee27/tt_gate_full_$HALF.txt EG_CONC=48 bash envgate_par_tt.sh
echo "$(date) gate_job half=$HALF finished"
