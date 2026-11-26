#!/bin/bash
# Usage:  bash submit_hyperparam.sh --dry-run     (checks the request, submits nothing)
#         bash submit_hyperparam.sh               (submits hyperparam.slurm on the burst QOS alberto.perezant-b)
# Finished outputs are skipped by the run scripts, so submitting again never overwrites data/.
cd "$(dirname "$0")"
mkdir -p logs
if [ "$1" = "--dry-run" ]; then sbatch --test-only hyperparam.slurm; exit 0; fi
sbatch --parsable hyperparam.slurm | tee -a logs/last_job.txt
