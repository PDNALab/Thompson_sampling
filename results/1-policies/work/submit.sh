#!/bin/bash
# Usage:  bash submit.sh --dry-run     (checks the request, submits nothing)
#         bash submit.sh               (submits run_policies.slurm on the burst QOS alberto.perezant-b)
# Finished outputs are skipped by run_policies.sh, so submitting again never overwrites data/.
cd "$(dirname "$0")"
mkdir -p logs
if [ "$1" = "--dry-run" ]; then sbatch --test-only run_policies.slurm; exit 0; fi
sbatch --parsable run_policies.slurm | tee -a logs/last_job.txt
