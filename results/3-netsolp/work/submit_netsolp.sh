#!/bin/bash
# Usage:  bash submit_netsolp.sh --dry-run     (checks the request, submits nothing)
#         bash submit_netsolp.sh               (3-task array of netsolp.slurm on the burst QOS alberto.perezant-b, 1 CPU, 2 GB each)
# Finished outputs are skipped by run_netsolp.sh, so submitting again never overwrites data/.
cd "$(dirname "$0")"
mkdir -p logs
if [ "$1" = "--dry-run" ]; then sbatch --test-only --array=0-2 netsolp.slurm; exit 0; fi
sbatch --parsable --array=0-2 netsolp.slurm | tee -a logs/last_job.txt
