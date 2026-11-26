#!/bin/bash
# Usage:  bash submit_thresholds.sh --dry-run     (checks the request, submits nothing)
#         bash submit_thresholds.sh               (6-task array of thresholds.slurm on the investment QOS alberto.perezant (the burst QOS alberto.perezant-b also works), 1 CPU, 2 GB each)
# Finished outputs are skipped by run_thresholds.sh, so submitting again never overwrites data/.
cd "$(dirname "$0")"
mkdir -p logs
if [ "$1" = "--dry-run" ]; then sbatch --test-only --array=0-5 thresholds.slurm; exit 0; fi
sbatch --parsable --array=0-5 thresholds.slurm | tee -a logs/last_job.txt
