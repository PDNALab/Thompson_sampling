#!/bin/bash
# Usage:  bash submit_clusterings.sh --dry-run     (checks the request, submits nothing)
#         bash submit_clusterings.sh               (12-task array of clusterings.slurm on the burst QOS alberto.perezant-b, 1 CPU, 2 GB each, at most 12 at a time)
# Finished outputs are skipped by run_clusterings.sh, so submitting again never overwrites data/.
cd "$(dirname "$0")"
mkdir -p logs
if [ "$1" = "--dry-run" ]; then sbatch --test-only --array=0-11 clusterings.slurm; exit 0; fi
sbatch --parsable --array=0-11 clusterings.slurm | tee -a logs/last_job.txt
