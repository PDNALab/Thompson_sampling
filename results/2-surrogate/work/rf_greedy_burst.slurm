#!/bin/bash
# RF-greedy (sklearn defaults, refit every round), one seed per array task.
# Burst QOS alberto.perezant-b (no --account; moved from the investment QOS on 2026-11-16 because the group CPU cap was full). Submit from this folder:  sbatch --array=101-103 rf_greedy_burst.slurm
# v2 (2026-11-16): the first array (51671766, 1 CPU, 6 h, no checkpoints) was cancelled after 25 min because a default forest fit grows from
#   2.3 s (5k rows) to 35 s (70k rows), i.e. ~6.5 h per seed. Now 4 threads (same picks as 1 thread, tested), 10 h limit, checkpoint every
#   100 rounds (data/*.ckpt) -> if a job dies, resubmit the same command and it resumes where it stopped.
#SBATCH --job-name=rf-greedy-default
#SBATCH --qos=alberto.perezant-b
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=10:00:00
#SBATCH --output=logs/rf_greedy_%a.log
cd "$SLURM_SUBMIT_DIR"
/orange/alberto.perezant/jokent.gaza/envs/thompson/bin/python rf_greedy.py --seed "$SLURM_ARRAY_TASK_ID" --refit_every 1 --n_jobs "$SLURM_CPUS_PER_TASK"
