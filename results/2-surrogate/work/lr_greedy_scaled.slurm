#!/bin/bash
# LR-greedy on standardized features (sklearn defaults, refit every round), one seed per array task, 1 CPU each.
# Investment QOS (no --account). Submit from this folder:  sbatch --array=101-103 lr_greedy_scaled.slurm
#SBATCH --job-name=lr-greedy-scaled
#SBATCH --qos=alberto.perezant
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=02:00:00
#SBATCH --output=logs/lr_greedy_scaled_%a.log
cd "$SLURM_SUBMIT_DIR"
/orange/alberto.perezant/jokent.gaza/envs/thompson/bin/python lr_greedy_scaled.py --seed "$SLURM_ARRAY_TASK_ID" --refit_every 1
