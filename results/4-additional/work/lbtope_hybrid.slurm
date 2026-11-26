#!/bin/bash
# LBtope Fixed: TS (tuned Beta(0.1, 0.9), k = 50, equal allocation) + RF picker, and the feature-vector ablation (D of 1-forest.png). 120 tasks (python lbtope_hybrid.py --list):
#   0-109   {hybrid, rf} x 11 feature sets x seeds 101-105, cap 50 % of N      (index = method * 55 + cond * 5 + seed offset)
#   110-119 hybrid_random (tuned TS, random pick; no forest), seeds 101-110, cap 50 %: control that must reproduce tsbench's arm policy
# v1 (job 51758668, cap 20 %) finished 120 tasks in minutes; the user then asked for 50 % (a first 50 % job was cancelled before any task ran), briefly for 100 % (never submitted), and finally 50 % -> this file. v1 outputs (*_cap0.2_*) are kept as prefixes.
# Burst QOS alberto.perezant-b (no --account; user's choice 2026-11-17), no throttle, 1 CPU per task (n_jobs does not change the picks; 1 CPU only lengthens the forest fits). Submit from this folder:
#   sbatch --test-only --array=0-119 lbtope_hybrid.slurm ; sbatch --array=0-119 lbtope_hybrid.slurm
# Estimate: v1 measured 4.5-7 min per task at the 20 % cap on 1 CPU; the 50 % cap has ~2.5x the rounds with larger fits -> ~30-50 min per task (3 h limit; checkpoints every 100 rounds, resubmit to resume). Checkpoints every 100 rounds: resubmit to resume.
#SBATCH --job-name=lbtope-hyb
#SBATCH --qos=alberto.perezant-b
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=03:00:00
#SBATCH --output=logs/lbtope_hyb_%a.log
cd "$SLURM_SUBMIT_DIR"
/orange/alberto.perezant/jokent.gaza/envs/thompson/bin/python -I lbtope_hybrid.py --task "$SLURM_ARRAY_TASK_ID" --n_jobs "$SLURM_CPUS_PER_TASK"
