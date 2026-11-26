#!/bin/bash
# Full-library LBtope, TS (BRD3 prior Beta(0.01, 0.99), k = 50, equal allocation, not tuned) + RF picker, 11 feature sets x seeds 101-105, and the random-pick control (seeds 101-110): 65 tasks
# (python lbtope_hybrid_b001.py --list). Outputs data/hybrid_b001/. Investment QOS alberto.perezant (moved from burst on 2026-11-18, user request), 1 CPU per task, no throttle (as the tuned-prior run: 15-36 min per task, 3 h limit).
#   sbatch --test-only --array=0-64 lbtope_hybrid_b001.slurm ; sbatch --array=0-64 lbtope_hybrid_b001.slurm
#SBATCH --job-name=lbtope-b001
#SBATCH --qos=alberto.perezant
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=03:00:00
#SBATCH --output=logs/lbtope_b001_%a.log
cd "$SLURM_SUBMIT_DIR"
/orange/alberto.perezant/jokent.gaza/envs/thompson/bin/python -I lbtope_hybrid_b001.py --task "$SLURM_ARRAY_TASK_ID" --n_jobs "$SLURM_CPUS_PER_TASK"
