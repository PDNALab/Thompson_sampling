#!/bin/bash
# Sparse LBtope: TS + RF picker, RF-greedy and feature ablation on the 5 sparse libraries, 355 tasks (python sparse_hybrid.py --list): per library (71 tasks) {hybrid, rf} x 11 feature sets x seeds 101-103,
# then the hybrid_random control (seeds 101-105); cap 50 % of N (10,826 queries). TS prior = BRD3's Beta(0.01, 0.99), k = 50, equal allocation, NOT tuned (user's choice 2026-11-18). Submit:
#   sbatch --test-only --array=0-354 sparse_hybrid.slurm ; sbatch --array=0-354 sparse_hybrid.slurm
# Burst QOS alberto.perezant-b at first; the tasks still pending on 2026-11-18 were moved to the investment QOS alberto.perezant on user request. 1 CPU per task (n_jobs does not change the picks). Estimate: ~8-20 min per task (LBtope at 32.6k peptides took 15-36 min to 16.3k queries; here 10.8k queries). Checkpoints every 100 rounds.
#SBATCH --job-name=sparse-hyb
#SBATCH --qos=alberto.perezant-b
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=02:00:00
#SBATCH --output=logs/sparse_hyb_%a.log
cd "$SLURM_SUBMIT_DIR"
/orange/alberto.perezant/jokent.gaza/envs/thompson/bin/python -I sparse_hybrid.py --task "$SLURM_ARRAY_TASK_ID" --n_jobs "$SLURM_CPUS_PER_TASK"
