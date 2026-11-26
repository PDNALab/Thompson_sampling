#!/bin/bash
# RF-greedy ablation with the removed feature vector replaced by noise (full minus B plus noise(B)); capped at 30k queries.
# Array 0-14: block = id % 5 (AAC DPC PHYS NTERM CTERM), seed = 101 + id / 5. Compare with full (rf_greedy first 30k) and minus_B (rf_ablation).
# Burst QOS alberto.perezant-b (no --account; investment QOS busy, user request 2026-11-17). Submit from this folder:  sbatch --array=0-14 rf_noise.slurm
# 2 threads (same picks as 1), checkpoint every 100 rounds: resubmit the same command to resume. The model has 470 features, so ~0.5-1 h per task.
#SBATCH --job-name=rf-noise
#SBATCH --qos=alberto.perezant-b
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=04:00:00
#SBATCH --output=logs/rf_noise_%a.log
cd "$SLURM_SUBMIT_DIR"
BLOCKS=(AAC DPC PHYS NTERM CTERM)
ID=$SLURM_ARRAY_TASK_ID
/orange/alberto.perezant/jokent.gaza/envs/thompson/bin/python rf_noise.py --block "${BLOCKS[$((ID % 5))]}" --seed $((101 + ID / 5)) --max_queries 30000 --n_jobs "$SLURM_CPUS_PER_TASK"
