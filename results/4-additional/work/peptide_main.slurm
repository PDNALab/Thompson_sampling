#!/bin/bash
# Peptide-array library, CD-HIT 0.5: RF-greedy and TS + RF picker (full 470 features, seeds 101-105) and the random-pick TS control (seeds 101-110): 20 tasks (python peptide_hybrid.py --set main --list).
# Investment QOS alberto.perezant (no --account), 2 CPUs per task (n_jobs does not change the picks). Submit from this folder:
#   sbatch --test-only --array=0-19 peptide_main.slurm ; sbatch --array=0-19 peptide_main.slurm
# Estimate from the BRD3 runs (142k peptides, 1,300 rounds, 4 CPUs): RF-greedy 1.5 h, hybrid 1 h; here ~1,540 rounds (cap 50 % of 165,946) on 2 CPUs -> ~3-4 h (RF-greedy), ~2-3 h (hybrid), control ~2 min. Checkpoints every 100 rounds.
#SBATCH --job-name=pa-main
#SBATCH --qos=alberto.perezant
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --time=12:00:00
#SBATCH --output=logs/main_%a.log
cd "$SLURM_SUBMIT_DIR"
/orange/alberto.perezant/jokent.gaza/envs/thompson/bin/python -I peptide_hybrid.py --set main --task "$SLURM_ARRAY_TASK_ID" --n_jobs "$SLURM_CPUS_PER_TASK"
