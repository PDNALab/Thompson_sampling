#!/bin/bash
# Peptide-array library, CD-HIT 0.5: feature-vector ablation of RF-greedy and TS + RF picker, 10 non-full feature sets x seeds 101-103: 60 tasks (python peptide_hybrid.py --set ablation --list).
# Burst QOS alberto.perezant-b (no --account), 1 CPU per task, no throttle. On 2026-11-18, while all 60 tasks were pending, half were moved to the investment QOS alberto.perezant (user request): tasks 0-14 and 30-44 (the five 'minus_<block>' feature sets, both methods); tasks 15-29 and 45-59 (the 'only_<block>' sets) stayed on burst. A resubmit of a task from this file goes to burst. The full-feature runs are in peptide_main.slurm. Submit from this folder:
#   sbatch --test-only --array=0-59 peptide_ablation.slurm ; sbatch --array=0-59 peptide_ablation.slurm
# Estimate: ~3x the 4-CPU BRD3 times on 1 CPU -> RF-greedy ~4-6 h, hybrid ~3-4 h per task (feature sets with fewer columns are faster); limit 12 h, checkpoints every 100 rounds (resubmit to resume).
#SBATCH --job-name=pa-abl
#SBATCH --qos=alberto.perezant-b
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=12:00:00
#SBATCH --output=logs/abl_%a.log
cd "$SLURM_SUBMIT_DIR"
/orange/alberto.perezant/jokent.gaza/envs/thompson/bin/python -I peptide_hybrid.py --set ablation --task "$SLURM_ARRAY_TASK_ID" --n_jobs "$SLURM_CPUS_PER_TASK"
