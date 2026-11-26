#!/bin/bash
# Feature-vector ablation of RF-greedy (leave-one-vector-out and one-vector-only), capped at 30k queries.
# Array 0-29: condition = id % 10 (minus_AAC .. minus_CTERM, only_AAC .. only_CTERM), seed = 101 + id / 10.
# Burst QOS alberto.perezant-b (no --account). Submit from this folder:  sbatch --array=0-29 rf_ablation.slurm
# 2 threads (same picks as 1), checkpoint every 100 rounds: resubmit the same command to resume. ~0.3-1 h per task.
#SBATCH --job-name=rf-ablation
#SBATCH --qos=alberto.perezant-b
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=04:00:00
#SBATCH --output=logs/rf_ablation_%a.log
cd "$SLURM_SUBMIT_DIR"
CONDS=(minus_AAC minus_DPC minus_PHYS minus_NTERM minus_CTERM only_AAC only_DPC only_PHYS only_NTERM only_CTERM)
ID=$SLURM_ARRAY_TASK_ID
/orange/alberto.perezant/jokent.gaza/envs/thompson/bin/python rf_ablation.py --cond "${CONDS[$((ID % 10))]}" --seed $((101 + ID / 10)) --max_queries 30000 --n_jobs "$SLURM_CPUS_PER_TASK"
