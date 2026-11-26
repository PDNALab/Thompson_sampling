#!/bin/bash
# Feature-vector ablation of the RF part of the hybrid (bandit picks the cluster, default RF ranks inside it), capped at 30k queries.
# Array 0-89: feature condition = id % 10 (minus_AAC .. minus_CTERM, only_AAC .. only_CTERM), bandit = (id / 10) % 3 (ts, ucb, eps_greedy), seed = 101 + id / 30.
# Same conditions, cap and seeds as the RF-greedy ablation (rf_ablation.slurm); the full-feature reference is bandit_<pol>_rf_seed<S>.npz (first 30k queries).
# Investment QOS alberto.perezant (no --account). Submit from this folder:  sbatch --array=0-89 bandit_rf_ablation.slurm
# 2 threads (same picks as 1), checkpoint every 100 rounds: resubmit the same command to resume.
#SBATCH --job-name=bandit-rf-abl
#SBATCH --qos=alberto.perezant
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=04:00:00
#SBATCH --output=logs/bandit_rf_abl_%A_%a.log
cd "$SLURM_SUBMIT_DIR"
CONDS=(minus_AAC minus_DPC minus_PHYS minus_NTERM minus_CTERM only_AAC only_DPC only_PHYS only_NTERM only_CTERM)
POL=(ts ucb eps_greedy)
ID=$SLURM_ARRAY_TASK_ID
/orange/alberto.perezant/jokent.gaza/envs/thompson/bin/python bandit_rf.py --policy "${POL[$(( (ID / 10) % 3 ))]}" --seed $((101 + ID / 30)) --within rf --cond "${CONDS[$((ID % 10))]}" --max_queries 30000 --n_jobs "$SLURM_CPUS_PER_TASK"
