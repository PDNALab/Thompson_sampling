#!/bin/bash
# Bandit picks the cluster (TS / UCB c=2 / eps-greedy eps=0.02, prior Beta(0.01,0.99), k=50, equal allocation),
# default RandomForestClassifier() refit every round ranks the peptides inside it. One task per bandit for one seed.
# Investment QOS alberto.perezant (no --account). History: seed 101 ran on burst, seeds 102-103 on investment (user request 2026-11-17). Submit from this folder:  sbatch --array=0-2 --export=ALL,SEED=101 bandit_rf.slurm
# Array index 0/1/2 = ts / ucb / eps_greedy. 4 threads (same picks as 1 thread), checkpoint every 100 rounds: resubmit the same command to resume.
#SBATCH --job-name=bandit-rf
#SBATCH --qos=alberto.perezant
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=10:00:00
#SBATCH --output=logs/bandit_rf_%A_%a.log
cd "$SLURM_SUBMIT_DIR"
POL=(ts ucb eps_greedy)
/orange/alberto.perezant/jokent.gaza/envs/thompson/bin/python bandit_rf.py --policy "${POL[$SLURM_ARRAY_TASK_ID]}" --seed "${SEED:-101}" --within rf --n_jobs "$SLURM_CPUS_PER_TASK"
