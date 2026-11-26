#!/bin/bash
#SBATCH --job-name=hyperparam
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --qos=alberto.perezant-b
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2gb
#SBATCH --time=04:00:00
# Hyperparameter sensitivity (UCB c, epsilon-greedy eps) and Beta-prior sweep, then the prior selection (~1.5 h on one CPU). Submit with submit_hyperparam.sh from this folder.
cd "$SLURM_SUBMIT_DIR"
ml conda && conda activate /orange/alberto.perezant/jokent.gaza/envs/thompson
bash run_hp_sensitivity.sh
bash run_hp_prior.sh
python select_prior.py
