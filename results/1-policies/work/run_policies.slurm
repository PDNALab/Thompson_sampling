#!/bin/bash
#SBATCH --job-name=policies
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --qos=alberto.perezant-b
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2gb
#SBATCH --time=02:00:00
# Runs run_policies.sh (9 simulation runs of 100 seeds each, ~15 min on one CPU). Submit with submit.sh from this folder.
cd "$SLURM_SUBMIT_DIR"
ml conda && conda activate /orange/alberto.perezant/jokent.gaza/envs/thompson
bash run_policies.sh
