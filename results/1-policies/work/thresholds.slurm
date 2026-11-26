#!/bin/bash
#SBATCH --job-name=thresholds
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#SBATCH --qos=alberto.perezant        # investment QOS (the burst QOS alberto.perezant-b was saturated; job 51668487 was moved with scontrol update)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2gb
#SBATCH --time=02:00:00
# One of the six relabelled libraries (task index 0-5 = d8 d10 d12 d15 p60 p80) with run_thresholds.sh. Submit with submit_thresholds.sh from this folder.
cd "$SLURM_SUBMIT_DIR"
ml conda && conda activate /orange/alberto.perezant/jokent.gaza/envs/thompson
NAMES=(d8 d10 d12 d15 p60 p80)
bash run_thresholds.sh ${NAMES[$SLURM_ARRAY_TASK_ID]}
