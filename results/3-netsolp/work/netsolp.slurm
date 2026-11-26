#!/bin/bash
#SBATCH --job-name=netsolp
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#SBATCH --qos=alberto.perezant-b      # burst QOS (job 51719504 was moved from the investment QOS with scontrol update)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2gb
#SBATCH --time=02:00:00
# One NetSolP label per task (task index 0-2 = top2 top5 top10) with run_netsolp.sh. Submit with submit_netsolp.sh from this folder.
cd "$SLURM_SUBMIT_DIR"
ml conda && conda activate /orange/alberto.perezant/jokent.gaza/envs/thompson
LABS=(top2 top5 top10)
bash run_netsolp.sh ${LABS[$SLURM_ARRAY_TASK_ID]}
