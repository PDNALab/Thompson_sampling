#!/bin/bash
#SBATCH --job-name=clusterings
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#SBATCH --qos=alberto.perezant-b
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2gb
#SBATCH --time=02:00:00
# One of the 12 clusterings (task index = 3 * threshold index + method index) with run_clusterings.sh. Submit with submit_clusterings.sh from this folder.
cd "$SLURM_SUBMIT_DIR"
ml conda && conda activate /orange/alberto.perezant/jokent.gaza/envs/thompson
METHODS=(cdhit easycluster easylinclust); THRS=(0.4 0.5 0.7 0.9)
bash run_clusterings.sh ${METHODS[$((SLURM_ARRAY_TASK_ID % 3))]} ${THRS[$((SLURM_ARRAY_TASK_ID / 3))]}
