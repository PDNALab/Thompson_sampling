#!/bin/bash
# CD-HIT sweep (0.4 ... 0.9) + diagnostics + TS replay (BRD3 settings) on the peptide-array library; 6 tasks (index = threshold 0.4, 0.5, ..., 0.9), 1 CPU each (CD-HIT -T 1; the replay is numpy).
# Investment QOS alberto.perezant (moved from burst on 2026-11-18, user request; no --account). Submit from this folder:  sbatch --test-only --array=0-5 cluster_peptides.slurm ; sbatch --array=0-5 cluster_peptides.slurm
# Estimate: LBtope (32.6k 12-mer-like peptides) took < 1 min per threshold; 166k overlapping 12-mers perhaps 5-30 min CD-HIT + ~3 min replay (20 TS + 20 random + 10 null runs of ~2-4 s). 3 h limit.
#SBATCH --job-name=pa-cluster
#SBATCH --qos=alberto.perezant
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=03:00:00
#SBATCH --output=logs/cluster_%a.log
cd "$SLURM_SUBMIT_DIR"
/orange/alberto.perezant/jokent.gaza/envs/thompson/bin/python -I cluster_peptides.py --task "$SLURM_ARRAY_TASK_ID"
