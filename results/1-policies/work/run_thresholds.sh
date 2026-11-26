#!/bin/bash
# Binder-definition sensitivity (3-threshold.png): uniform random, Thompson sampling, UCB (c = 2.0) and epsilon-greedy (eps = 0.02) on the BRD3 library (CD-HIT 0.5 clusters) relabelled with other cutoffs
# (make_threshold_libraries.py: d8, d10, d12, d15 = distance cutoff in A at pLDDT 70; p60, p80 = pLDDT cutoff at 20 A). Same settings as run_policies.sh: Beta(0.01, 0.99) prior, k = 50, equal allocation, batch 50,
# seed set 5000, evaluation seeds 101-200; the settings were selected on the paper definition (20 A, pLDDT 70) and are NOT re-tuned. The paper definition itself is data/curves_sequence (run_policies.sh).
# Usage: bash run_thresholds.sh [name]      (no argument: all six; names d8 d10 d12 d15 p60 p80; used by the Slurm array of thresholds.slurm)
# Needs the conda environment 'thompson'. ~15-20 min per name on one CPU. Output ../data/curves_thresholds/<name>/{random,ts,ucb,epsgreedy}.npz (OUT=/some/dir changes ../data); finished outputs are skipped.
cd "$(dirname "$0")"
RUN=../../scripts/run_experiment.py; OUT=${OUT:-../data}; INPUTS=${INPUTS:-/blue/alberto.perezant/jokent.gaza/ts_threshold_inputs}
[ -f "$INPUTS/d8.csv" ] || python make_threshold_libraries.py "$INPUTS"
COMMON="--partition sequence --k 50 --alloc equal --alpha0 0.01 --beta0 0.99 --seeds 101-200"
for N in ${1:-d8 d10 d12 d15 p60 p80}; do
  D=$OUT/curves_thresholds/$N; mkdir -p $D; IN=$INPUTS/$N.csv
  nice -n 10 python $RUN --data $IN $COMMON --method random --out $D/random.npz
  nice -n 10 python $RUN --data $IN $COMMON --method ts --out $D/ts.npz
  nice -n 10 python $RUN --data $IN $COMMON --method ucb --c 2.0 --out $D/ucb.npz
  nice -n 10 python $RUN --data $IN $COMMON --method eps_greedy --eps 0.02 --out $D/epsgreedy.npz
done
echo THRESHOLDS_DONE
