#!/bin/bash
# Hyperparameter sensitivity of UCB (exploration weight c) and epsilon-greedy (exploration rate eps) on the BRD3 library (CD-HIT 0.5 clusters).
# Prior Beta(0.2, 9.8), k = 50 clusters per round, equal allocation, batch 50, seed set 5000 (counted as queries).
#   seeds 1-20    : tuning seeds   (the settings of the paper were selected here: UCB c = 2.0, epsilon-greedy eps = 0.02)
#   seeds 101-200 : evaluation seeds (the figure 0-hyperparam shows these)
#   UCB c in {0.05, 0.1, 0.25, 0.5, 1.0, 1.41, 2.0, 5.0};  epsilon-greedy eps in {0.01, 0.02, 0.05, 0.1, 0.2, 0.3}
# Output ../data/hp_sensitivity/seeds{1-20,101-200}/{ucb_c<c>,epsgreedy_e<eps>}.npz   (OUT=/some/dir changes ../data)
# Needs the conda environment 'thompson'. ~30 min on one CPU. Finished outputs are skipped, never overwritten.
cd "$(dirname "$0")"
RUN=../../scripts/run_experiment.py; LIB=../data/library_cdhit_0.5.csv; OUT=${OUT:-../data}
[ -f "$LIB" ] || python make_library.py "$LIB"
COMMON="--data $LIB --partition sequence --k 50 --alpha0 0.2 --beta0 9.8 --alloc equal"
for SEEDS in 1-20 101-200; do
  D=$OUT/hp_sensitivity/seeds$SEEDS; mkdir -p $D
  for C in 0.05 0.1 0.25 0.5 1.0 1.41 2.0 5.0; do python $RUN $COMMON --method ucb --c $C --seeds $SEEDS --out $D/ucb_c$C.npz; done
  for E in 0.01 0.02 0.05 0.1 0.2 0.3; do python $RUN $COMMON --method eps_greedy --eps $E --seeds $SEEDS --out $D/epsgreedy_e$E.npz; done
done
echo SENSITIVITY_DONE
