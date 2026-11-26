#!/bin/bash
# Beta-prior sweep for UCB (c = 2.0), epsilon-greedy (eps = 0.02) and Thompson sampling (equal allocation) on the BRD3 library (CD-HIT 0.5 clusters); k = 50, batch 50, seed set 5000.
# prior = (alpha0, beta0) = (m*s, (1-m)*s) with prior mean m in {0.005, 0.01, 0.02, 0.05, 0.1, 0.25} and prior strength s in {0.3, 1, 3, 10, 30, 100}, plus Beta(1, 1) and Beta(0.5, 0.5).
# Tuning seeds 1-20 only (the prior of each method is selected on them by select_prior.py and then evaluated once on the evaluation seeds; see run_policies.sh).
# Output ../data/hp_prior/seeds1-20/{ucb,epsgreedy,ts}_m<m>_s<s>.npz   (OUT=/some/dir changes ../data)
# Needs the conda environment 'thompson'. ~45 min on one CPU. Finished outputs are skipped, never overwritten.
cd "$(dirname "$0")"
RUN=../../scripts/run_experiment.py; LIB=../data/library_cdhit_0.5.csv; OUT=${OUT:-../data}
[ -f "$LIB" ] || python make_library.py "$LIB"
D=$OUT/hp_prior/seeds1-20; mkdir -p $D
COMMON="--data $LIB --partition sequence --k 50 --alloc equal --seeds 1-20"
python - <<'P' | while read N A B; do
for m in (0.005, 0.01, 0.02, 0.05, 0.1, 0.25):
    for s in (0.3, 1, 3, 10, 30, 100): print(f"m{m}_s{s} {m*s:.8g} {(1-m)*s:.8g}")
print("m0.5_s2 1 1"); print("m0.5_s1 0.5 0.5")
P
  python $RUN $COMMON --alpha0 $A --beta0 $B --method ucb --c 2.0 --out $D/ucb_$N.npz
  python $RUN $COMMON --alpha0 $A --beta0 $B --method eps_greedy --eps 0.02 --out $D/epsgreedy_$N.npz
  python $RUN $COMMON --alpha0 $A --beta0 $B --method ts --out $D/ts_$N.npz
done
echo PRIOR_SWEEP_DONE
