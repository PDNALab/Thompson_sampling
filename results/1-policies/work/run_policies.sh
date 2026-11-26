#!/bin/bash
# Discovery curves of the selection policies of the paper figure (figures/1-baselines.png, plot/1-baselines.py) on the BRD3 library (CD-HIT 0.5 clusters).
#   Thompson sampling (TS), UCB (c = 2.0), epsilon-greedy (eps = 0.02): prior Beta(0.01, 0.99), k = 50 clusters per round, equal allocation, batch 50, seed set 5000 (counted as queries),
#   plus uniform random sampling and the cluster-uniform random control; 100 evaluation seeds (101-200).
# Hyperparameters were selected on tuning seeds 1-20 (see 1-policies in the paper's analysis); the evaluation seeds do not overlap with them.
#   data/curves_sequence/  : the policies on the sequence clusters      (random, ts, ucb, epsgreedy, cluster_uniform)
#   data/curves_randpart/  : the same policies on RANDOM partitions with the same cluster-size distribution (null control; ts, ucb, epsgreedy, cluster_uniform)
# Needs the conda environment 'thompson' (numpy, pandas). ~15 min on one CPU.
# Usage:  bash run_policies.sh            (from any directory; finished outputs are skipped, never overwritten)
#         OUT=/some/dir bash run_policies.sh   writes to /some/dir/{curves_sequence,curves_randpart} instead of ../data
cd "$(dirname "$0")"
RUN=../../scripts/run_experiment.py; LIB=../data/library_cdhit_0.5.csv; OUT=${OUT:-../data}
COMMON="--data $LIB --watchlist ../data/watchlist.csv --k 50 --alloc equal --alpha0 0.01 --beta0 0.99 --seeds 101-200"
[ -f "$LIB" ] || python make_library.py "$LIB"
mkdir -p "$OUT/curves_sequence" "$OUT/curves_randpart"
S="$OUT/curves_sequence"; R="$OUT/curves_randpart"
python $RUN $COMMON --method random          --out $S/random.npz
python $RUN $COMMON --method ts              --out $S/ts.npz
python $RUN $COMMON --method ucb --c 2.0     --out $S/ucb.npz
python $RUN $COMMON --method eps_greedy --eps 0.02 --out $S/epsgreedy.npz
python $RUN $COMMON --method cluster_uniform --out $S/cluster_uniform.npz
python $RUN $COMMON --partition random --method ts              --out $R/ts.npz
python $RUN $COMMON --partition random --method ucb --c 2.0     --out $R/ucb.npz
python $RUN $COMMON --partition random --method eps_greedy --eps 0.02 --out $R/epsgreedy.npz
python $RUN $COMMON --partition random --method cluster_uniform --out $R/cluster_uniform.npz
echo POLICIES_DONE
