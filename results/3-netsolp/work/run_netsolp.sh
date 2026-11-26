#!/bin/bash
# Discovery curves of uniform random, Thompson sampling (TS), UCB and epsilon-greedy on the NetSolP solubility labels (figures/3-netsolp.png, plot/3-netsolp.py).
# Same 142,338 peptides and CD-HIT 0.5 clusters as the BRD3 library, with the label "predicted solubility above a threshold" instead of "binds BRD3-ET" (labels of the SI: data/si_top{2,5,10}.csv,
# score > 0.865 / 0.82 / 0.77, i.e. 2,757 / 7,382 / 14,532 positives = ~2% / 5% / 10% of the library).
# The policies use the settings of the BRD3 figure 1-policies/figures/1-baselines.png (A) unchanged, nothing is re-tuned for NetSolP:
#   Beta(0.01, 0.99) prior, k = 50 clusters per round, equal allocation, batch 50, seed set 5000 (counted as queries), UCB c = 2.0, epsilon-greedy eps = 0.02; 100 evaluation seeds (101-200).
# Output data/curves/<label>/{random,ts,ucb,epsgreedy}.npz. Needs the conda environment 'thompson'. Roughly 5 min per label on one CPU.
# Usage:  bash run_netsolp.sh top2 | top5 | top10     (finished outputs are skipped, never overwritten)
cd "$(dirname "$0")"
LAB=${1:?label: top2, top5 or top10}
RUN=../../scripts/run_experiment.py; LIB=../data/si_$LAB.csv; OUT=${OUT:-../data}/curves/$LAB
COMMON="--data $LIB --k 50 --alloc equal --alpha0 0.01 --beta0 0.99 --seeds 101-200"
mkdir -p $OUT
run() { name=$1; shift; [ -s $OUT/$name.npz ] && { echo "skip $LAB/$name"; return; }; python $RUN $COMMON "$@" --out $OUT/$name.npz; }
run random    --method random
run ts        --method ts
run ucb       --method ucb --c 2.0
run epsgreedy --method eps_greedy --eps 0.02
echo "NETSOLP_DONE $LAB"
