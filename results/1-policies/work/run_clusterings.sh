#!/bin/bash
# Discovery curves of Thompson sampling, UCB (c = 2.0) and epsilon-greedy (eps = 0.02) on the 12 clusterings of the BRD3 library (CD-HIT, MMseqs2 easy-cluster, easy-linclust x identity 0.4, 0.5, 0.7, 0.9),
# Beta(0.01, 0.99) prior, k = 50, equal allocation, batch 50, seed set 5000, evaluation seeds 101-200. Uniform random sampling does not depend on the clustering: data/curves_sequence/random.npz (run_policies.sh) is used. The settings were selected on CD-HIT 0.5 (tuning seeds 1-20) and are not re-tuned.
# Usage: bash run_clusterings.sh [method thr]      (no arguments: all 12; method in cdhit easycluster easylinclust, thr in 0.4 0.5 0.7 0.9; used by the Slurm array of clusterings.slurm)
# Needs the conda environment 'thompson'. ~10-20 min per clustering on one CPU. Output ../data/curves_clusterings/<method>_<thr>/{ts,ucb,epsgreedy}.npz (OUT=/some/dir changes ../data); finished outputs are skipped.
# The clustering inputs (38 MB) are built by make_cluster_inputs.py into INPUTS (default /blue/alberto.perezant/jokent.gaza/ts_cluster_inputs).
cd "$(dirname "$0")"
RUN=../../scripts/run_experiment.py; OUT=${OUT:-../data}; INPUTS=${INPUTS:-/blue/alberto.perezant/jokent.gaza/ts_cluster_inputs}
[ -f "$INPUTS/cdhit_0.5.csv" ] || python make_cluster_inputs.py "$INPUTS"
COMMON="--partition sequence --k 50 --alloc equal --alpha0 0.01 --beta0 0.99 --seeds 101-200"
D0=$OUT/curves_clusterings; mkdir -p $D0
for THR in ${2:-0.4 0.5 0.7 0.9}; do for M in ${1:-cdhit easycluster easylinclust}; do
  D=$D0/${M}_$THR; mkdir -p $D; IN=$INPUTS/${M}_$THR.csv
  nice -n 10 python $RUN --data $IN $COMMON --method ts --out $D/ts.npz
  nice -n 10 python $RUN --data $IN $COMMON --method ucb --c 2.0 --out $D/ucb.npz
  nice -n 10 python $RUN --data $IN $COMMON --method eps_greedy --eps 0.02 --out $D/epsgreedy.npz
done; done
echo CLUSTERINGS_DONE
