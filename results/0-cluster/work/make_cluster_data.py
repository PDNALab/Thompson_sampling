"""Compact tables behind the cluster-information figure (plot/0-cluster_info.py), computed from the raw files of the repository:

  clusters/{cd,easy-cluster,easy-linclust}-{0.4,0.5,0.7,0.9}.csv   name, sequence, cluster   the 12 clusterings of the 142,338 unique 25-mers (CD-HIT; MMseqs2 easy-cluster; MMseqs2 easy-linclust)
  data/dict.csv                                                    name, plddt_i, dist_i, rog_i (i = 0..4)   AF-CBA results (five AlphaFold2 models per peptide)

A peptide is a binder if at least 4 of its 5 models have a mean pLDDT >= 70 and a mean distance to the key ET residues < 20 A (3,393 binders).

Output (../data):
  cluster_sizes.csv      tool, identity, cluster, size, binders      one row per cluster of each of the 12 clusterings (panels A and B)
  shift_neighbours.csv   shift, n_pairs, n_focal, p_partner_binder_given_binder, p_partner_binder_given_nonbinder
                         for windows shifted by `shift` residues along a protein (the 24-/23-/22-residue overlap is identical): the probability that the partner window is a binder given that the
                         focal peptide is a binder / is not a binder; both directions of every pair are used (panel C). This uses the overlap of the windows only, not any clustering.
Usage: python make_cluster_data.py     (conda environment 'thompson': numpy, pandas; ~1 min)
"""
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "data"
REPO = HERE.parents[2]                                    # results/0-cluster/work -> repository root
OUT.mkdir(exist_ok=True)

ref = pd.read_csv(REPO / "clusters" / "cd-0.5.csv")        # defines the peptide order
names, seqs = ref["name"].values, ref["sequence"].values
dic = pd.read_csv(REPO / "data" / "dict.csv").set_index("name").loc[names]
n_pass = sum(((dic[f"plddt_{i}"].values >= 70) & (dic[f"dist_{i}"].values < 20)).astype(int) for i in range(5))
y = (n_pass >= 4).astype(int)
print(f"{len(y):,} peptides, {int(y.sum()):,} binders")

# ---- cluster sizes and binders per cluster, 12 clusterings
rows = []
for tool in ("cd", "easy-cluster", "easy-linclust"):
    for thr in (0.4, 0.5, 0.7, 0.9):
        c = pd.read_csv(REPO / "clusters" / f"{tool}-{thr}.csv").set_index("name").loc[names, "cluster"].values
        g = pd.factorize(c)[0]
        size, binders = np.bincount(g), np.bincount(g, weights=y).astype(int)
        rows.append(pd.DataFrame({"tool": tool, "identity": thr, "cluster": np.arange(len(size)), "size": size, "binders": binders}))
        print(f"{tool:14s} {thr}: {len(size):6d} clusters, median size {np.median(size):.0f}, singletons {np.mean(size == 1):.1%}")
pd.concat(rows).to_csv(OUT / "cluster_sizes.csv", index=False)

# ---- windows shifted by d residues (peptide j is the window d residues after peptide i)
shift_rows = []
for d in (1, 2, 3):
    head = defaultdict(list)
    for j, s in enumerate(seqs):
        head[s[:-d]].append(j)
    pairs = np.array([(i, j) for i, s in enumerate(seqs) for j in head.get(s[d:], ()) if j != i])
    both = np.concatenate([pairs, pairs[:, ::-1]])         # ordered pairs: every window is the focal one once
    fa, fb = y[both[:, 0]], y[both[:, 1]]
    shift_rows.append(dict(shift=d, n_pairs=len(pairs), n_focal=int((fa == 1).sum()), p_partner_binder_given_binder=fb[fa == 1].mean(),
                           p_partner_binder_given_nonbinder=fb[fa == 0].mean()))
    print(f"shift {d}: {len(pairs):,} pairs, P(partner binder | binder) = {shift_rows[-1]['p_partner_binder_given_binder']:.3f}")
pd.DataFrame(shift_rows).to_csv(OUT / "shift_neighbours.csv", index=False)
print("written", OUT)
