"""Data for ../figures/2-sampling.png: which binders are still unfound when Thompson sampling plateaus, and the clusters of the five known ET binders.

  recovery of the binders of the library as a function of the number of queries (100 TS runs, evaluation seeds 101-200, up to 70,000 queries), separately for the binders of clusters with a binder fraction
  (binders / peptides of the cluster in the library) < 0.05, 0.05-0.1, 0.1-0.2, 0.2-0.5 and >= 0.5, and for all binders        -> ../data/recovery_by_purity.npz
  size, number of binders and binder fraction of the CD-HIT 0.5 cluster of each known epitope (data/watchlist.csv)            -> ../data/known_epitope_clusters.csv
  per-bin summary at 70,000 queries                                                                                           -> ../data/unsampled_by_purity.csv

Settings as in run_policies.sh: Beta(0.01, 0.99) prior, k = 50 clusters per round, equal allocation, batch 50, seed set 5,000. The binder fraction of a cluster uses the labels of the library (it is an explanatory
quantity, not something the policy sees). Usage: python unsampled_by_purity.py      (conda environment 'thompson'; ~4 min on one CPU, nice it)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
sys.path.insert(0, str(HERE.parents[1] / "scripts"))
from tsbench import Library, make_policy, run               # noqa: E402

S, B, K, MAXQ = 5000, 50, 50, 70000
EDGES = [0.0, 0.05, 0.1, 0.2, 0.5, 1.0001]                   # bins of the cluster binder fraction
GRID = np.arange(S, MAXQ + 1, 250)                           # queries
lib_df = pd.read_csv(DATA / "library_cdhit_0.5.csv")
lib = Library(lib_df.label.values, lib_df.cluster.values)
policy = make_policy("ts")

size = np.bincount(lib.groups, minlength=lib.n_groups)
nbind = np.bincount(lib.groups, weights=lib.labels, minlength=lib.n_groups)
frac = nbind / size
binders = np.flatnonzero(lib.labels == 1)
bin_of = np.digitize(frac[lib.groups[binders]], EDGES[1:-1])  # 0..4
n_in_bin = np.bincount(bin_of, minlength=5)

seeds = range(101, 201)
curves = np.zeros((6, len(seeds), len(GRID)), dtype=np.float32)   # rows: the 5 bins, then all binders; fraction of the binders found
for r, seed in enumerate(seeds):
    o = run(lib, policy, seed, S=S, B=B, k=K, alloc="equal", alpha0=0.01, beta0=0.99, max_queries=MAXQ, return_order=True)
    pos = np.full(lib.n, np.iinfo(np.int64).max); pos[o["order"]] = np.arange(len(o["order"]))     # position in the query order (0-based)
    pb = pos[binders]
    for j in range(5):
        curves[j, r] = (pb[bin_of == j][:, None] < GRID[None, :]).mean(0)
    curves[5, r] = (pb[:, None] < GRID[None, :]).mean(0)
np.savez_compressed(DATA / "recovery_by_purity.npz", grid=GRID, curves=curves, edges=np.array(EDGES), n_binders_in_bin=n_in_bin, seeds=np.array(list(seeds)))

rows = []
for j in range(5):
    f = curves[j, :, -1]
    rows.append(dict(bin=f"[{EDGES[j]}, {min(EDGES[j + 1], 1.0)})", binders=int(n_in_bin[j]), unfound_frac_mean=1 - f.mean(), unfound_frac_sd=f.std(ddof=1), unfound_binders_mean=(1 - f.mean()) * n_in_bin[j]))
summ = pd.DataFrame(rows)
summ["share_of_unfound"] = summ.unfound_binders_mean / summ.unfound_binders_mean.sum()
summ.to_csv(DATA / "unsampled_by_purity.csv", index=False)
print(summ.round(3).to_string(index=False))
print(f"unfound binders at {MAXQ} queries: {(1 - curves[5, :, -1].mean()) * len(binders):.0f} of {len(binders)}")

wl = pd.read_csv(DATA / "watchlist.csv")
idx = pd.Series(np.arange(lib.n), index=lib_df["id"].values).loc[wl["id"].values].values
g = lib.groups[idx]
pd.DataFrame(dict(name=wl["name"], cluster=lib_df.cluster.values[idx], size=size[g], binders=nbind[g].astype(int), binder_fraction=frac[g])).to_csv(DATA / "known_epitope_clusters.csv", index=False)
print(pd.read_csv(DATA / "known_epitope_clusters.csv").round(3).to_string(index=False))
