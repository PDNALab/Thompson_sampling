"""How similar are the 50 peptides picked in one batch? The bandit arms (Thompson sampling, UCB, epsilon-greedy) and uniform random on 100 evaluation seeds (101-200), first 25,000 queries, BRD3 library.

Same statistics as batch_similarity_forest.py (the forest runs, seeds 101-103, are in data/batch_similarity.npz; script batch_similarity_forest.py), per batch of 50 picks after the 5,000-peptide seed set:
  neardup : fraction of the picks with a shift-neighbour (sliding-window shift <= 3 residues) among the other picks of the same batch
  groups  : number of connected groups of the shift-neighbour graph of the batch (50 = all picks unrelated)
  clusters: number of distinct CD-HIT 0.5 clusters among the 50 picks
Settings of the arms: those of 1-policies/figures/1-baselines.png (Beta(0.01, 0.99) prior, k = 50 clusters per round, equal allocation, batch 50, seed set 5,000, UCB c = 2.0, epsilon-greedy eps = 0.02).
Check: for seeds 101-103 the statistics of random, ts and ucb are identical to those stored in data/batch_similarity.npz (printed at the end).
Output ../data/batch_similarity_arms.npz  (<method>__stats: seeds x 400 batches x 3 (neardup, groups, clusters), seeds).
Usage: nice -n 10 python batch_similarity_arms.py     (conda environment 'thompson'; a few minutes on one CPU; run from any directory; never overwrites an existing output)"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                                        # results/2-surrogate
sys.path.insert(0, str(ROOT.parent / "scripts"))          # results/scripts/tsbench
from tsbench import Library, make_policy, run, run_random  # noqa: E402

OUT = ROOT / "data" / "batch_similarity_arms.npz"
if OUT.exists():
    sys.exit(f"{OUT} exists; not overwritten")
SEEDS, S, B, MAXQ, MAXSHIFT = list(range(101, 201)), 5000, 50, 25000, 3

lib_df = pd.read_csv(ROOT.parent / "1-policies" / "data" / "library_cdhit_0.5.csv")
seq = pd.read_csv(ROOT / "data" / "sequences.csv").set_index("id")["sequence"]
seqs = seq.loc[lib_df["id"]].values
n = len(seqs)
lib = Library(lib_df["label"].values, lib_df["cluster"].values)
cl = lib.groups

nbrs = [[] for _ in range(n)]                             # shift neighbours by hashing (library ids are not in protein order)
for d in range(1, MAXSHIFT + 1):
    head = {}
    for j in range(n):
        head.setdefault(seqs[j][:-d], []).append(j)
    for i in range(n):
        for j in head.get(seqs[i][d:], ()):
            if j != i:
                nbrs[i].append(j); nbrs[j].append(i)
nbrs = [set(x) for x in nbrs]
print("neighbour graph built", flush=True)


def batch_stats(picks):
    ps = set(int(p) for p in picks); idx = {int(p): k for k, p in enumerate(picks)}; parent = list(range(len(picks)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a
    dup = 0
    for p in picks:
        nb = nbrs[int(p)] & ps
        dup += bool(nb)
        for q in nb:
            parent[find(idx[int(p)])] = find(idx[q])
    return dup / len(picks), len({find(k) for k in range(len(picks))}), len(set(cl[picks]))


def per_batch(order):
    nb = (MAXQ - S) // B
    out = np.zeros((nb, 3))
    for t in range(nb):
        out[t] = batch_stats(order[S + t * B: S + (t + 1) * B])
    return out


stats = {m: [] for m in ("random", "ts", "ucb", "epsgreedy")}
for s in SEEDS:
    runs = {"random": run_random(lib, s, S=S, B=B, max_queries=MAXQ, return_order=True)}
    for m, p, kw in (("ts", "ts", {}), ("ucb", "ucb", dict(c=2.0)), ("epsgreedy", "eps_greedy", dict(eps=0.02))):
        runs[m] = run(lib, make_policy(p, **kw), s, S=S, B=B, k=50, alloc="equal", alpha0=0.01, beta0=0.99, max_queries=MAXQ, return_order=True)
    for m, o in runs.items():
        stats[m].append(per_batch(o["order"]))
    if s % 20 == 0:
        print("seed", s, flush=True)
np.savez_compressed(OUT, **{f"{m}__stats": np.stack(v) for m, v in stats.items()}, seeds=np.array(SEEDS), S=S, B=B)

old = np.load(ROOT / "data" / "batch_similarity.npz")      # check against the stored runs of seeds 101-103
for m in ("random", "ts", "ucb"):
    new = np.stack(stats[m][:3])
    print(f"{m:7s} seeds 101-103 identical to data/batch_similarity.npz: {np.allclose(new, old[f'{m}__stats'])}")
for m, v in stats.items():
    a = np.stack(v)
    print(f"{m:9s} similar picks {100 * a[:, :, 0].mean():.3f}%  groups {a[:, :, 1].mean():.2f}  clusters {a[:, :, 2].mean():.2f}")
