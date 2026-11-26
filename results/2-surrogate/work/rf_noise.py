"""RF-greedy ablation with the removed feature vector REPLACED BY NOISE (full minus B plus noise(B)).

Question: is a drop in the leave-one-out runs (rf_ablation.py, minus_<B>) due to the INFORMATION in B, or just to changing the feature set
(fewer columns, a different sqrt(p) = max_features, a different mix of columns)? Here the forest still has all 470 columns and sqrt(p) = 21:
  noise_<B> = the other four vectors unchanged + noise(B), where noise(B) has the same number of columns as B and every column is the real column
              of B independently permuted across all 142,338 peptides (marginal distribution of each column kept; its association with the label,
              with the other columns of B and with the other vectors destroyed).
Reading: full ~ noise_B  -> B carried no information the forest used;  minus_B ~ noise_B -> the loss is information;  minus_B > noise_B -> the forest
is hurt by useless columns (dilution);  noise_B < full -> B was informative.

The noise matrix is drawn once per run from default_rng([777, seed, block index]) (fixed through the run, different for every seed) and is NOT used for
anything but this ablation. Everything else is rf_greedy.run: default RandomForestClassifier(), refit after every batch, same seed set, 50 picks per round.
Output OUT/rf_noise_<B>_seed<S>.npz (same arrays as rf_greedy.py), stops at --max_queries (default 30,000); checkpoints every 100 rounds, rerun to resume.
Usage: python rf_noise.py --block DPC --seed 101 [--max_queries 30000] [--n_jobs 2] [--out DIR]
"""
import argparse
from pathlib import Path

import numpy as np

import rf_greedy as R

BLOCKS = ["AAC", "DPC", "PHYS", "NTERM", "CTERM"]


def noise_block(B, rng):
    """every column of B independently permuted across rows."""
    return np.stack([B[rng.permutation(len(B)), j] for j in range(B.shape[1])], axis=1)


def build(block, seed, z=None):
    z = R.load_blocks() if z is None else z
    rng = np.random.default_rng([777, seed, BLOCKS.index(block)])
    parts = [noise_block(z[b], rng) if b == block else z[b] for b in BLOCKS]
    return np.hstack(parts).astype(np.float32), z["label"].astype(np.int8)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--block", required=True, choices=BLOCKS)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--max_queries", type=int, default=30000)
    ap.add_argument("--n_jobs", type=int, default=1)
    ap.add_argument("--out", type=Path, default=R.DATA)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    X, y = build(a.block, a.seed)
    assert X.shape == (142338, 470)
    print(f"noise_{a.block}: {X.shape[1]} features (block {a.block} replaced by independently permuted columns), seed {a.seed}", flush=True)
    f = a.out / f"rf_noise_{a.block}_seed{a.seed}.npz"
    res = R.run(X, y, a.seed, max_queries=a.max_queries, n_jobs=a.n_jobs, ckpt=f.with_suffix(".ckpt"))
    np.savez_compressed(f, **res, seed=a.seed, block=a.block)
    print("saved", f)
