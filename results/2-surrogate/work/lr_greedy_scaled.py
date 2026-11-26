"""LR-greedy as lr_greedy.py, but on standardized features. Everything else is identical (same loop,
seed sets, 470 features, LogisticRegression() defaults, refit every round); the only change is the preprocessing.

Standardisation = StandardScaler fitted ONCE on the features of the whole library (mean 0, std 1 per column). It uses no labels,
and the sequences of all candidates are known before screening, so nothing is leaked; it is not refitted as labels accumulate.

Output: OUT/lr_greedy_scaled_refit<R>_seed<S>.npz (same arrays as lr_greedy.py).
Usage: python lr_greedy_scaled.py --seed 101 [--refit_every 1] [--max_queries 70000] [--out DIR]
"""
import argparse
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler

from lr_greedy import run
from rf_greedy import DATA, load_features

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--refit_every", type=int, default=1)
    ap.add_argument("--max_queries", type=int, default=70000)
    ap.add_argument("--out", type=Path, default=DATA)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    X, y = load_features()
    Xs = StandardScaler().fit_transform(X).astype(np.float32)
    res = run(Xs, y, a.seed, refit_every=a.refit_every, max_queries=a.max_queries)
    f = a.out / f"lr_greedy_scaled_refit{a.refit_every}_seed{a.seed}.npz"
    np.savez_compressed(f, **res, seed=a.seed, refit_every=a.refit_every)
    print("saved", f)
