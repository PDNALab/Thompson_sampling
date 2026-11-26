"""LR-greedy with plain scikit-learn defaults, refit after every batch.

Identical loop to rf_greedy.py (same seed sets, same 470 features, same batch size, same tie-breaking); only the model differs:
  LogisticRegression()   defaults: L2, C = 1.0, lbfgs, max_iter = 100, no class weights, no feature scaling, cold start every fit.
Not a default, deliberately not added: feature standardisation. The 470 columns have very different scales (column std 0.002-0.76),
so with C = 1 the L2 penalty shrinks the small-scale columns (dipeptide frequencies) hardest. This is part of what "default" means here.

Output (per seed, OUT/lr_greedy_default_refit<R>_seed<S>.npz): order, fit_s, score_s, top_p, hits as in rf_greedy.py, plus
  n_iter     int16 per fit: lbfgs iterations used
  not_conv   int8 per fit: 1 if sklearn raised a ConvergenceWarning

Usage: python lr_greedy.py --seed 101 [--refit_every 1] [--max_queries 70000] [--out DIR]
"""
import argparse
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

from rf_greedy import DATA, load_features


def run(X, y, seed, S=5000, B=50, refit_every=1, max_queries=70000, log_every=100):
    n = len(y)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)                       # same first RNG call as tsbench / rf_greedy -> same seed set
    seed_idx = perm[:S]
    avail = np.ones(n, dtype=bool)
    avail[seed_idx] = False
    labelled = seed_idx.copy()
    order = [seed_idx.astype(np.int32)]
    fit_s, score_s, top_p, hits, n_iter, not_conv = [], [], [], [], [], []
    queried, cum = S, int(y[seed_idx].sum())
    print(f"seed {seed}: seed set {S}, binders in seed {cum}", flush=True)
    t, p = 0, None
    while queried < max_queries and avail.any():
        if t % refit_every == 0:
            t0 = time.time()
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                lr = LogisticRegression().fit(X[labelled], y[labelled])
            t1 = time.time()
            rows = np.flatnonzero(avail)
            p = np.full(n, -1.0, dtype=np.float32)
            p[rows] = lr.predict_proba(X[rows])[:, 1]
            fit_s.append(t1 - t0); score_s.append(time.time() - t1)
            n_iter.append(int(lr.n_iter_[0])); not_conv.append(int(any(issubclass(x.category, ConvergenceWarning) for x in w)))
        cand = np.flatnonzero(avail)
        s = p[cand] + 1e-7 * rng.random(len(cand))   # random tie-breaking only
        b = min(B, len(cand))
        pick = cand[np.argpartition(-s, b - 1)[:b]] if len(cand) > b else cand
        avail[pick] = False
        labelled = np.concatenate([labelled, pick])
        order.append(pick.astype(np.int32))
        top_p.append(float(np.clip(p[pick], 0, 1).mean())); hits.append(int(y[pick].sum()))
        queried += len(pick); cum += hits[-1]; t += 1
        if t % log_every == 0:
            print(f"  round {t} queried {queried} binders {cum} ({cum / 3393:.1%}) last-{log_every}-round hit rate "
                  f"{np.sum(hits[-log_every:]) / (B * log_every):.3f} non-converged fits {int(np.sum(not_conv))} "
                  f"elapsed fit+score {sum(fit_s) + sum(score_s):.0f}s", flush=True)
    return dict(order=np.concatenate(order), fit_s=np.array(fit_s, np.float32), score_s=np.array(score_s, np.float32),
                top_p=np.array(top_p, np.float32), hits=np.array(hits, np.int8),
                n_iter=np.array(n_iter, np.int16), not_conv=np.array(not_conv, np.int8))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--refit_every", type=int, default=1)
    ap.add_argument("--max_queries", type=int, default=70000)
    ap.add_argument("--out", type=Path, default=DATA)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    X, y = load_features()
    res = run(X, y, a.seed, refit_every=a.refit_every, max_queries=a.max_queries)
    f = a.out / f"lr_greedy_default_refit{a.refit_every}_seed{a.seed}.npz"
    np.savez_compressed(f, **res, seed=a.seed, refit_every=a.refit_every)
    print("saved", f)
