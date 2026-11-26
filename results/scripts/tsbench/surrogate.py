"""Cluster-blind surrogate-model acquisition (peptide-level; the clusters are never used).

A model is trained on the peptides labelled so far and scores every unqueried peptide; the next batch is the
top-B of those scores. Models: random forest ('rf') or logistic regression ('lr').
Acquisitions:
  greedy       top predicted probability
  ucb          RF only: mean + c * std across trees
  ts           RF only: each round, score with the mean of a random subset of m trees (ensemble Thompson sampling)
  uncertainty  closest to p = 0.5 (uncertainty sampling; improves the model rather than finding hits)
The model is refit every `refit_every` rounds; scores are cached in between and queried peptides masked.
The seed set is identical to the arm-based runs for the same seed (same first RNG call).
"""
import time

import numpy as np


def _make_rf(n_estimators, min_samples_leaf, seed, n_jobs):
    from sklearn.ensemble import RandomForestClassifier

    return RandomForestClassifier(n_estimators=n_estimators, max_features="sqrt", min_samples_leaf=min_samples_leaf,
                                  max_samples=0.3, class_weight="balanced_subsample", n_jobs=n_jobs, random_state=seed)


def _tree_probs(rf, X, rows, n_jobs):
    """(n, T) float32 matrix of per-tree P(binder); only `rows` are computed (others stay 0)."""
    from joblib import Parallel, delayed

    Xr = X[rows]
    cols = Parallel(n_jobs=n_jobs, prefer="threads")(delayed(lambda t: t.predict_proba(Xr)[:, 1])(t) for t in rf.estimators_)
    P = np.zeros((X.shape[0], len(rf.estimators_)), dtype=np.float32)
    P[rows] = np.stack(cols, axis=1)
    return P


def run_surrogate(lib, X, model, acq, seed, S=5000, B=50, refit_every=10, max_queries=70000, watch=None,
                  c=1.0, m_trees=10, n_estimators=100, min_samples_leaf=3, lr_C=0.5, n_jobs=4, return_order=False, seed_exclude=None):
    if model == "lr" and acq in ("ucb", "ts"):
        raise ValueError("ucb / ts need an ensemble (use model='rf')")
    rng = np.random.default_rng(seed)
    n = lib.n
    y = lib.labels
    perm = rng.permutation(n)
    if seed_exclude is not None:   # rows that may not be in the seed (they stay in the pool); default None = unchanged behaviour
        ex = seed_exclude[perm]
        perm = np.concatenate([perm[~ex], perm[ex]])
    seed_idx = perm[:S]

    avail = np.ones(n, dtype=bool)
    avail[seed_idx] = False
    labeled = list(seed_idx)
    queried, cum = len(seed_idx), int(y[seed_idx].sum())
    Q, C = [queried], [cum]
    order_log = [seed_idx] if return_order else None

    lookup = None
    found = np.zeros(0, dtype=np.int32)
    if watch is not None and len(watch):
        lookup = np.full(n, -1, dtype=np.int32)
        lookup[np.asarray(watch)] = np.arange(len(watch), dtype=np.int32)
        found = np.full(len(watch), -1, dtype=np.int32)
        w = lookup[seed_idx]
        found[w[w >= 0]] = S

    if model == "lr":
        from sklearn.linear_model import LogisticRegression

        mu, sd = X.mean(0), X.std(0) + 1e-6
        Xs = ((X - mu) / sd).astype(np.float32)
        lr = LogisticRegression(C=lr_C, class_weight="balanced", max_iter=200, warm_start=True)
    else:
        Xs = X

    fit_seconds, n_fit, t = 0.0, 0, 0
    scores = P = None
    while queried < max_queries and avail.any():
        if t % refit_every == 0:
            t0 = time.time()
            idx = np.asarray(labeled)
            rows = np.flatnonzero(avail)
            if model == "lr":
                lr.fit(Xs[idx], y[idx])
                p = np.zeros(n, dtype=np.float32)
                p[rows] = lr.predict_proba(Xs[rows])[:, 1]
                scores = p if acq == "greedy" else -np.abs(p - 0.5)
            else:
                rf = _make_rf(n_estimators, min_samples_leaf, seed * 1000 + t, n_jobs)
                rf.fit(X[idx], y[idx])
                if acq in ("ucb", "ts"):
                    P = _tree_probs(rf, X, rows, n_jobs)
                    if acq == "ucb":
                        scores = P.mean(1) + c * P.std(1)
                else:
                    p = np.zeros(n, dtype=np.float32)
                    p[rows] = rf.predict_proba(X[rows])[:, 1]
                    scores = p if acq == "greedy" else -np.abs(p - 0.5)
            fit_seconds += time.time() - t0
            n_fit += 1
        if model == "rf" and acq == "ts":
            sub = rng.choice(P.shape[1], size=min(m_trees, P.shape[1]), replace=False)
            sc = P[:, sub].mean(1)
        else:
            sc = scores
        cand = np.flatnonzero(avail)
        s = sc[cand] + 1e-7 * rng.random(len(cand))            # random tie-breaking
        b = min(B, len(cand))
        picked = cand[np.argpartition(-s, b - 1)[:b]] if len(cand) > b else cand
        avail[picked] = False
        labeled.extend(picked.tolist())
        queried += len(picked)
        cum += int(y[picked].sum())
        Q.append(queried)
        C.append(cum)
        if lookup is not None:
            w = lookup[picked]
            found[w[w >= 0]] = queried
        if return_order:
            order_log.append(picked)
        t += 1

    out = {"queried": np.asarray(Q, dtype=np.int64), "cum": np.asarray(C, dtype=np.int64), "found_at": found,
           "seed_idx": seed_idx, "fit_seconds": fit_seconds, "n_fit": n_fit}
    if return_order:
        out["order"] = np.concatenate(order_log)
    return out
