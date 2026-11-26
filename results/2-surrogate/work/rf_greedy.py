"""RF-greedy with plain scikit-learn defaults, refit after every batch.

Loop (pure exploitation, nothing else):
  1. seed set = the same 5,000 peptides as the arm / TS runs for this seed (default_rng(seed).permutation(n)[:S])
  2. fit RandomForestClassifier() on every labelled peptide  (defaults: 100 trees, sqrt features, min_samples_leaf 1,
     full bootstrap, no class weights)
  3. score every unlabelled peptide with P(binder), query the top B = 50
  4. repeat from 2 until max_queries
Only choice that is not sklearn's default: exact score ties (very common with unpruned trees) are broken at random.

Features: the 470 of the forest runs in the draft = AAC 20 + DPC 400 + PHYS 10 + N-term 20 + C-term 20
(blocks built on the fly by load_blocks() from data/sequences.csv with tsbench.features, rows in the order of results/1-policies/data/library_cdhit_0.5.csv; about a minute).

Output (per seed, OUT/rf_greedy_default_refit<R>_seed<S>.npz):
  order      int32  library row of every query, in order (seed set first)
  fit_s, score_s   float32 per round: seconds spent fitting / scoring
  top_p      float32 per round: mean predicted P of the 50 picks (what the model believed)
  hits       int8 per round: binders among the 50 picks (what actually happened)

Usage: python rf_greedy.py --seed 101 [--refit_every 1] [--max_queries 70000] [--n_jobs 1] [--out DIR]
"""
import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                                        # results/2-surrogate
REPO = HERE.parents[2]
DATA = ROOT / "data"
sys.path.insert(0, str(REPO / "results" / "scripts"))     # results/scripts/tsbench
from tsbench.features import build, terminal              # noqa: E402

BLOCKS = ["AAC", "DPC", "PHYS", "NTERM", "CTERM"]


def load_blocks():
    """The five feature vectors (AAC 20, DPC 400, PHYS 10, NTERM 20, CTERM 20) and the label of the BRD3 library, rows as in results/1-policies/data/library_cdhit_0.5.csv."""
    lib = pd.read_csv(REPO / "results" / "1-policies" / "data" / "library_cdhit_0.5.csv")
    seqs = pd.read_csv(DATA / "sequences.csv").set_index("id")["sequence"].loc[lib["id"]].values
    _, f2 = build(seqs)
    return dict(AAC=f2[:, :20], DPC=f2[:, 20:420], PHYS=f2[:, 420:430], NTERM=terminal(seqs, "N"), CTERM=terminal(seqs, "C"), label=lib["label"].values.astype(np.int8))


def load_features():
    z = load_blocks()
    X = np.hstack([z[b] for b in BLOCKS]).astype(np.float32)
    y = z["label"]
    assert X.shape == (142338, 470) and y.sum() == 3393 and np.isfinite(X).all()
    return X, y


def save_ckpt(path, rng, order, fit_s, score_s, top_p, hits, t):
    """Checkpoint = everything the loop needs to continue bit-for-bit (RNG state included). Written atomically."""
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "wb") as fh:
        pickle.dump(dict(rng=rng.bit_generator.state, order=order, fit_s=fit_s, score_s=score_s, top_p=top_p, hits=hits, t=t), fh)
    tmp.replace(path)


def run(X, y, seed, S=5000, B=50, refit_every=1, max_queries=70000, n_jobs=1, log_every=100, ckpt=None, ckpt_every=100):
    assert ckpt is None or ckpt_every % refit_every == 0, "checkpoints must fall on refit rounds"
    n = len(y)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)                       # same first RNG call as tsbench -> same seed set
    seed_idx = perm[:S]
    avail = np.ones(n, dtype=bool)
    avail[seed_idx] = False
    labelled = seed_idx.copy()
    order = [seed_idx.astype(np.int32)]
    fit_s, score_s, top_p, hits = [], [], [], []
    queried, cum = S, int(y[seed_idx].sum())
    print(f"seed {seed}: seed set {S}, binders in seed {cum}", flush=True)
    t, p = 0, None
    if ckpt is not None and Path(ckpt).exists():           # resume
        c = pickle.load(open(ckpt, "rb"))
        rng.bit_generator.state = c["rng"]
        order, fit_s, score_s, top_p, hits, t = c["order"], c["fit_s"], c["score_s"], c["top_p"], c["hits"], c["t"]
        labelled = np.concatenate(order)
        avail[labelled] = False
        queried, cum = len(labelled), int(y[labelled].sum())
        print(f"  resumed from {ckpt} at round {t}, queried {queried}, binders {cum}", flush=True)
    while queried < max_queries and avail.any():
        if ckpt is not None and t > 0 and t % ckpt_every == 0 and t % refit_every == 0 and len(fit_s) == t // refit_every:
            save_ckpt(ckpt, rng, order, fit_s, score_s, top_p, hits, t)
        if t % refit_every == 0:
            t0 = time.time()
            rf = RandomForestClassifier(random_state=seed * 100000 + t, n_jobs=n_jobs).fit(X[labelled], y[labelled])
            t1 = time.time()
            rows = np.flatnonzero(avail)
            p = np.full(n, -1.0, dtype=np.float32)
            p[rows] = rf.predict_proba(X[rows])[:, 1]
            fit_s.append(t1 - t0); score_s.append(time.time() - t1)
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
            print(f"  round {t} queried {queried} binders {cum} ({cum / 3393:.1%}) last-100-round hit rate "
                  f"{np.sum(hits[-log_every:]) / (B * log_every):.3f} elapsed fit+score {sum(fit_s) + sum(score_s):.0f}s", flush=True)
    return dict(order=np.concatenate(order), fit_s=np.array(fit_s, np.float32), score_s=np.array(score_s, np.float32),
                top_p=np.array(top_p, np.float32), hits=np.array(hits, np.int8))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--refit_every", type=int, default=1)
    ap.add_argument("--max_queries", type=int, default=70000)
    ap.add_argument("--n_jobs", type=int, default=1)
    ap.add_argument("--out", type=Path, default=DATA)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    X, y = load_features()
    f = a.out / f"rf_greedy_default_refit{a.refit_every}_seed{a.seed}.npz"
    res = run(X, y, a.seed, refit_every=a.refit_every, max_queries=a.max_queries, n_jobs=a.n_jobs, ckpt=f.with_suffix(".ckpt"))
    np.savez_compressed(f, **res, seed=a.seed, refit_every=a.refit_every)
    print("saved", f)
