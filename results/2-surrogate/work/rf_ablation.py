"""Feature-vector ablation of RF-greedy (method 1; plain sklearn defaults, refit every round, same seed sets).

The 470 features are five vectors ("blocks"), built by rf_greedy.load_blocks():
  AAC   20  amino-acid fractions of the 25-mer              DPC   400  dipeptide frequencies (ordered adjacent pairs / 24)
  PHYS  10  physicochemical summaries                       NTERM  20  amino-acid fractions of the first 3 residues
  CTERM 20  amino-acid fractions of the last 3 residues
Conditions: minus_<B> (leave one vector out) and only_<B> (use only that vector). The full set is rf_greedy.py itself.
Same loop and settings as rf_greedy.py (it is called, not copied); only the columns of X change. Note that sklearn's default
max_features = sqrt(p) changes with p (full 21, minus_DPC 8, only_AAC 4 ...): this is part of 'default settings'.
The run stops at --max_queries (default 30,000): the trajectory of a refit-every-round run does not depend on where it stops, so the first
30k queries of a full rf_greedy.py run are the reference.

Output OUT/rf_ablation_<cond>_seed<S>.npz (same arrays as rf_greedy.py); checkpoints every 100 rounds, rerun to resume.
Usage: python rf_ablation.py --cond minus_DPC --seed 101 [--max_queries 30000] [--n_jobs 2] [--out DIR]
"""
import argparse
from pathlib import Path

import numpy as np

import rf_greedy as R

BLOCKS = ["AAC", "DPC", "PHYS", "NTERM", "CTERM"]
CONDS = [f"minus_{b}" for b in BLOCKS] + [f"only_{b}" for b in BLOCKS]


def columns(cond):
    if cond == "full":
        return BLOCKS
    kind, b = cond.split("_")
    assert b in BLOCKS and kind in ("minus", "only"), cond
    return [c for c in BLOCKS if c != b] if kind == "minus" else [b]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", required=True, choices=["full"] + CONDS)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--max_queries", type=int, default=30000)
    ap.add_argument("--n_jobs", type=int, default=1)
    ap.add_argument("--out", type=Path, default=R.DATA)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    z = R.load_blocks()
    used = columns(a.cond)
    X = np.hstack([z[b] for b in used]).astype(np.float32)
    y = z["label"].astype(np.int8)
    print(f"{a.cond}: blocks {used}, {X.shape[1]} features, seed {a.seed}", flush=True)
    f = a.out / f"rf_ablation_{a.cond}_seed{a.seed}.npz"
    res = R.run(X, y, a.seed, max_queries=a.max_queries, n_jobs=a.n_jobs, ckpt=f.with_suffix(".ckpt"))
    np.savez_compressed(f, **res, seed=a.seed, cond=a.cond, blocks=np.array(used))
    print("saved", f)
