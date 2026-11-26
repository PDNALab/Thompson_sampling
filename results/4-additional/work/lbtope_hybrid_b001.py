"""Redo of the full-library LBtope TS + RF-picker runs with the BRD3 prior Beta(0.01, 0.99) (k = 50, equal allocation, NOT tuned) instead of the LBtope-tuned Beta(0.1, 0.9).
Same loop and library as lbtope_hybrid.py (imported); only the prior and the output folder differ: data/hybrid_b001/ (the tuned-prior outputs in data/hybrid/ are kept).
RF-greedy does not depend on the bandit prior, so its runs (data/hybrid/lbtope_rf_<cond>_cap0.5_seed<S>.npz) are reused as they are.
Tasks (65): 0-54 hybrid (TS + forest picker) x 11 feature sets x seeds 101-105; 55-64 hybrid_random (TS with a random pick; control) seeds 101-110; cap 50 % of N.
Usage: python lbtope_hybrid_b001.py --task I [--n_jobs 1] | --list"""
import argparse, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import lbtope_hybrid as L          # noqa: E402
OUT = L.D / 'hybrid_b001'; A0, B0 = 0.01, 0.99
TASKS = [('hybrid', c, s, 0.5) for c in L.CONDS for s in range(101, 106)] + [('hybrid_random', 'full', s, 0.5) for s in range(101, 111)]
if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--task', type=int); ap.add_argument('--n_jobs', type=int, default=1); ap.add_argument('--list', action='store_true'); ap.add_argument('--out', type=Path, default=OUT); a = ap.parse_args()
    if a.list: print(len(TASKS), 'tasks'); sys.exit(0)
    method, cond, seed, frac = TASKS[a.task]; lib, X, y = L.load(cond); n = lib.n; S = int(round(0.035 * n)); mq = int(round(frac * n)); a.out.mkdir(parents=True, exist_ok=True)
    f = a.out / f'lbtope_{method}_{cond}_cap{frac}_seed{seed}.npz'; print(f.name, 'N', n, 'features', X.shape[1], 'max_queries', mq, f'prior Beta({A0}, {B0})', flush=True)
    res = L.run_hybrid(lib, X, seed, within='rf' if method == 'hybrid' else 'random', S=S, k=50, alpha0=A0, beta0=B0, alloc='equal', max_queries=mq, n_jobs=a.n_jobs, ckpt=f.with_suffix('.ckpt'))
    np.savez_compressed(f, **res, seed=seed, method=method, cond=cond, max_frac=frac, alpha0=A0, beta0=B0); print('saved', f)
