"""Sparse LBtope: TS + RF picker, RF-greedy and the feature-vector ablation on each of the 5 sparse libraries (sparse_lbtope.py build).
Loops = lbtope_hybrid.run_hybrid (TS + forest picker; validated against 2-surrogate/work/bandit_rf.run) and rf_greedy_loop.run (RF-greedy); features = rows orig_row of data/lbtope/fixed__F470.npy.
TS settings (NOT tuned, the same for every library): the BRD3 settings of results/1-policies, Beta(0.01, 0.99), k = 50, equal allocation (sparse_lbtope.py tune exists but was not used). Seed set = round(0.035 N) = 758, cap = 50 % of N = 10,826 queries.
Tasks (355 = 5 libraries x 71): per library {hybrid, rf} x 11 feature sets x seeds 101-103 (66) then hybrid_random control (random pick, no forest) seeds 101-105 (5).
Usage: python sparse_hybrid.py --task I [--n_jobs 1] | --list | --ds D --method M --cond C --seed S [--alpha0 a --beta0 b --max_frac f --out DIR]
Output data/sparse/ds<D>/hybrid/sparse_<method>_<cond>_cap<f>_seed<S>.npz (+ .ckpt)."""
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; BENCH = HERE.parents[1]; sys.path.insert(0, str(BENCH / 'scripts')); sys.path.insert(0, str(HERE))
import lbtope_hybrid as L           # noqa: E402
import rf_greedy_loop as G           # noqa: E402
D = L.D; SP = D / 'sparse'; PER = 2 * len(L.CONDS) * 3 + 5
def tasks():
    t = []
    for ds in range(1, 6):
        t += [(ds, m, c, s, 0.5) for m in ('hybrid', 'rf') for c in L.CONDS for s in (101, 102, 103)] + [(ds, 'hybrid_random', 'full', s, 0.5) for s in range(101, 106)]
    return t
def load(ds, cond):
    lib0 = pd.read_csv(SP / f'ds{ds}' / 'library.csv'); y = lib0.label.values.astype(np.int8); X0 = L.features()[lib0.orig_row.values]
    kind, b = (None, None) if cond == 'full' else cond.split('_'); cols = L.BL if cond == 'full' else ([c for c in L.BL if c != b] if kind == 'minus' else [b])
    return L.Library(y, np.load(SP / f'ds{ds}' / 'cdhit0.5.npy')), np.ascontiguousarray(np.hstack([X0[:, L.SL[c]] for c in cols])), y
if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--task', type=int); ap.add_argument('--list', action='store_true'); ap.add_argument('--ds', type=int); ap.add_argument('--method'); ap.add_argument('--cond', default='full')
    ap.add_argument('--seed', type=int); ap.add_argument('--max_frac', type=float, default=0.5); ap.add_argument('--n_jobs', type=int, default=1); ap.add_argument('--alpha0', type=float); ap.add_argument('--beta0', type=float); ap.add_argument('--out', type=Path)
    a = ap.parse_args(); T = tasks()
    if a.list: print(len(T), 'tasks,', PER, 'per library'); sys.exit(0)
    if a.task is not None: a.ds, a.method, a.cond, a.seed, a.max_frac = T[a.task]
    if a.alpha0 is None: a.alpha0, a.beta0 = 0.01, 0.99          # BRD3 prior of results/1-policies (user's choice 2026-11-18)
    lib, X, y = load(a.ds, a.cond); n = lib.n; S = int(round(0.035 * n)); mq = int(round(a.max_frac * n)); out = a.out or SP / f'ds{a.ds}' / 'hybrid'; out.mkdir(parents=True, exist_ok=True)
    f = out / f'sparse_{a.method}_{a.cond}_cap{a.max_frac}_seed{a.seed}.npz'; print(f.name, f'ds{a.ds} N {n} positives {int(y.sum())} features {X.shape[1]} max_queries {mq} prior Beta({a.alpha0}, {a.beta0})', flush=True)
    if a.method == 'rf': res = G.run(X, y, a.seed, S, max_queries=mq, n_jobs=a.n_jobs, ckpt=f.with_suffix('.ckpt'))
    else: res = L.run_hybrid(lib, X, a.seed, within='rf' if a.method == 'hybrid' else 'random', S=S, k=50, alpha0=a.alpha0, beta0=a.beta0, alloc='equal', max_queries=mq, n_jobs=a.n_jobs, ckpt=f.with_suffix('.ckpt'))
    np.savez_compressed(f, **res, ds=a.ds, seed=a.seed, method=a.method, cond=a.cond, max_frac=a.max_frac, alpha0=a.alpha0, beta0=a.beta0); print('saved', f)
