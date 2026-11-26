"""Peptide-array library, CD-HIT 0.5 arms: RF-greedy, TS + RF picker and the feature-vector ablation with the 470 BRD3 features.
Loops are the validated ones: lbtope_hybrid.run_hybrid (TS + forest picker / random pick; copied from 2-surrogate/work/bandit_rf.py) and rf_greedy_loop.run (RF-greedy, sklearn defaults, refit every round);
imported from lbtope_hybrid.py and scripts/rf_greedy_loop.py. Library data/library.csv (165,946 12-mers, 3,335 positives = reactive in >= 10 of 100 COVID IgG samples and 0 of 10 healthy controls), arms data/cdhit0.5.npy
(5,877 clusters), features data/F470.npy.  TS settings = BRD3's (results/1-policies): Beta(0.01, 0.99), k = 50, equal allocation, NOT tuned. Seed set = round(0.035 N) = 5,808, batch 50, cap 50 % of N (82,973 queries).
Conditions as rf_ablation.py: full | minus_<B> | only_<B>, B in AAC DPC PHYS NTERM CTERM (mtry = sqrt(#columns) changes with the condition).
Task sets:  main     (20 tasks): 0-4 rf full seeds 101-105; 5-9 hybrid full seeds 101-105; 10-19 hybrid_random (TS, random pick: control) seeds 101-110
            ablation (60 tasks): {rf, hybrid} x 10 non-full feature sets x seeds 101-103  (index = method * 30 + cond * 3 + seed offset; the full-feature runs are the main set)
Usage: python peptide_hybrid.py --set main|ablation --task I [--n_jobs 2] [--max_frac f] | --list
Output data/runs/peptide_<method>_<cond>_cap<f>_seed<S>.npz (+ .ckpt every 100 rounds: rerun the same command to resume)."""
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; BENCH = HERE.parents[1]
sys.path.insert(0, str(BENCH / 'scripts')); sys.path.insert(0, str(HERE))
import lbtope_hybrid as L           # noqa: E402  (run_hybrid, SL, BL, CONDS)
import rf_greedy_loop as G           # noqa: E402  (RF-greedy loop)
D = HERE.parent / 'data' / 'covid'; A0, B0 = 0.01, 0.99
MAIN = [('rf', 'full', s, 0.5) for s in range(101, 106)] + [('hybrid', 'full', s, 0.5) for s in range(101, 106)] + [('hybrid_random', 'full', s, 0.5) for s in range(101, 111)]
ABL = [(m, c, s, 0.5) for m in ('rf', 'hybrid') for c in L.CONDS[1:] for s in (101, 102, 103)]; SETS = dict(main=MAIN, ablation=ABL)
def load(cond):
    d = pd.read_csv(D / 'library.csv', usecols=['label']); y = d.label.values.astype(np.int8); X0 = np.load(D / 'F470.npy'); g = np.load(D / 'cdhit0.5.npy')
    kind, b = (None, None) if cond == 'full' else cond.split('_'); cols = L.BL if cond == 'full' else ([c for c in L.BL if c != b] if kind == 'minus' else [b])
    return L.Library(y, g), np.ascontiguousarray(np.hstack([X0[:, L.SL[c]] for c in cols])), y
if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--set', default='main'); ap.add_argument('--task', type=int); ap.add_argument('--n_jobs', type=int, default=1); ap.add_argument('--max_frac', type=float); ap.add_argument('--list', action='store_true')
    ap.add_argument('--out', type=Path, default=D / 'runs'); a = ap.parse_args(); T = SETS[a.set]
    if a.list: print(a.set, len(T), 'tasks'); sys.exit(0)
    method, cond, seed, frac = T[a.task]; frac = a.max_frac or frac; lib, X, y = load(cond); n = lib.n; S = int(round(0.035 * n)); mq = int(round(frac * n)); a.out.mkdir(parents=True, exist_ok=True)
    f = a.out / f'peptide_{method}_{cond}_cap{frac}_seed{seed}.npz'; print(f.name, f'N {n} positives {int(y.sum())} clusters {lib.n_groups} features {X.shape[1]} seed set {S} max_queries {mq}', flush=True)
    if method == 'rf': res = G.run(X, y, seed, S, max_queries=mq, n_jobs=a.n_jobs, ckpt=f.with_suffix('.ckpt'))
    else: res = L.run_hybrid(lib, X, seed, within='rf' if method == 'hybrid' else 'random', S=S, k=50, alpha0=A0, beta0=B0, alloc='equal', max_queries=mq, n_jobs=a.n_jobs, ckpt=f.with_suffix('.ckpt'))
    np.savez_compressed(f, **res, seed=seed, method=method, cond=cond, max_frac=frac, alpha0=A0, beta0=B0); print('saved', f)
