"""How similar are the 50 peptides picked in one batch? Forest (cluster-blind, greedy) vs TS / UCB (cluster arms, k = 50, equal allocation) vs uniform random, BRD3, seeds 101-103, first 25,000 queries.

Forest = the forest of the ablation / stress tests (100 trees, min_samples_leaf 1, refit every 10 rounds, F2 features, seed 5,000); arms = Beta(0.01, 0.99) prior, UCB c = 2.0 (as results/1-policies/work/run_policies.sh).
Per batch (50 picks after the 5,000-peptide seed) it records
  neardup : fraction of picks with a shift-neighbour (sliding-window shift <= 3 residues) among the other picks of the same batch        (as redundancy.py)
  groups  : number of connected groups of the shift-neighbour graph of the batch (50 = all picks unrelated; fewer = picks come from the same sliding-window neighbourhoods)
  clusters: number of distinct CD-HIT 0.5 clusters among the 50 picks
  and the picks (library row indices) of every batch, for the example heatmap.
Output ../data/batch_similarity.npz (never overwritten).  Usage: nice -n 10 python batch_similarity_forest.py [--seeds 101,102,103] [--out FILE]     (conda environment 'thompson'; ~15-20 min on one CPU, dominated by the forest)"""
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent / 'scripts'))          # results/scripts/tsbench
from tsbench import Library, make_policy, run, run_random
from tsbench import surrogate as _sur
from tsbench.features import build


def _patch_rf():
    """Forest of the ablations: balanced_subsample, max_samples 0.3, and a predict_proba that survives a seed set without positives (only one class seen so far)."""
    from sklearn.ensemble import RandomForestClassifier

    class SafeRFC(RandomForestClassifier):
        def predict_proba(self, X):
            p = super().predict_proba(X)
            if p.shape[1] == 1:
                one = np.ones(len(X)); zero = np.zeros(len(X))
                return np.column_stack([one, zero]) if self.classes_[0] == 0 else np.column_stack([zero, one])
            return p

    def make(n_estimators, min_samples_leaf, seed, n_jobs):
        return SafeRFC(n_estimators=n_estimators, max_features='sqrt', min_samples_leaf=min_samples_leaf, max_samples=0.3,
                       class_weight='balanced_subsample', n_jobs=n_jobs, random_state=seed)
    _sur._make_rf = make


_patch_rf()

ap = argparse.ArgumentParser(); ap.add_argument('--seeds', default='101,102,103'); ap.add_argument('--out', type=Path, default=ROOT / 'data' / 'batch_similarity.npz'); args = ap.parse_args()
if args.out.exists(): sys.exit(f'{args.out} exists; not overwritten')
SEEDS, S, B, MAXQ, MAXSHIFT = [int(x) for x in args.seeds.split(',')], 5000, 50, 25000, 3
lab = pd.read_csv(ROOT.parent / '1-policies' / 'data' / 'library_cdhit_0.5.csv'); seq = pd.read_csv(ROOT / 'data' / 'sequences.csv').set_index('id')['sequence']; seqs = seq.loc[lab['id']].values; n = len(seqs)
lib = Library(lab['label'].values, lab['cluster'].values); X = build(seqs)[1].astype(np.float32); cl = lib.groups
nbrs = [[] for _ in range(n)]                    # shift neighbours by hashing (library ids are not in protein order), as redundancy.py
for d in range(1, MAXSHIFT + 1):
    head = {}
    for j in range(n): head.setdefault(seqs[j][:-d], []).append(j)
    for i in range(n):
        for j in head.get(seqs[i][d:], ()):
            if j != i: nbrs[i].append(j); nbrs[j].append(i)
nbrs = [set(x) for x in nbrs]

def batch_stats(picks):
    ps = set(int(p) for p in picks); idx = {p: k for k, p in enumerate(picks)}; parent = list(range(len(picks)))
    def find(a):
        while parent[a] != a: parent[a] = parent[parent[a]]; a = parent[a]
        return a
    dup = 0
    for p in picks:
        nb = nbrs[int(p)] & ps
        dup += bool(nb)
        for q in nb: parent[find(idx[int(p)])] = find(idx[q])
    return dup / len(picks), len({find(k) for k in range(len(picks))}), len(set(cl[picks]))

def per_batch(order):
    nb = (MAXQ - S) // B; out = np.zeros((nb, 3)); picks = np.zeros((nb, B), dtype=np.int64)
    for t in range(nb):
        p = order[S + t * B: S + (t + 1) * B]; picks[t] = p; out[t] = batch_stats(p)
    return out, picks

res = {}
for s in SEEDS:
    runs = {'random': run_random(lib, s, S=S, B=B, max_queries=MAXQ, return_order=True)}
    for m, kw in (('ts', {}), ('ucb', dict(c=2.0))):
        runs[m] = run(lib, make_policy(m, **kw), s, S=S, B=B, k=50, alloc='equal', alpha0=0.01, beta0=0.99, max_queries=MAXQ, return_order=True)
    runs['forest'] = _sur.run_surrogate(lib, X, 'rf', 'greedy', s, S=S, B=B, refit_every=10, max_queries=MAXQ, min_samples_leaf=1, n_jobs=1, return_order=True)
    for name, o in runs.items():
        st, pk = per_batch(o['order']); res.setdefault(name, []).append((st, pk))
    print('seed', s, 'done', {k: np.round(v[-1][0].mean(0), 3).tolist() for k, v in res.items()}, flush=True)
out = {}
for name, v in res.items():
    out[f'{name}__stats'] = np.stack([a for a, _ in v]); out[f'{name}__picks'] = np.stack([b for _, b in v])
np.savez_compressed(args.out, **out, seeds=np.array(SEEDS), S=S, B=B)
for name in res: print(name, 'near-dup %.3f groups %.1f clusters %.1f' % tuple(np.mean([a.mean(0) for a, _ in res[name]], 0)))
