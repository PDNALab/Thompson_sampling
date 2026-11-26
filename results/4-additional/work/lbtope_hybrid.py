"""LBtope Fixed - Thompson sampling (tuned prior Beta(0.1, 0.9), k = 50, equal allocation) with a random-forest picker inside the chosen clusters, and the feature-vector ablation of
subfigure D of results/2-surrogate/figures/1-forest.png.

Hybrid loop = 2-surrogate/work/bandit_rf.py (copied; only the library, prior, allocation and k are parameters). Per round (batch B = 50):
  1. TS scores every non-exhausted CD-HIT 0.5 cluster with one Beta(alpha, beta) draw and takes the top k;  2. a RandomForestClassifier() (sklearn defaults) is refit on
  EVERY labelled peptide; 3. in each chosen cluster the unqueried peptide with the highest forest P(positive) is queried (random tie-break);  4. posteriors updated.
Defaults = the TS settings TUNED on LBtope Fixed at CD-HIT 0.5 (tune_lbtope.py, seeds 1-20): Beta(0.1, 0.9), k = 50, EQUAL allocation (the paper's Beta(0.2, 9.8) with proportional
allocation is equivalent except for the prior: at k = B = 50 the two allocations gave identical picks).
Methods: hybrid (TS + forest picker), hybrid_random (same loop, random pick = the tuned arm policy, needs no features: control that must reproduce tsbench; its q50 fold should be ~1.58),
         rf (RF-greedy, rf_greedy_loop.run, no clusters).   Features: data/lbtope/fixed__F470.npy (AAC 0:20, DPC 20:420, PHYS 420:430, NTERM 430:450, CTERM 450:470).
Conditions (as rf_ablation.py): full | minus_<B> | only_<B>, B in AAC DPC PHYS NTERM CTERM.   (mtry = sqrt(#columns) changes with the condition: full 21, minus_DPC 8, only_AAC 4.)
Usage: python lbtope_hybrid.py --method hybrid|hybrid_random|rf --seed 101 [--cond full] [--max_frac 0.2] [--n_jobs 2] [--out data/hybrid] [--task I]
Output <out>/lbtope_<method>_<cond>_cap<frac>_seed<S>.npz  (order, hits, fit_s, score_s, top_p ...) + .ckpt every 100 rounds."""
import argparse, pickle, sys, time
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; BENCH = HERE.parents[1]; sys.path.insert(0, str(BENCH / 'scripts')); sys.path.insert(0, str(HERE))
from tsbench import Library, make_policy      # noqa: E402
from tsbench.policies import State            # noqa: E402
from tsbench.sim import _alloc                # noqa: E402
SL = dict(AAC=slice(0, 20), DPC=slice(20, 420), PHYS=slice(420, 430), NTERM=slice(430, 450), CTERM=slice(450, 470)); BL = list(SL)
CONDS = ['full'] + [f'minus_{b}' for b in BL] + [f'only_{b}' for b in BL]
D = HERE.parent / 'data' / 'lbtope'
def features():
    """The 470 BRD3 features of data/lbtope/fixed.csv (AAC 20 + DPC 400 + PHYS 10 + NTERM 20 + CTERM 20), cached in data/lbtope/fixed__F470.npy (git-ignored, rebuilt on demand)."""
    cache = D / 'fixed__F470.npy'
    if cache.exists(): return np.load(cache)
    from tsbench.features import build, terminal
    seqs = pd.read_csv(D / 'fixed.csv').sequence.values; assert all(len(s) == 20 for s in seqs); _, f2 = build(seqs)
    X = np.hstack([f2, terminal(seqs, 'N'), terminal(seqs, 'C')]).astype(np.float32); assert X.shape == (len(seqs), 470) and np.isfinite(X).all(); np.save(cache, X); return X
def load(cond):
    d = pd.read_csv(D / 'fixed.csv'); y = d.label.values.astype(np.int8); g = np.load(D / 'fixed__cdhit0.5.npy'); X = features()
    kind, b = (None, None) if cond == 'full' else cond.split('_'); cols = BL if cond == 'full' else ([c for c in BL if c != b] if kind == 'minus' else [b])
    return Library(y, g), np.ascontiguousarray(np.hstack([X[:, SL[c]] for c in cols])), y
def run_hybrid(lib, X, seed, within='rf', S=0, B=50, k=50, alpha0=0.1, beta0=0.9, alloc='equal', policy_name='ts', max_queries=None, n_jobs=1, ckpt=None, ckpt_every=100, log_every=50):
    from sklearn.ensemble import RandomForestClassifier
    policy = make_policy(policy_name); rng = np.random.default_rng(seed); n, G, y, grp = lib.n, lib.n_groups, lib.labels, lib.groups
    seed_idx = rng.permutation(n)[:S]; members_all = np.argsort(grp, kind='stable'); size = np.bincount(grp, minlength=G); end = np.cumsum(size)
    members = [members_all[e - s:e] for s, e in zip(size, end)]
    avail = np.ones(n, dtype=bool); avail[seed_idx] = False; order = [seed_idx.astype(np.int32)]; fit_s, score_s, top_p, hits, phase = [], [], [], [], []; t = 0
    if ckpt is not None and Path(ckpt).exists():
        c = pickle.load(open(ckpt, 'rb')); rng.bit_generator.state = c['rng']; order, fit_s, score_s, top_p, hits, phase, t = c['order'], c['fit_s'], c['score_s'], c['top_p'], c['hits'], c['phase'], c['t']
        avail[np.concatenate(order)] = False; print(f'  resumed from {ckpt} at round {t}', flush=True)
    labelled = np.concatenate(order); yl, gl = y[labelled].astype(np.float64), grp[labelled]
    alpha = alpha0 + np.bincount(gl, weights=yl, minlength=G); beta = beta0 + np.bincount(gl, weights=1 - yl, minlength=G); rem = size - np.bincount(gl, minlength=G)
    queried, cum = len(labelled), int(y[labelled].sum()); max_queries = max_queries or n
    print(f'{policy_name} + {within}, seed {seed}: S {S}, k {k}, {alloc} allocation, Beta({alpha0}, {beta0}); start {queried} queries, {cum} positives', flush=True); last_ck = t
    while queried < max_queries and rem.sum() > 0:
        if ckpt is not None and t % ckpt_every == 0 and t > last_ck:
            tmp = Path(str(ckpt) + '.tmp'); pickle.dump(dict(rng=rng.bit_generator.state, order=order, fit_s=fit_s, score_s=score_s, top_p=top_p, hits=hits, phase=phase, t=t), open(tmp, 'wb')); tmp.replace(ckpt); last_ck = t
        act = np.flatnonzero(rem > 0); b = int(min(B, rem.sum())); m = min(k, len(act), b); st = State(alpha[act], beta[act], act, rng, t)
        sc = policy.scores(st); sel = policy.choose(sc, m, st); chosen = act[sel]; quota = _alloc(sc[sel], b, alloc)
        rf = None; ready = within == 'rf' and 0 < cum < queried; phase.append(int(ready))
        if within == 'rf' and not ready: fit_s.append(0.0)
        if ready:
            t0 = time.time(); rf = RandomForestClassifier(random_state=seed * 100000 + t, n_jobs=n_jobs).fit(X[labelled], y[labelled]); fit_s.append(time.time() - t0)
        t0 = time.time(); cache = {}
        def ranked(c):
            if c not in cache:
                cand = members[c][avail[members[c]]]; p = rf.predict_proba(X[cand])[:, 1] if rf is not None else np.zeros(len(cand))
                o = np.argsort(-(p + 1e-7 * rng.random(len(cand))), kind='stable'); cache[c] = (cand[o], p[o])
            return cache[c]
        picks, ps, leftover = [], [], 0
        for c, qc in zip(chosen, quota):
            rows, p = ranked(c); take = int(min(qc, len(rows))); picks.append(rows[:take]); ps.append(p[:take]); leftover += int(qc) - take
        if leftover > 0:
            is_ch = np.zeros(len(act), dtype=bool); is_ch[sel] = True
            for pos in np.argsort(-sc):
                if is_ch[pos]: continue
                rows, p = ranked(act[pos]); take = int(min(leftover, len(rows)))
                if take > 0: picks.append(rows[:take]); ps.append(p[:take]); leftover -= take
                if leftover == 0: break
        picked = np.concatenate(picks); score_s.append(time.time() - t0); top_p.append(float(np.concatenate(ps).mean()) if ready else float('nan'))
        avail[picked] = False; yp = y[picked].astype(np.float64); np.add.at(alpha, grp[picked], yp); np.add.at(beta, grp[picked], 1.0 - yp); np.add.at(rem, grp[picked], -1)
        labelled = np.concatenate([labelled, picked]); order.append(picked.astype(np.int32)); hits.append(int(yp.sum())); queried += len(picked); cum += hits[-1]; t += 1
        if t % log_every == 0: print(f'  round {t} queried {queried} positives {cum} ({cum / lib.total_binders:.1%}) last-{log_every}-round hit rate {np.sum(hits[-log_every:]) / (B * log_every):.3f}', flush=True)
    return dict(order=np.concatenate(order), hits=np.array(hits, np.int8), fit_s=np.array(fit_s, np.float32), score_s=np.array(score_s, np.float32), top_p=np.array(top_p, np.float32), phase=np.array(phase, np.int8))
# ---- task table for the array (final, 2026-11-17: every cap is 50 % of N; v1 stopped at 20 % = 6,526 queries (files *_cap0.2_*) and its files are prefixes of these runs and are kept;
#      100 % was considered and dropped by the user before anything was submitted):
#      0-109 ablation (method x cond x seed 101-105, cap 50 %), 110-119 control hybrid_random (seeds 101-110, cap 50 %)
SEEDS = list(range(101, 106)); ABL = [(m, c, s, 0.5) for m in ('hybrid', 'rf') for c in CONDS for s in SEEDS]
TASKS = ABL + [('hybrid_random', 'full', s, 0.5) for s in range(101, 111)]
if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--method'); ap.add_argument('--cond', default='full'); ap.add_argument('--seed', type=int); ap.add_argument('--max_frac', type=float, default=0.2)
    ap.add_argument('--n_jobs', type=int, default=1); ap.add_argument('--out', type=Path, default=D / 'hybrid'); ap.add_argument('--task', type=int); ap.add_argument('--list', action='store_true')
    ap.add_argument('--alpha0', type=float, default=0.1); ap.add_argument('--beta0', type=float, default=0.9); ap.add_argument('--alloc', default='equal'); ap.add_argument('--k', type=int, default=50)
    a = ap.parse_args()
    if a.list: print(len(TASKS), 'tasks'); sys.exit(0)
    if a.task is not None: a.method, a.cond, a.seed, a.max_frac = TASKS[a.task]
    lib, X, y = load(a.cond); n = lib.n; S = int(round(0.035 * n)); mq = int(round(a.max_frac * n)); a.out.mkdir(parents=True, exist_ok=True)
    f = a.out / f'lbtope_{a.method}_{a.cond}_cap{a.max_frac}_seed{a.seed}.npz'; print(f.name, 'N', n, 'features', X.shape[1], 'max_queries', mq, flush=True)
    if a.method == 'rf':
        import rf_greedy_loop as G
        res = G.run(X, y, a.seed, S, max_queries=mq, n_jobs=a.n_jobs, ckpt=f.with_suffix('.ckpt'))
    else:
        res = run_hybrid(lib, X, a.seed, within='rf' if a.method == 'hybrid' else 'random', S=S, k=a.k, alpha0=a.alpha0, beta0=a.beta0, alloc=a.alloc, max_queries=mq, n_jobs=a.n_jobs, ckpt=f.with_suffix('.ckpt'))
    np.savez_compressed(f, **res, seed=a.seed, method=a.method, cond=a.cond, max_frac=a.max_frac); print('saved', f)
