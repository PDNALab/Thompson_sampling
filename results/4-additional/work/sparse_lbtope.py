"""Sparse LBtope: simulate a sparse library by randomly HOLDING OUT positives of LBtope Fixed until the global hit rate is ~5 % (5 independent draws).
build : for ds in 1..5: keep all 20,569 negatives and a random subset of P' = round(0.05 * 20,569 / 0.95) = 1,083 of the 12,059 positives (rng seed 5000 + ds); the held-out positives
        are REMOVED from the library (not relabelled); rows keep the original order. CD-HIT 0.5 (-n 3 -l 5 as for BRD3) is re-run on each reduced library.
        Output data/sparse/ds<i>/{library.csv (orig_row, id, sequence, label), library.fa, cdhit0.5.npy, cluster_stats.csv}.  orig_row indexes data/lbtope/fixed.csv / data/lbtope/fixed__F470.npy.
tune  : TS (equal allocation, k = 50 = the structural maximum) prior grid on seeds 1-20 (objective: paper-definition AUC), mean {0.01, 0.02, 0.05, 0.1, 0.2} x strength {0.1, 1, 10, 100};
        evaluation of the winner on seeds 101-200 vs uniform random (paired). Output data/sparse/ds<i>/tune.json.   (the prior has to be re-tuned: the base rate moved from 0.37 to 0.05)
Usage: python sparse_lbtope.py build | python sparse_lbtope.py tune --ds I"""
import argparse, json, subprocess, sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; BENCH = HERE.parents[1]; sys.path.insert(0, str(BENCH / 'scripts'))
D = HERE.parent / 'data' / 'lbtope'; SP = D / 'sparse'; CH = '/orange/alberto.perezant/jokent.gaza/envs/thompson/bin/cd-hit'; TARGET = 0.05; NDS = 5
MEANS, STRENGTHS = [0.01, 0.02, 0.05, 0.1, 0.2], [0.1, 1, 10, 100]; TUNE, EVAL = range(1, 21), range(101, 201); B = 50
def build():
    import cluster_stats as C
    d = pd.read_csv(D / 'fixed.csv'); neg = np.flatnonzero(d.label.values == 0); pos = np.flatnonzero(d.label.values == 1); keep_p = int(round(TARGET * len(neg) / (1 - TARGET)))
    print(f'LBtope Fixed: {len(pos)} positives, {len(neg)} negatives; keep {keep_p} positives -> N {len(neg) + keep_p}, hit rate {keep_p / (len(neg) + keep_p):.4f}', flush=True); rows = []
    for i in range(1, NDS + 1):
        rng = np.random.default_rng(5000 + i); idx = np.sort(np.concatenate([neg, rng.choice(pos, keep_p, replace=False)])); w = SP / f'ds{i}'; w.mkdir(parents=True, exist_ok=True)
        lib = d.iloc[idx][['id', 'sequence', 'label']].reset_index(drop=True); lib.insert(0, 'orig_row', idx); lib['id'] = [f'ds{i}_{j:05d}' for j in range(len(lib))]; lib.to_csv(w / 'library.csv', index=False)
        with open(w / 'library.fa', 'w') as fh:
            for a, s in zip(lib.id, lib.sequence): fh.write(f'>{a}\n{s}\n')
        r = subprocess.run([CH, '-i', str(w / 'library.fa'), '-o', str(w / 'cdhit0.5'), '-c', '0.5', '-n', '3', '-l', '5', '-M', '0', '-T', '1', '-d', '0'], capture_output=True, text=True); assert r.returncode == 0, r.stderr[-300:]
        cid, k = {}, -1
        for line in open(str(w / 'cdhit0.5') + '.clstr'):
            if line.startswith('>Cluster'): k += 1
            else: cid[line.split('>')[1].split('...')[0]] = k
        g = np.array([cid[a] for a in lib.id]); np.save(w / 'cdhit0.5.npy', g); y = lib.label.values.astype(float)
        st, cnt, _ = C.stats(f'sparse_ds{i}', 'cdhit_0.5', g, y, np.random.default_rng(1)); st['n_clusters'] = len(cnt); st['ds'] = i; rows.append(st)
        print(f"ds{i}: N {len(lib)}, positives {int(y.sum())} ({y.mean():.2%}), clusters {len(cnt)}, singletons {st['singleton_frac']:.0%}, label info {st['excess_mi_frac_of_H']:.1%} of H, clusters for 50 % of positives "
              f"{st['arms_for_50pct_pos']} (random partition {st['arms_for_50pct_pos_randpart']:.0f}), P(nbr positive | positive) {st['loo_rate_given_pos']:.2f} vs | negative {st['loo_rate_given_neg']:.2f}", flush=True)
    pd.DataFrame(rows).to_csv(SP / 'cluster_stats.csv', index=False)
def tune(i):
    from tsbench import Library, make_policy, make_grid, run, run_random
    from tsbench.sim import to_grid
    from tsbench.metrics import auc_paper, queries_to_fraction
    w = SP / f'ds{i}'; lib0 = pd.read_csv(w / 'library.csv'); lib = Library(lib0.label.values, np.load(w / 'cdhit0.5.npy')); n = lib.n; P = lib.total_binders; S = int(round(0.035 * n)); grid = make_grid(S, n, B)
    def metr(o): c = to_grid(o['queried'], o['cum'], grid); return dict(auc=float(auc_paper(grid, c[None], P)[0]), q50=float(queries_to_fraction(grid, c[None], P, 0.5)[0]))
    go = lambda cfg, s, L=lib: run(L, make_policy('ts'), s, S=S, B=B, k=50, alloc='equal', alpha0=cfg['m'] * cfg['s'], beta0=(1 - cfg['m']) * cfg['s'])      # noqa: E731
    res = []
    for m in MEANS:
        for s_ in STRENGTHS:
            cfg = dict(m=m, s=s_); r = [metr(go(cfg, s)) for s in TUNE]; res.append(dict(cfg=cfg, auc=float(np.mean([x['auc'] for x in r])), q50=float(np.nanmean([x['q50'] for x in r])))); print(cfg, round(res[-1]['auc'], 4), flush=True)
    best = max(res, key=lambda r: r['auc']); cfg = best['cfg']; mr = [metr(run_random(lib, s, S=S, B=B)) for s in EVAL]; mb = [metr(go(cfg, s)) for s in EVAL]; mn = [metr(go(cfg, s, lib.random_partition(9001 + s % 10))) for s in EVAL]
    q = lambda ms: float(np.nanmean([x['q50'] for x in ms]))                                                                                                                  # noqa: E731
    ev = dict(random_q50=q(mr), best_q50=q(mb), null_q50=q(mn), fold=(n / 2) / q(mb), ceiling_fold=(n / 2) / (0.5 * P), best_auc=float(np.mean([x['auc'] for x in mb])))
    edge = {'m': cfg['m'] in (MEANS[0], MEANS[-1]), 's': cfg['s'] in (STRENGTHS[0], STRENGTHS[-1])}; print('best', cfg, 'edges', edge, 'eval', json.dumps(ev))
    cfg_out = dict(alpha0=cfg['m'] * cfg['s'], beta0=(1 - cfg['m']) * cfg['s'], k=50, alloc='equal'); json.dump(dict(ds=i, N=n, positives=P, S=S, results=res, best=best, prior=cfg_out, at_grid_edge=edge, eval=ev), open(w / 'tune.json', 'w'), indent=1)
if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('stage'); ap.add_argument('--ds', type=int); a = ap.parse_args()
    build() if a.stage == 'build' else tune(a.ds)
