"""How does CD-HIT (as for BRD3) cluster the LBtope sets?  Sets (files added by the user, from webs.iiitd.edu.in/raghava/lbtope):
  fixed     LBtope_Fixed-{positive,negative}.txt                        20-mers (12,063 pos / 20,589 neg)
  varnr     Lbtope_variable_non-redundant-{positive,negative}.txt       5-50 mers, already CD-HIT-80 % non-redundant by the authors
Writes data/lbtope/<set>.csv (id, sequence, label), runs cd-hit at 0.5 (-n 3 as for BRD3), 0.6 (-n 4), 0.7 / 0.8 (-n 5) with -l 5 -M 0 -T 1 -d 0,
and computes the diagnostics of the other benchmarks (cluster_stats.stats): number of clusters, singleton share, permutation-corrected
label information, clusters holding 50 % of the positives vs a size-matched random partition, neighbour label agreement (leave-one-out).
Non-standard sequences are dropped first (v2, 2026-11-17; v1 kept the 24 of 'fixed' and also ran 'varnr': data/lbtope/*_v1_both_sets.csv). Usage: python cluster_lbtope.py [fixed] [varnr].
Output data/lbtope/cluster_stats.csv, data/lbtope/<set>__cdhit<c>.npy (cluster id per row)."""
import subprocess, sys, shutil
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; BENCH = HERE.parents[1]; sys.path.insert(0, str(BENCH / 'scripts'))
import cluster_stats as C
CH = '/orange/alberto.perezant/jokent.gaza/envs/thompson/bin/cd-hit'; OUT = HERE.parent / 'data' / 'lbtope'; OUT.mkdir(parents=True, exist_ok=True)
SETS = {'fixed': ('LBtope_Fixed-positive.txt', 'LBtope_Fixed-negative.txt'), 'varnr': ('Lbtope_variable_non-redundant-positive.txt', 'LBtope_variable_non-redundant-negative.txt')}
WORD = {0.4: 2, 0.5: 3, 0.6: 4, 0.7: 5, 0.8: 5, 0.9: 5}; AA = set('ACDEFGHIKLMNPQRSTVWY')
rng = np.random.default_rng(1); rows = []
ONLY = [a for a in sys.argv[1:]] or ['fixed']
for name, (pf, nf) in {k: v for k, v in SETS.items() if k in ONLY}.items():
    P = [s.strip().upper() for s in open(OUT / 'raw' / pf) if s.strip()]; Nn = [s.strip().upper() for s in open(OUT / 'raw' / nf) if s.strip()]
    d = pd.DataFrame({'sequence': P + Nn, 'label': [1] * len(P) + [0] * len(Nn)}); d.insert(0, 'id', [f'{name}{i:05d}' for i in range(len(d))])
    ok = d.sequence.map(lambda s: set(s) <= AA); nonstd = int((~ok).sum())
    if nonstd:
        print(f'  dropping {nonstd} sequences with non-standard residues (the BRD3 feature code rejects them): positives {int(d.label[~ok].sum())}, negatives {int((1 - d.label[~ok]).sum())}')
        d = d[ok].reset_index(drop=True); d['id'] = [f'{name}{i:05d}' for i in range(len(d))]; P = [s_ for s_ in d.sequence[d.label == 1]]; Nn = [s_ for s_ in d.sequence[d.label == 0]]
    dup = int(d.sequence.duplicated().sum()); both = len(set(P) & set(Nn))
    print(f'{name}: {len(P)} pos + {len(Nn)} neg = {len(d)}; length {d.sequence.str.len().min()}-{d.sequence.str.len().max()}; non-standard {nonstd}; duplicated sequences {dup}; in both classes {both}', flush=True)
    d.to_csv(OUT / f'{name}.csv', index=False); y = d.label.values.astype(float)
    with open(OUT / f'{name}.fa', 'w') as fh:
        for i, s in zip(d.id, d.sequence): fh.write(f'>{i}\n{s}\n')
    for c, n in WORD.items():
        pre = OUT / f'{name}__cdhit{c}'
        r = subprocess.run([CH, '-i', str(OUT / f'{name}.fa'), '-o', str(pre), '-c', str(c), '-n', str(n), '-l', '5', '-M', '0', '-T', '1', '-d', '0'], capture_output=True, text=True)
        if r.returncode: print('cd-hit failed', name, c, r.stderr[-300:]); continue
        cid, k = {}, -1
        for line in open(str(pre) + '.clstr'):
            if line.startswith('>Cluster'): k += 1
            else: cid[line.split('>')[1].split('...')[0]] = k
        g = np.array([cid.get(i, -1) for i in d.id]); assert (g >= 0).all(), 'sequences dropped by cd-hit'
        np.save(OUT / f'{name}__cdhit{c}.npy', g)
        st, cnt, _ = C.stats(f'lbtope_{name}', f'cdhit_{c}', g, y, rng); st['n_clusters'] = len(cnt); rows.append(st)
        print(f"  cd-hit {c} (-n {n}): clusters {len(cnt)}, singletons {st['singleton_frac']:.0%}, max size {st['size_max']}, label info {st['excess_mi_frac_of_H']:.1%} of H, "
              f"clusters for 50 % of positives {st['arms_for_50pct_pos']} (random partition {st['arms_for_50pct_pos_randpart']:.0f}), "
              f"P(neighbour positive | positive) {st['loo_rate_given_pos']:.2f} vs | negative {st['loo_rate_given_neg']:.2f} (base {st['base_rate']:.2f})", flush=True)
pd.DataFrame(rows).to_csv(OUT / 'cluster_stats.csv', index=False)
