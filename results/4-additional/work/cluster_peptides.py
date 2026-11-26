"""CD-HIT clustering of the peptide-array library at one identity threshold, cluster diagnostics, and the Thompson-sampling replay (BRD3 settings, not tuned).
CD-HIT as for BRD3 / LBtope: -c <thr> -n <word> (0.4: 2, 0.5: 3, 0.6: 4, 0.7-0.9: 5) -l 5 -M 0 -T 1 -d 0.  Diagnostics = cluster_stats.stats (permutation-corrected label information, clusters holding 50 % of positives vs a
size-matched random partition, neighbour label agreement). Replay (tsbench simulator): uniform random, TS with Beta(0.01, 0.99), k = 50, equal allocation (the BRD3 settings of results/1-policies), seeds 101-120, seed set 3.5 %,
plus the same TS on a random partition with identical cluster sizes (null, seeds 101-110). Speed-up = (N / 2) / queries to 50 % of the positives; ceiling = N / positives.
Tasks: index 0-5 = thr 0.4 ... 0.9.  Output data/cdhit<thr>.npy (cluster id per library row) and data/cluster_<thr>.json.   Usage: python cluster_peptides.py --task I [--n_sub N] [--thr T]"""
import argparse, json, subprocess, sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; BENCH = HERE.parents[1]; sys.path.insert(0, str(BENCH / 'scripts'))
D = HERE.parent / 'data' / 'covid'; CH = '/orange/alberto.perezant/jokent.gaza/envs/thompson/bin/cd-hit'; THR = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]; WORD = {0.4: 2, 0.5: 3, 0.6: 4, 0.7: 5, 0.8: 5, 0.9: 5}
if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--task', type=int); ap.add_argument('--thr', type=float); ap.add_argument('--n_sub', type=int, default=0); ap.add_argument('--out', type=Path, default=D); a = ap.parse_args()
    thr = a.thr or THR[a.task]; import cluster_stats as C
    from tsbench import Library, make_policy, make_grid, run, run_random
    from tsbench.sim import to_grid
    from tsbench.metrics import queries_to_fraction
    d = pd.read_csv(D / 'library.csv'); d = d.iloc[:a.n_sub] if a.n_sub else d; n = len(d); y = d.label.values; P = int(y.sum()); a.out.mkdir(parents=True, exist_ok=True); fa = a.out / f'library{"_sub" if a.n_sub else ""}.fa'
    with open(fa, 'w') as fh:
        for i, s in zip(d.id, d.sequence): fh.write(f'>{i}\n{s}\n')
    pre = a.out / f'cdhit{thr}{"_sub" if a.n_sub else ""}'; r = subprocess.run([CH, '-i', str(fa), '-o', str(pre), '-c', str(thr), '-n', str(WORD[thr]), '-l', '5', '-M', '0', '-T', '1', '-d', '0'], capture_output=True, text=True); assert r.returncode == 0, r.stderr[-400:]
    cid, k = {}, -1
    for line in open(str(pre) + '.clstr'):
        if line.startswith('>Cluster'): k += 1
        else: cid[line.split('>')[1].split('...')[0]] = k
    g = np.array([cid[i] for i in d.id]); np.save(str(pre) + '.npy', g)
    st, cnt, _ = C.stats('peptide_array', f'cdhit_{thr}', g, y.astype(float), np.random.default_rng(1)); lib = Library(y, g); S = int(round(0.035 * n)); grid = make_grid(S, n, 50)
    q50 = lambda o: float(queries_to_fraction(grid, to_grid(o['queried'], o['cum'], grid)[None], P, 0.5)[0])                                                        # noqa: E731
    ts = lambda L, s: run(L, make_policy('ts'), s, S=S, B=50, k=50, alloc='equal', alpha0=0.01, beta0=0.99)                                                        # noqa: E731
    qr = np.nanmean([q50(run_random(lib, s, S=S, B=50)) for s in range(101, 121)]); qt = np.array([q50(ts(lib, s)) for s in range(101, 121)]); qn = np.nanmean([q50(ts(lib.random_partition(9001 + s % 10), s)) for s in range(101, 111)])
    out = dict(thr=thr, N=n, positives=P, clusters=int(len(cnt)), singleton_frac=float(st['singleton_frac']), size_max=int(st['size_max']), label_info=float(st['excess_mi_frac_of_H']), arms50=int(st['arms_for_50pct_pos']), arms50_rand=float(st['arms_for_50pct_pos_randpart']),
               nbr_pos_given_pos=float(st['loo_rate_given_pos']), nbr_pos_given_neg=float(st['loo_rate_given_neg']), q50_random=float(qr), q50_ts=float(np.nanmean(qt)), q50_ts_sd=float(np.nanstd(qt)), fold=float(qr / np.nanmean(qt)), fold_null=float(qr / qn), ceiling=float((n / 2) / (0.5 * P)))
    (a.out / f'cluster_{thr}{"_sub" if a.n_sub else ""}.json').write_text(json.dumps(out, indent=1)); print(json.dumps(out))
