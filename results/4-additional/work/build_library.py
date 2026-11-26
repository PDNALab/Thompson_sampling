"""SARS-CoV-2 / coronavirus peptide microarray (Mishra et al., Commun Biol 2021; Dryad doi:10.5061/dryad.2547d7wnz, CC0): 166,465 12-mer probes (11-aa-overlap tiling
of coronavirus proteomes), plasma IgG signal of 132 samples. Builds the library used for the Thompson-sampling benchmark.
Label (user's choice 2026-11-18): positive = signal > 10,000 (the authors' cutoff) in >= 10 of the 100 COVID-19 IgG samples (Mild_COVID, Severe_COVID, Asymptomatic; two time points per subject, so the
samples are not independent) AND in none of the 10 healthy controls (removes non-specific background).
Cleaning: drop the 500 'random_probe' control probes (not part of the viral tiling), probes that are not 12 residues long (19) or have non-standard residues, and duplicated sequences
(first copy kept; a duplicate with a conflicting label is dropped entirely).  Output data/library.csv (orig_row, id, sequence, label, seq_id, position, n_covid, n_healthy) and data/F470.npy
(the 470 BRD3 features: AAC 20 + DPC 400 + PHYS 10 + NTERM 20 + CTERM 20 = tsbench.features.build + tsbench.features.terminal, rows in library order).
Usage: python build_library.py [library|features]  (default both)"""
import re, sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; BENCH = HERE.parents[1]; sys.path.insert(0, str(BENCH / 'scripts'))
D = HERE.parent / 'data' / 'covid'; AA = set('ACDEFGHIKLMNPQRSTVWY'); K = 10
def library():
    d = pd.read_csv(D / 'raw' / 'IgG_aggregate_file_covid_pub.csv', low_memory=False); cols = list(d.columns[17:]); grp = np.array([re.split(r'_\d', c)[0] for c in cols]); R = d[cols].values.astype(float) > 10000
    cov = np.array([g.startswith(('Mild', 'Severe', 'Asymptomatic')) for g in grp]); n_cov = R[:, cov].sum(1); n_hl = R[:, grp == 'Healthy_Controls_China'].sum(1)
    lib = pd.DataFrame({'orig_row': np.arange(len(d)), 'sequence': d.PROBE_SEQUENCE.astype(str).values, 'seq_id': d.SEQ_ID.astype(str).values, 'position': d.POSITION.values, 'n_covid': n_cov, 'n_healthy': n_hl})
    lib['label'] = ((lib.n_covid >= K) & (lib.n_healthy == 0)).astype(int); print(f'probes {len(lib)}; positives before cleaning {int(lib.label.sum())}')
    steps = [('random_probe controls', lib.seq_id.str.startswith('random_probe')), ('length != 12', lib.sequence.str.len() != 12), ('non-standard residues', ~lib.sequence.map(lambda s: set(s) <= AA))]
    for nm, m in steps: print(f'  drop {nm}: {int(m.sum())} (positives {int(lib.label[m].sum())})'); lib = lib[~m]
    g = lib.groupby('sequence').label.agg(['nunique', 'size']); conflict = set(g.index[(g['nunique'] > 1)]); dup = int(lib.sequence.duplicated().sum()); print(f'  duplicated sequences (extra copies): {dup}; with conflicting labels: {len(conflict)}')
    lib = lib[~lib.sequence.isin(conflict)].drop_duplicates('sequence').reset_index(drop=True); lib.insert(1, 'id', [f'P{i:06d}' for i in range(len(lib))]); lib.to_csv(D / 'library.csv', index=False)
    print(f'library: N {len(lib)}, positives {int(lib.label.sum())} ({lib.label.mean():.2%}), composition by SEQ_ID class:'); print(lib.seq_id.str.replace(r'_\d+$', '', regex=True).str.replace(r'\d+$', '', regex=True).value_counts().head(10).to_string())
def features():
    from tsbench.features import build, terminal
    lib = pd.read_csv(D / 'library.csv'); seqs = lib.sequence.values; assert all(len(s) == 12 for s in seqs); _, f2 = build(seqs)
    X = np.hstack([f2, terminal(seqs, 'N'), terminal(seqs, 'C')]).astype(np.float32); assert X.shape == (len(lib), 470) and np.isfinite(X).all(); np.save(D / 'F470.npy', X)
    r = np.ptp(X, 0); print('features', X.shape, 'constant columns', int((r == 0).sum()))
if __name__ == '__main__':
    st = sys.argv[1:] or ['library', 'features']
    if 'library' in st: library()
    if 'features' in st: features()
