"""Thompson sampling arm (random pick inside the cluster) and uniform random sampling on the SARS-CoV-2 peptide-array library, 100 evaluation seeds (101-200): the reference curves of figures/2-covid.png.

Library: 165,946 12-mers, 3,335 positives (2.0 %; reactive in >= 10 of 100 COVID IgG samples and in none of 10 healthy controls), CD-HIT 0.5 clusters (5,877; data/covid/library.csv, cdhit0.5.npy).
Settings of the arm: those of 1-policies/figures/1-baselines.png (Beta(0.01, 0.99) prior, k = 50 clusters per round, equal allocation, batch 50), seed set 3.5 % of the library (counted as queries), run to 50 % of the library.
Output data/covid/arms.npz: grid (queried peptides), ts and random (seeds x grid: cumulative positives), seeds.   (The same simulations as work/analyze_main.py, which re-simulated them at every run.)
Usage: nice -n 10 python covid_arms.py     (conda environment 'thompson'; never overwrites an existing output)"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                                        # results/4-additional
sys.path.insert(0, str(ROOT.parent / "scripts"))          # results/scripts/tsbench
from tsbench import Library, make_grid, make_policy, run, run_random  # noqa: E402
from tsbench.sim import to_grid                           # noqa: E402

D = ROOT / "data" / "covid"
OUT = D / "arms.npz"
if OUT.exists():
    sys.exit(f"{OUT} exists; not overwritten")
SEEDS, B, A0, B0 = list(range(101, 201)), 50, 0.01, 0.99
y = pd.read_csv(D / "library.csv", usecols=["label"]).label.values.astype(np.int8)
n = len(y); S = int(round(0.035 * n)); Q50 = int(round(0.5 * n))
lib = Library(y, np.load(D / "cdhit0.5.npy"))
grid = make_grid(S, n, B); grid = grid[grid <= Q50]
ts, rnd = [], []
for s in SEEDS:
    o = run(lib, make_policy("ts"), s, S=S, B=B, k=50, alloc="equal", alpha0=A0, beta0=B0, max_queries=Q50 + B)
    ts.append(to_grid(o["queried"], o["cum"], grid))
    o = run_random(lib, s, S=S, B=B)
    rnd.append(to_grid(o["queried"], o["cum"], grid))
    if s % 20 == 0:
        print("seed", s, flush=True)
print(f"N {n} positives {int(y.sum())} clusters {lib.n_groups} seed set {S} queries to {grid[-1]}:  TS {np.mean(ts, 0)[-1]:.1f}  random {np.mean(rnd, 0)[-1]:.1f} positives")
np.savez_compressed(OUT, grid=grid, ts=np.array(ts), random=np.array(rnd), seeds=np.array(SEEDS))
print("saved", OUT)
