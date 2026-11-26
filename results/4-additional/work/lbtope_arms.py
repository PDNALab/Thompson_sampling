"""Thompson sampling arm (random pick inside the cluster) and uniform random sampling on the LBtope Fixed libraries, 100 evaluation seeds (101-200): the reference curves of figures/1-lbtope.png.

Libraries: the unmodified LBtope Fixed library (32,628 peptides, 37 % positives; data/lbtope/fixed.csv, CD-HIT 0.5 clusters fixed__cdhit0.5.npy) and the five sparse versions
(21,652 peptides, 5 % positives; data/lbtope/sparse/ds<i>/library.csv, cdhit0.5.npy; positives randomly held out).
Settings of the arm: those of 1-policies/figures/1-baselines.png (Beta(0.01, 0.99) prior, k = 50 clusters per round, equal allocation, batch 50), seed set 3.5 % of the library (counted as queries), run to 50 % of the library.
Output data/lbtope/arms.npz: for every tag (full, ds1..ds5) <tag>__grid (queried peptides), <tag>__ts and <tag>__random (seeds x grid: cumulative positives), seeds.
(The same simulations as in analyze_all.py, which re-simulated them at every run.)
Usage: nice -n 10 python lbtope_arms.py     (conda environment 'thompson'; a few minutes on one CPU; never overwrites an existing output)"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                                        # results/4-additional
sys.path.insert(0, str(ROOT.parent / "scripts"))          # results/scripts/tsbench
from tsbench import Library, make_grid, make_policy, run, run_random  # noqa: E402
from tsbench.sim import to_grid                           # noqa: E402

OUT = ROOT / "data" / "lbtope" / "arms.npz"
if OUT.exists():
    sys.exit(f"{OUT} exists; not overwritten")
D = ROOT / "data" / "lbtope"
LIBS = {"full": (D / "fixed.csv", D / "fixed__cdhit0.5.npy")}
LIBS.update({f"ds{i}": (D / "sparse" / f"ds{i}" / "library.csv", D / "sparse" / f"ds{i}" / "cdhit0.5.npy") for i in range(1, 6)})
SEEDS, B, A0, B0 = list(range(101, 201)), 50, 0.01, 0.99

out = {}
for tag, (lib_csv, cl_npy) in LIBS.items():
    y = pd.read_csv(lib_csv).label.values.astype(np.int8)
    n = len(y); S = int(round(0.035 * n)); Q50 = int(round(0.5 * n))
    lib = Library(y, np.load(cl_npy))
    grid = make_grid(S, n, B); grid = grid[grid <= Q50]
    ts, rnd = [], []
    for s in SEEDS:
        o = run(lib, make_policy("ts"), s, S=S, B=B, k=50, alloc="equal", alpha0=A0, beta0=B0, max_queries=Q50)
        ts.append(to_grid(o["queried"], o["cum"], grid))
        o = run_random(lib, s, S=S, B=B, max_queries=Q50)
        rnd.append(to_grid(o["queried"], o["cum"], grid))
    out[f"{tag}__grid"], out[f"{tag}__ts"], out[f"{tag}__random"] = grid, np.array(ts), np.array(rnd)
    print(f"{tag:4s} N {n:6d} positives {int(y.sum()):6d} clusters {lib.n_groups:5d} seed set {S:5d} queries to {grid[-1]:5d}:  TS {np.mean(ts, 0)[-1]:8.1f}  random {np.mean(rnd, 0)[-1]:8.1f} positives", flush=True)
np.savez_compressed(OUT, **out, seeds=np.array(SEEDS))
print("saved", OUT)
