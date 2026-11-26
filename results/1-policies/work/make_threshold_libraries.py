"""Libraries of the binder-definition sensitivity analysis (3-threshold.png): the BRD3 library (CD-HIT 0.5 clusters) relabelled at other cutoffs.

A peptide is a binder if >= 4 of its 5 AF-CBA models have mean pLDDT >= p and distance to the key ET residues < d (data/dict.csv of the repository; paper: p = 70, d = 20 A).
Variants: distance cutoff d = 8, 10, 12, 15 A at pLDDT 70 (names d8 ... d15) and pLDDT cutoff p = 60, 80 at 20 A (p60, p80); the paper definition (d20, p70) is data/library_cdhit_0.5.csv.

Output  <out_dir>/<name>.csv            id, cluster, label      (default /blue/alberto.perezant/jokent.gaza/ts_threshold_inputs; used by run_thresholds.sh)
        ../data/threshold_counts.csv      name, kind, dist, plddt, n_binders, known_kept (how many of the 5 known epitopes of data/watchlist.csv remain binders)
        ../data/limiting_distance.csv     id, limit               limiting distance of every peptide passing at 20 A = 4th smallest distance among its models with pLDDT >= 70 (histogram of panel A)
Usage: python make_threshold_libraries.py [out_dir]      (conda environment 'thompson')
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/blue/alberto.perezant/jokent.gaza/ts_threshold_inputs")
out.mkdir(parents=True, exist_ok=True)
cl = pd.read_csv(REPO / "clusters" / "cd-0.5.csv")
dic = pd.read_csv(REPO / "data" / "dict.csv").set_index("name").loc[cl["name"]]
P = dic[[f"plddt_{i}" for i in range(5)]].values
D = dic[[f"dist_{i}" for i in range(5)]].values
watch = pd.read_csv(HERE.parent / "data" / "watchlist.csv")
widx = pd.Series(np.arange(len(cl)), index=cl["name"].values).loc[watch["id"]].values

VARIANTS = [("d8", "dist", 8, 70), ("d10", "dist", 10, 70), ("d12", "dist", 12, 70), ("d15", "dist", 15, 70), ("d20", "dist", 20, 70), ("p60", "plddt", 20, 60), ("p80", "plddt", 20, 80)]
rows = []
for name, kind, d, p in VARIANTS:
    y = (((P >= p) & (D < d)).sum(1) >= 4).astype(int)
    pd.DataFrame({"id": cl["name"].values, "cluster": cl["cluster"].values, "label": y}).to_csv(out / f"{name}.csv", index=False)
    rows.append(dict(name=name, kind=kind, dist=d, plddt=p, n_binders=int(y.sum()), known_kept=int(y[widx].sum())))
    print(rows[-1])
pd.DataFrame(rows).to_csv(HERE.parent / "data" / "threshold_counts.csv", index=False)

limit = np.sort(np.where(P >= 70, D, np.inf), axis=1)[:, 3]
ok = np.isfinite(limit) & (limit < 20)
pd.DataFrame({"id": cl["name"].values[ok], "limit": limit[ok]}).to_csv(HERE.parent / "data" / "limiting_distance.csv", index=False)
print(f"limiting distance: {ok.sum()} peptides pass at 20 A")
