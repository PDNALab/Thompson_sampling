"""Build the 12 clustering inputs of the BRD3 library (CD-HIT, MMseqs2 easy-cluster, easy-linclust x identity 0.4, 0.5, 0.7, 0.9) for run_clusterings.sh:

  <out_dir>/<method>_<thr>.csv   id, cluster, label     (method: cdhit, easycluster, easylinclust; labels as in make_library.py: binder = >= 4 of 5 AF-CBA models with pLDDT >= 70 and distance < 20 A)
  ../data/cluster_counts.csv     method, thr, n_clusters (number of clusters of every clustering; used by plot/0-hyperparam.py)

Raw files: clusters/{cd,easy-cluster,easy-linclust}-<thr>.csv of the repository and data/dict.csv.
Usage: python make_cluster_inputs.py [out_dir]      (default /blue/alberto.perezant/jokent.gaza/ts_cluster_inputs; the inputs are 38 MB and are not kept in results/)
"""
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/blue/alberto.perezant/jokent.gaza/ts_cluster_inputs")
out.mkdir(parents=True, exist_ok=True)
FILES = {"cdhit": "cd", "easycluster": "easy-cluster", "easylinclust": "easy-linclust"}
dic = pd.read_csv(REPO / "data" / "dict.csv").set_index("name")
n_pass = sum(((dic[f"plddt_{i}"].values >= 70) & (dic[f"dist_{i}"].values < 20)).astype(int) for i in range(5))
label = pd.Series((n_pass >= 4).astype(int), index=dic.index)
rows = []
for m, stem in FILES.items():
    for thr in ("0.4", "0.5", "0.7", "0.9"):
        cl = pd.read_csv(REPO / "clusters" / f"{stem}-{thr}.csv")
        lib = pd.DataFrame({"id": cl["name"].values, "cluster": cl["cluster"].values, "label": label.loc[cl["name"]].values})
        lib.to_csv(out / f"{m}_{thr}.csv", index=False)
        rows.append(dict(method=m, thr=float(thr), n_clusters=lib.cluster.nunique(), n_peptides=len(lib), n_binders=int(lib.label.sum())))
        print(rows[-1])
pd.DataFrame(rows).to_csv(HERE.parent / "data" / "cluster_counts.csv", index=False)
