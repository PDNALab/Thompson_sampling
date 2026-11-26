"""Build data/library_cdhit_0.5.csv (id, cluster, label) for the BRD3 peptide library from the raw files of the repository:

  clusters/cd-0.5.csv   name, sequence, cluster   CD-HIT clustering of the 142,338 unique 25-mers at identity 0.5 (8,004 clusters)
  data/dict.csv         name, plddt_i, dist_i, rog_i (i = 0..4)   AF-CBA results: five AlphaFold2 models per peptide

A peptide is a binder (label 1) if at least 4 of its 5 models have a mean pLDDT >= 70 and a mean distance to the key ET residues < 20 A (Methods, 'Peptide Library and AF-CBA Classifications').
This reproduces the 3,393 binders of the paper. Usage:  python make_library.py [output.csv]      (default ../data/library_cdhit_0.5.csv)
"""
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]                                   # results/1-policies/work -> repository root
out = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parent / "data" / "library_cdhit_0.5.csv"

cl = pd.read_csv(REPO / "clusters" / "cd-0.5.csv")
dic = pd.read_csv(REPO / "data" / "dict.csv").set_index("name").loc[cl["name"]]
n_pass = sum(((dic[f"plddt_{i}"].values >= 70) & (dic[f"dist_{i}"].values < 20)).astype(int) for i in range(5))
lib = pd.DataFrame({"id": cl["name"].values, "cluster": cl["cluster"].values, "label": (n_pass >= 4).astype(int)})
out.parent.mkdir(parents=True, exist_ok=True)
lib.to_csv(out, index=False)
print(f"{out}: {len(lib)} peptides, {lib.cluster.nunique()} clusters, {int(lib.label.sum())} binders")
