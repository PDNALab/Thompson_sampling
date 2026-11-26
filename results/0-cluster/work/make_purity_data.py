"""Tables behind the cluster purity figure (plot/1-purity.py): how informative are the identity-0.5 clusters about binding, for the three clustering methods?

Inputs (repository root): clusters/{cd,easy-cluster,easy-linclust}-0.5.csv  name, sequence, cluster   (142,338 unique 25-mers)
                          data/dict.csv                                     name, plddt_i, dist_i (AF-CBA, five models)
A peptide is a binder if at least 4 of its 5 models have a mean pLDDT >= 70 and a mean distance to the key ET residues < 20 A (3,393 binders); n_pass = number of models passing both cutoffs.

Output (../data):
  purity_peptides.csv  name, sequence, label, n_pass, cluster_{cd,mm,lin}   (cluster ids are strings of the three clusterings)
  purity_clusters.csv  tool, cluster, size, binders                          one row per cluster (purity = binders / size)
  purity_summary.csv   tool, n_clusters, singleton_frac, clusters_with_binder, H_label_bits, mi_bits, mi_perm_bits, mi_excess_bits,
                       loo_p_given_binder, loo_p_given_nonbinder, base_rate     (loo = another random member of the same cluster is a binder, leave-one-out; mi_perm = mean of 20 label permutations)
  purity_pairs.npz     for every binder and every other member of its cluster (per tool): focal, partner, identity, shift, partner label, partner n_pass
                       identity = best ungapped identity (matches / overlap over shifts |s| <= 12, overlap >= 13), shift = the shift that gives it
  purity_random_pairs.npz  the same for 150 random partners per binder (reference, independent of the clustering)
Usage: nice -n 10 python make_purity_data.py     (conda environment 'thompson'; a few minutes on one CPU)
"""
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "data"
REPO = HERE.parents[2]
rng = np.random.default_rng(0)

TOOLS = [("cd", "CD-HIT"), ("easy-cluster", "MMseqs2"), ("easy-linclust", "LINCLUST")]
ref = pd.read_csv(REPO / "clusters" / "cd-0.5.csv")
names, seqs = ref["name"].values, ref["sequence"].values
dic = pd.read_csv(REPO / "data" / "dict.csv").set_index("name").loc[names]
n_pass = sum(((dic[f"plddt_{i}"].values >= 70) & (dic[f"dist_{i}"].values < 20)).astype(int) for i in range(5))
y = (n_pass >= 4).astype(int)
n, nb = len(y), int(y.sum())
assert nb == 3393
cl = {t: pd.factorize(pd.read_csv(REPO / "clusters" / f"{t}-0.5.csv").set_index("name").loc[names, "cluster"].values)[0] for t, _ in TOOLS}
pd.DataFrame({"name": names, "sequence": seqs, "label": y, "n_pass": n_pass, **{f"cluster_{k}": pd.read_csv(REPO / "clusters" / f"{t}-0.5.csv").set_index("name").loc[names, "cluster"].values
              for (t, _), k in zip(TOOLS, ("cd", "mm", "lin"))}}).to_csv(OUT / "purity_peptides.csv", index=False)


def entropy(p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def mutual_info(g, lab):
    cnt, b = np.bincount(g), np.bincount(g, weights=lab)
    f = b[cnt > 0] / cnt[cnt > 0]
    return entropy(lab.mean()) - (cnt[cnt > 0] / len(lab) * entropy(f)).sum()


A = np.array([[ord(ch) for ch in s] for s in seqs], dtype=np.uint8)


def best_identity(i, J, max_shift=12, min_overlap=13):
    best, bs = np.zeros(len(J)), np.zeros(len(J), np.int8)
    for s in range(-max_shift, max_shift + 1):
        L = 25 - abs(s)
        if L < min_overlap:
            continue
        x, z = (A[i, s:], A[J, :L]) if s >= 0 else (A[i, :L], A[J, -s:])
        v = (z == x[None, :]).mean(1)
        up = v > best
        best[up], bs[up] = v[up], s
    return best, bs


rows_c, rows_s, pairs = [], [], {}
focal = np.flatnonzero(y == 1)
for t, name in TOOLS:
    g = cl[t]
    size, bind = np.bincount(g), np.bincount(g, weights=y).astype(int)
    rows_c.append(pd.DataFrame({"tool": name, "cluster": np.arange(len(size)), "size": size, "binders": bind}))
    mi = mutual_info(g, y)
    mip = np.mean([mutual_info(g, rng.permutation(y)) for _ in range(20)])
    s_i = size[g]; b_i = bind[g]
    ok = s_i > 1
    loo = (b_i - y) / np.maximum(s_i - 1, 1)                    # fraction of the other members that are binders
    rows_s.append(dict(tool=name, n_clusters=len(size), singleton_frac=(size == 1).mean(), clusters_with_binder=int((bind > 0).sum()), H_label_bits=entropy(y.mean()),
                       mi_bits=mi, mi_perm_bits=mip, mi_excess_bits=mi - mip, loo_p_given_binder=loo[(y == 1) & ok].mean(), loo_p_given_nonbinder=loo[(y == 0) & ok].mean(), base_rate=y.mean()))
    members = {}
    for i, c in enumerate(g):
        members.setdefault(c, []).append(i)
    F, Pn, ID, SH = [], [], [], []
    for i in focal:
        m = np.array(members[g[i]]); m = m[m != i]
        if len(m):
            idn, sh = best_identity(i, m)
            F.append(np.full(len(m), i)); Pn.append(m); ID.append(idn); SH.append(sh)
    pairs[name] = [np.concatenate(v) for v in (F, Pn, ID, SH)]
    print(name, "clusters", len(size), "pairs", len(pairs[name][0]), flush=True)
pd.concat(rows_c).to_csv(OUT / "purity_clusters.csv", index=False)
summ = pd.DataFrame(rows_s)
summ.to_csv(OUT / "purity_summary.csv", index=False)
print(summ.round(4).to_string())
np.savez_compressed(OUT / "purity_pairs.npz", **{f"{k}__{f}": v for k, p in pairs.items() for f, v in zip(("focal", "partner", "identity", "shift"), p)})
F, Pn, ID, SH = [], [], [], []
for i in focal:
    J = rng.integers(0, n, 150); J = J[J != i]
    idn, sh = best_identity(i, J); F.append(np.full(len(J), i)); Pn.append(J); ID.append(idn); SH.append(sh)
np.savez_compressed(OUT / "purity_random_pairs.npz", focal=np.concatenate(F), partner=np.concatenate(Pn), identity=np.concatenate(ID), shift=np.concatenate(SH))
print("done")
