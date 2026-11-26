"""Generates ../figures/1-purity.png: how pure and how informative are the clusters? (BRD3 library, 142,338 peptides, 3,393 binders; sequence identity 0.5; CD-HIT, MMseqs2 easy-cluster, easy-linclust)

  A  purity: share of the binders found in clusters of each purity (binder fraction of the cluster; clusters without binders, 89-92% of all clusters, are not shown)
  B  highly similar peptides with discordant labels: pairs of the same CD-HIT cluster that differ only by substitutions (same window, 22-24 of 25 residues identical), a binder and a non-binder with
     at most 2 of the 5 AF2 models passing; differing residues in red (one example per cluster, the most similar first)

Input (../data, produced by ../work/make_purity_data.py): purity_peptides.csv, purity_clusters.csv, purity_summary.csv, purity_pairs.npz, purity_random_pairs.npz.
(The within-cluster enrichment, the entropy of the cluster labels and the identity vs binder probability curve are not drawn; their numbers are still printed.)
Usage: python 1-purity.py     (conda environment 'thompson': numpy, pandas, matplotlib; run from any directory; prints the numbers behind the figure)
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent              # results/0-cluster
DATA, FIG = ROOT / "data", ROOT / "figures"
FIG.mkdir(exist_ok=True)

plt.rcParams.update({
    "figure.figsize": (3.5, 2.625),
    "xtick.direction": "in", "xtick.major.size": 3, "xtick.major.width": 0.5, "xtick.minor.size": 1.5, "xtick.minor.width": 0.5, "xtick.minor.visible": False, "xtick.top": False,
    "ytick.direction": "in", "ytick.major.size": 3, "ytick.major.width": 0.5, "ytick.minor.size": 1.5, "ytick.minor.width": 0.5, "ytick.minor.visible": True, "ytick.right": True,
    "axes.linewidth": 0.5, "grid.linewidth": 0.5, "lines.linewidth": 1.0, "legend.frameon": False,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.05,
    "font.family": "serif", "font.serif": ["STIXGeneral"], "mathtext.fontset": "cm", "axes.formatter.use_mathtext": True, "text.usetex": False,
    "axes.grid": False, "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
})
INK, MID, LIGHT, GRID, RED = "#0b0b0b", "#6b6a65", "#b5b4ae", "#e6e5e1", "#d62728"
LAB_FS, TICK_FS, LEG_FS, SPINE_W = 30, 24, 20, 2.5
TOOLS = [("CD-HIT", INK), ("MMseqs2", MID), ("LINCLUST", LIGHT)]

pep = pd.read_csv(DATA / "purity_peptides.csv")
cl = pd.read_csv(DATA / "purity_clusters.csv")
summ = pd.read_csv(DATA / "purity_summary.csv").set_index("tool")
zp, zr = np.load(DATA / "purity_pairs.npz"), np.load(DATA / "purity_random_pairs.npz")
y, npass, seq = pep["label"].values, pep["n_pass"].values, pep["sequence"].values
base = y.mean()
H = summ["H_label_bits"].iloc[0]

W, HH = 22.9, 9.7
fig = plt.figure(figsize=(W, HH))


def ax_in(x0, y0, w, h):
    return fig.add_axes([x0 / W, y0 / HH, w / W, h / HH])


def style(a):
    a.tick_params(axis="both", which="major", labelsize=TICK_FS, width=2, length=6)
    a.yaxis.grid(True, color=GRID, lw=0.8); a.set_axisbelow(True)
    for sp in a.spines.values():
        sp.set_linewidth(SPINE_W)


def grouped(a, groups, values, w=0.26, **kw):
    """values[tool] = list over groups; bars side by side, one colour per tool"""
    x = np.arange(len(groups))
    for k, (t, col) in enumerate(TOOLS):
        a.bar(x + (k - 1) * w, values[t], w * 0.92, color=col, edgecolor=INK, linewidth=1.5, label=t, zorder=3, **kw)
    a.set_xticks(x); a.set_xticklabels(groups)
    return x, w


Y1, H1 = 1.9, 6.6
XE = 10.9                                                                  # left edge of the examples panel
axA, axE = ax_in(2.9, Y1, 6.6, H1), ax_in(XE, Y1, 11.0, H1)                # A is square

# ---- A: purity
EDG = [0, 0.05, 0.1, 0.2, 0.5, 1.0001]
LAB = ["<5%", "5–10%", "10–20%", "20–50%", "≥50%"]
share = {}
for t, _ in TOOLS:
    c = cl[cl.tool == t]; pur = (c.binders / c["size"]).values; has = c.binders.values > 0
    share[t] = [100 * c.binders.values[(pur >= lo) & (pur < hi) & has].sum() / y.sum() for lo, hi in zip(EDG[:-1], EDG[1:])]
    print(f"{t}: {100 * (~has).mean():.1f}% of the {len(c):,} clusters without binder; share of binders by purity bin {np.round(share[t], 1)}; median purity of the clusters with a binder {np.median(pur[has]):.3f}; clusters >= 50% pure {(pur >= 0.5).sum()}")
grouped(axA, LAB, share)
axA.set_xlabel("Binder fraction of the cluster", fontsize=LAB_FS); axA.set_ylabel("Binders (%)", fontsize=LAB_FS)
axA.set_ylim(0, 48); axA.set_yticks(range(0, 41, 10)); axA.legend(fontsize=LEG_FS - 3, frameon=True, loc="upper left", handlelength=1.2, borderpad=0.4)
axA.set_xticklabels(LAB, fontsize=TICK_FS - 8)
style(axA)

# ---- numbers of the enrichment and of the entropy (not drawn)
for t, _ in TOOLS:
    print(f"{t}: another member of the cluster is a binder: {100 * summ.loc[t, 'loo_p_given_binder']:.1f}% for a binder, {100 * summ.loc[t, 'loo_p_given_nonbinder']:.2f}% for a non-binder (library {100 * base:.2f}%)")
    print(f"{t}: mutual information {100 * summ.loc[t, 'mi_bits'] / H:.1f}% of the label entropy, permuted labels {100 * summ.loc[t, 'mi_perm_bits'] / H:.1f}%, excess {100 * summ.loc[t, 'mi_excess_bits'] / H:.1f}%")

# ---- numbers of the identity vs binder-probability curve (not drawn)
BINS = [0, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.96, 1.0001]
BL = ["<0.4", "0.4–0.5", "0.5–0.6", "0.6–0.7", "0.7–0.8", "0.8–0.9", "0.9–0.96", "≥0.96"]


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


xs = np.arange(len(BL))
rp, ri = y[zr["partner"]], zr["identity"]
for k, (t, col) in enumerate(TOOLS):
    p, ide = zp[f"{t}__partner"], zp[f"{t}__identity"]
    P, LO, HI, N = [], [], [], []
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        m = (ide >= lo) & (ide < hi); n_ = m.sum(); kk = y[p[m]].sum()
        P.append(100 * kk / n_); a_, b_ = wilson(kk, n_); LO.append(100 * a_); HI.append(100 * b_); N.append(n_)
    P = np.array(P)
    print(f"{t}: P(partner is a binder) by identity bin {np.round(P, 1)}; n {N}")
for j, (lo, hi) in enumerate(zip(BINS[:-1], BINS[1:])):
    m = (ri >= lo) & (ri < hi)
    print("random pairs", BL[j], int(m.sum()), f"{100 * rp[m].mean():.1f}%" if m.sum() else "-")

# ---- B: discordant examples
f_, p_, i_, s_ = (zp[f"CD-HIT__{k}"] for k in ("focal", "partner", "identity", "shift"))
g_ = pep["cluster_cd"].values
sub = (s_ == 0) & (i_ >= 0.88)
disc = sub & (y[p_] == 0)
print(f"CD-HIT: {sub.sum()} binder pairs of the same cluster differing only by substitutions (22–24 of 25 identical); {disc.sum()} with a non-binder partner "
      f"(partner models passing 0/1/2/3: {np.bincount(npass[p_[disc]], minlength=5)[:4].tolist()}); partner is a binder in {100 * (1 - disc.sum() / sub.sum()):.0f}% of the pairs")
sel = np.flatnonzero(disc & (npass[p_] <= 2))
sel = sel[np.lexsort((-npass[f_[sel]], -i_[sel]))]
ex, seen = [], set()
for j in sel:
    if g_[f_[j]] in seen:
        continue
    seen.add(g_[f_[j]]); ex.append(j)
    if len(ex) == 6:
        break
axE.set_xlim(0, 1); axE.set_ylim(0, 1); axE.axis("off")
axE.text(0.0, 1.0, "Same CD-HIT cluster, 1–3 substitutions,\ndifferent label (AF2 models passing, of 5)", fontsize=LEG_FS + 2, va="top", ha="left")
row = 0.76 / (len(ex) - 1)
for r, j in enumerate(ex):
    a, b = seq[f_[j]], seq[p_[j]]
    ytop = 0.84 - r * row
    for line, (s, lab) in enumerate(((a, f"binder {npass[f_[j]]}/5"), (b, f"non-binder {npass[p_[j]]}/5"))):
        yy = ytop - line * 0.058
        axE.text(0.0, yy, lab, fontsize=LEG_FS, va="center", ha="left", family="serif")
        for q, ch in enumerate(s):
            axE.text(0.30 + q * 0.0285, yy, ch, fontsize=LEG_FS + 1, va="center", ha="center", family="monospace", color=RED if a[q] != b[q] else INK, fontweight="bold" if a[q] != b[q] else "normal")
    print(f"example {r + 1}: identity {i_[j]:.2f}  {a} ({npass[f_[j]]}/5)  {b} ({npass[p_[j]]}/5)  cluster {g_[f_[j]]}")

FW, FH = fig.get_figwidth(), fig.get_figheight()
for x0, y0, L in ((1.3, Y1 + H1, "A"), (XE, Y1 + H1, "B")):
    fig.text(x0 / FW, (y0 + 0.45) / FH, L, fontsize=34, fontweight="bold")

out = FIG / "1-purity.png"
fig.savefig(out, dpi=130, bbox_inches="tight")
print("saved", out)
