"""Generates ../figures/0-cluster_info.png: cluster information in the BRD3 library (142,338 peptides, 3,393 binders).

  A  fraction of clusters in each size bin for CD-HIT, MMseqs2 easy-cluster (LINCLUST/MMseqs2) and easy-linclust (LINCLUST) at identity 0.4, 0.5, 0.7 and 0.9 (four sub-panels; each method's bars sum to one)
  B  fraction of the binders contained in the clusters with the most binders (clusters ranked by their number of binders; identity 0.5; linear axis cut at 15% of the clusters, all binders are covered by ~12%)
  C  probability P (%) that a peptide whose window is shifted by 1-3 residues from a binder is also a binder (dotted line: library binder rate); uses the window overlap only, no clustering

Input (../data, produced by ../work/make_cluster_data.py): cluster_sizes.csv (tool, identity, cluster, size, binders), shift_neighbours.csv.
Usage: python 0-cluster_info.py     (conda environment 'thompson': numpy, pandas, matplotlib; run from any directory)
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

ROOT = Path(__file__).resolve().parent.parent              # results/0-cluster
DATA, FIG = ROOT / "data", ROOT / "figures"
FIG.mkdir(exist_ok=True)

# Plot style: the SciencePlots styles ['science', 'no-latex'], set explicitly so that scienceplots is not needed
plt.rcParams.update({
    "figure.figsize": (3.5, 2.625),
    "xtick.direction": "in", "xtick.major.size": 3, "xtick.major.width": 0.5, "xtick.minor.size": 1.5, "xtick.minor.width": 0.5, "xtick.minor.visible": True, "xtick.top": True,
    "ytick.direction": "in", "ytick.major.size": 3, "ytick.major.width": 0.5, "ytick.minor.size": 1.5, "ytick.minor.width": 0.5, "ytick.minor.visible": True, "ytick.right": True,
    "axes.linewidth": 0.5, "grid.linewidth": 0.5, "lines.linewidth": 1.0, "legend.frameon": False,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.05,
    "font.family": "serif", "font.serif": ["STIXGeneral"], "mathtext.fontset": "cm", "axes.formatter.use_mathtext": True, "text.usetex": False,
    "axes.grid": False, "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
})
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
LAB_FS, TICK_FS, LEG_FS, SPINE_W = 30, 24, 20, 2.5          # label, tick and legend font sizes; spine width


def ygrid(ax):
    ax.grid(axis="y", color=GRID, lw=0.8)


sizes = pd.read_csv(DATA / "cluster_sizes.csv")
shift = pd.read_csv(DATA / "shift_neighbours.csv")
one = sizes[(sizes.tool == "cd") & (sizes.identity == 0.5)]
base = one.binders.sum() / one["size"].sum()                  # library binder rate (3,393 / 142,338)

CL = [("cd", "CD-HIT", dict(color=INK, ls="-")), ("easy-cluster", "LINCLUST/MMseqs2", dict(color="#6b6a65", ls="--")), ("easy-linclust", "LINCLUST", dict(color="#b5b4ae", ls=":"))]
THR = [0.4, 0.5, 0.7, 0.9]
EDGES = [1, 2, 3, 5, 9, 17, 33, 65, 10 ** 9]
LABELS = ["1", "2", "3-4", "5-8", "9-16", "17-32", "33-64", ">64"]

W, H = 25, 14.8
fy = lambda y: 1 - y / H                                                  # distance from the top (inches) -> figure fraction
fig = plt.figure(figsize=(W, H))
gsA = GridSpec(1, 4, figure=fig, left=0.06, right=0.99, top=fy(0.9), bottom=fy(5.6), wspace=0)       # the four panels of A touch each other
gsBC = GridSpec(1, 2, figure=fig, left=0.06, right=0.99, top=fy(8.0), bottom=fy(13.2), wspace=0.2)    # B and C close below A
axA = [fig.add_subplot(gsA[0, i]) for i in range(4)]; axB = fig.add_subplot(gsBC[0, 0]); axC = fig.add_subplot(gsBC[0, 1])

# ---- A: fraction of clusters per size bin
for i, thr in enumerate(THR):
    a = axA[i]; w = 0.27
    for k, (tool, nice, st) in enumerate(CL):
        sz = sizes[(sizes.tool == tool) & (sizes.identity == thr)]["size"].values
        frac = np.histogram(sz, bins=EDGES)[0] / len(sz)
        a.bar(np.arange(len(LABELS)) + (k - 1) * w, frac, w, color=st["color"], edgecolor="white", lw=0.6, label=nice if i == 0 else None)
    a.set_xticks(range(len(LABELS))); a.set_xticklabels(LABELS, rotation=45, ha="right"); a.set_ylim(0, 0.8); ygrid(a); a.set_axisbelow(True)
    a.set_title(f"{thr}", fontsize=LAB_FS)
    if i == 0:
        a.set_ylabel("Fraction of clusters", fontsize=LAB_FS); a.legend(frameon=True, fontsize=LEG_FS, loc="upper right")
    else:
        a.tick_params(labelleft=False)

# ---- B: binders covered by the clusters with the most binders (identity 0.5)
for tool, nice, st in CL:
    b = np.sort(sizes[(sizes.tool == tool) & (sizes.identity == 0.5)]["binders"].values)[::-1]
    cum = np.cumsum(b) / b.sum()
    axB.plot((np.arange(len(cum)) + 1) / len(cum) * 100, cum * 100, color=st["color"], ls=st["ls"], lw=3.2, label=nice)
axB.set_xlim(0, 15)                                                                  # linear axis: all binders are covered by ~12% of the clusters
axB.set_xlabel("Clusters, ranked by number of binders (% of clusters)", fontsize=LAB_FS); axB.set_ylabel("Binders covered (%)", fontsize=LAB_FS); ygrid(axB); axB.legend(frameon=True, fontsize=LEG_FS, loc="lower right")

# ---- C: shifted windows
axC.bar(shift["shift"], shift["p_partner_binder_given_binder"] * 100, color=MUTED, width=0.55)
axC.axhline(base * 100, color=INK, ls=":", lw=3, label=f"library binder rate ({base * 100:.1f}%)"); axC.legend(frameon=True, fontsize=LEG_FS, loc="upper right")
for _, r in shift.iterrows():
    axC.text(r["shift"], r["p_partner_binder_given_binder"] * 100 + 1.8, f"{r['p_partner_binder_given_binder'] * 100:.0f}%", ha="center", fontsize=LEG_FS)
axC.set_xticks([1, 2, 3]); axC.set_xlabel("Shift between the two peptides", fontsize=LAB_FS); axC.set_ylabel("P (%)", fontsize=LAB_FS); axC.set_ylim(0, 100); ygrid(axC); axC.set_axisbelow(True)     # grid lines behind the bars

for a in axA + [axB, axC]:
    a.tick_params(axis="both", which="major", labelsize=TICK_FS, width=2, length=6)
    for sp in a.spines.values():
        sp.set_linewidth(SPINE_W)
xm = (axA[0].get_position().x0 + axA[-1].get_position().x1) / 2                      # one "Cluster size" label centred under the four panels
fig.text(xm, fy(5.6) - 0.072, "Cluster size", fontsize=LAB_FS, ha="center", va="top")
axA[0].text(-0.30, 1.14, "A", transform=axA[0].transAxes, fontsize=38, fontweight="bold")
axB.text(-0.10, 1.06, "B", transform=axB.transAxes, fontsize=38, fontweight="bold")
axC.text(-0.10, 1.06, "C", transform=axC.transAxes, fontsize=38, fontweight="bold")

out = FIG / "0-cluster_info.png"
fig.savefig(out, dpi=100, bbox_inches="tight")
print("saved", out)
