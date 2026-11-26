"""Generates ../figures/2-sampling.png: why the discovery curve of Thompson sampling (TS) plateaus and why some known binders are found later than others.

  A  fraction of the 3,393 binders found by TS as a function of the number of peptides queried (mean of 100 runs, up to 70,000 queries), separately for the binders of clusters with a binder fraction
     (binders / peptides of the cluster) < 0.05, 0.05-0.1, 0.1-0.2, 0.2-0.5 and >= 0.5; dashed = all binders. Binders of clusters that are rich in binders are found early; the binders that are still missing
     at 70,000 queries sit in clusters that are mostly non-binders (their Beta posteriors stay low and the clusters are rarely chosen)
  B  CD-HIT 0.5 clusters of the five known ET binders (BRG1, INO80B, CHD4, NSD3, BICRA): number of peptides (grey) and of binders (purple) in the cluster

Input  (../data): recovery_by_purity.npz, unsampled_by_purity.csv, known_epitope_clusters.csv    (produced by ../work/unsampled_by_purity.py; TS runs as in run_policies.sh on the evaluation seeds 101-200)
Usage: python 2-sampling.py     (conda environment 'thompson': numpy, pandas, matplotlib; run from any directory)
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                                        # results/1-policies
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
LGREY, GRID = "#b5b4ae", "#e6e5e1"
LAB_FS, TICK_FS, LEG_FS, SPINE_W = 30, 24, 20, 2.5                                                      # label, tick and legend font sizes; spine width
BIN_COL = ["#3d1d63", "#2f6db5", "#1faa8f", "#8fc63f", "#f0c419"]                                         # bins of the cluster binder fraction, low -> high
BIN_LAB = ["< 0.05", "0.05–0.1", "0.1–0.2", "0.2–0.5", "≥ 0.5"]
BINDER_COL = "#7b5ea7"                                                                                    # binders in B

z = np.load(DATA / "recovery_by_purity.npz")
grid, curves, n_bin = z["grid"] / 1000, z["curves"], z["n_binders_in_bin"]
known = pd.read_csv(DATA / "known_epitope_clusters.csv")
summ = pd.read_csv(DATA / "unsampled_by_purity.csv")
print(summ.round(3).to_string(index=False))
print(known.round(3).to_string(index=False))

fig = plt.figure(figsize=(25, 8.5))
gs = GridSpec(1, 2, figure=fig, width_ratios=[1.6, 1], wspace=0.25)
axA = fig.add_subplot(gs[0, 0]); axB = fig.add_subplot(gs[0, 1])

# A: recovery of the binders of every cluster-purity bin
for j in range(5):
    axA.plot(grid, curves[j].mean(0), color=BIN_COL[j], lw=4, label=BIN_LAB[j])
axA.plot(grid, curves[5].mean(0), color="black", lw=3, ls="--", label="All binders")
axA.set_xlabel("Peptides queried (×10$^3$)", fontsize=LAB_FS); axA.set_ylabel("Fraction of binders found", fontsize=LAB_FS)
axA.set_xlim(0, 72); axA.set_ylim(0, 1.02); axA.yaxis.grid(True, color=GRID, lw=0.8); axA.set_axisbelow(True)
# one-line legend above A (the first entry is the legend title)
h, l = axA.get_legend_handles_labels()
h = [Line2D([], [], alpha=0)] + h; l = ["Cluster binder fraction:"] + l
axA.legend(h, l, fontsize=LEG_FS - 3, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=len(h), handlelength=1.2, handletextpad=0.4, columnspacing=0.8)

# B: composition of the clusters of the known binders
x = np.arange(len(known))
axB.bar(x, known["size"], width=0.78, color=LGREY, edgecolor="black", linewidth=2, label="Peptides")
axB.bar(x, known["binders"], width=0.52, color=BINDER_COL, edgecolor="black", linewidth=2, label="Binders")
for xi, (n, nb) in enumerate(zip(known["size"], known["binders"])):
    axB.text(xi, n + 0.5, f"{n}", ha="center", va="bottom", fontsize=LEG_FS)
    axB.text(xi, nb + 0.5, f"{nb}", ha="center", va="bottom", fontsize=LEG_FS)
axB.set_xticks(x); axB.set_xticklabels(known["name"]); axB.set_ylabel("Count", fontsize=LAB_FS)
axB.set_ylim(0, 32); axB.yaxis.grid(True, color=GRID, lw=0.8); axB.set_axisbelow(True)
axB.legend(fontsize=LEG_FS, frameon=True, loc="upper right")

for a in (axA, axB):
    a.tick_params(axis="both", which="major", labelsize=TICK_FS, width=2, length=6)
    for sp in a.spines.values():
        sp.set_linewidth(SPINE_W)

# panel letters (same height, in figure coordinates)
inv = fig.transFigure.inverted()
xA, yA = inv.transform(axA.transAxes.transform((-0.11, 1.15))); xB = inv.transform(axB.transAxes.transform((-0.17, 1.15)))[0]
fig.text(xA, yA, "A", fontsize=34, fontweight="bold"); fig.text(xB, yA, "B", fontsize=34, fontweight="bold")

out = FIG / "2-sampling.png"
fig.savefig(out, dpi=130, bbox_inches="tight")
print("saved", out)
