"""Generates ../figures/3-threshold.png: sensitivity of the results to the definition of a binder (>= 4 of 5 AF-CBA models with pLDDT >= p and distance < d; paper: p = 70, d = 20 A).

  A  how close to the boundary are the binders? Histogram of the limiting distance of the 3,393 peptides that pass at 20 A (4th smallest distance among the models with pLDDT >= 70); dotted lines = the cutoffs tested in B and C
  B  number of binders in the library for distance cutoffs of 8-20 A (pLDDT >= 70) and the change relative to the paper definition (3,393 binders)
  C  queries needed to recover 50% of those binders (mean of the 100 runs) for uniform random, Thompson sampling, UCB and epsilon-greedy
  D  number of binders for pLDDT cutoffs of 60-80 (distance < 20 A)
  E  queries to recover 50% of those binders

Input  (../data): limiting_distance.csv, threshold_counts.csv (../work/make_threshold_libraries.py);
        curves_thresholds/{d8,d10,d12,d15,p60,p80}/{random,ts,ucb,epsgreedy}.npz (../work/run_thresholds.sh; cumulative binders after every batch of 50 queries, evaluation seeds 101-200);
        curves_sequence/{random,ts,ucb,epsgreedy}.npz for the paper definition (d20 = p70; ../work/run_policies.sh).
Settings: Beta(0.01, 0.99) prior, k = 50 clusters per round (CD-HIT 0.5), equal allocation, UCB c = 2.0, epsilon-greedy eps = 0.02, selected on the paper definition and not re-tuned for the other cutoffs.
Usage: python 3-threshold.py     (conda environment 'thompson': numpy, pandas, matplotlib; run from any directory)
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                                        # results/1-policies
DATA, FIG = ROOT / "data", ROOT / "figures"
sys.path.insert(0, str(ROOT.parent / "scripts"))          # results/scripts/tsbench
from tsbench import metrics                               # noqa: E402

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
BLUE, ORANGE, GREEN, LGREY, MUTED, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#b5b4ae", "#52514e", "#e6e5e1"   # TS, UCB, epsilon-greedy, uniform, histogram/bars, grid lines
LAB_FS, TICK_FS, LEG_FS, SPINE_W = 30, 24, 20, 2.5                                                          # label, tick and legend font sizes; spine width

POLICIES = [("random", "Uniform random", LGREY, ":"), ("ts", "Thompson sampling", BLUE, "-"), ("ucb", "UCB", ORANGE, "-"), ("epsgreedy", "ε-greedy", GREEN, "-.")]
counts = pd.read_csv(DATA / "threshold_counts.csv").set_index("name")


def q50(name, key):
    """mean queries to recover 50% of the binders of one relabelled library (name d20 = paper definition)"""
    f = DATA / "curves_sequence" / f"{key}.npz" if name == "d20" else DATA / "curves_thresholds" / name / f"{key}.npz"
    z = np.load(f)
    q = metrics.queries_to_fraction(z["grid"], z["curves"], int(counts.loc[name, "n_binders"]), 0.5)
    return np.nanmean(q)


def bars_and_lines(a_bar, a_line, names, xticks, xlabel):
    """bars: binders in the library; lines: queries to 50% recovery of every policy"""
    pos = np.arange(len(names)); base = counts.loc["d20", "n_binders"]
    nb = counts.loc[names, "n_binders"].values
    a_bar.bar(pos, nb, width=0.62, color=MUTED, edgecolor="black", linewidth=2)
    for i, v in enumerate(nb):
        txt = f"{v}\n(-{100 * (1 - v / base):.0f}%)" if v < base else f"{v}\n(+{100 * (v / base - 1):.0f}%)" if v > base else f"{v}"
        a_bar.text(i, v + base * 0.02, txt, ha="center", va="bottom", fontsize=LEG_FS - 2)
    a_bar.set_ylim(0, base * 1.5); a_bar.set_ylabel("Binders in the library", fontsize=LAB_FS)
    for key, lab, col, ls in POLICIES:
        y = np.array([q50(n, key) for n in names]) / 1000
        print(f"{xlabel[:12]:12s} {key:10s}", " ".join(f"{v:5.1f}" for v in y))
        a_line.plot(pos, y, color=col, ls=ls, lw=3, marker="o", ms=14, mec="black", mew=1.6, label=lab)
    a_line.set_ylabel("Queries to 50% recovery\n(×10$^3$)", fontsize=LAB_FS); a_line.yaxis.grid(True, color=GRID, lw=0.8); a_line.set_axisbelow(True)
    for a in (a_bar, a_line):
        a.set_xticks(pos); a.set_xticklabels(xticks); a.set_xlabel(xlabel, fontsize=LAB_FS)
        a.set_xlim(-0.6, len(names) - 0.4)
    a_bar.yaxis.grid(True, color=GRID, lw=0.8); a_bar.set_axisbelow(True)


# ---- figure
fig = plt.figure(figsize=(20, 21))
gs = GridSpec(3, 2, figure=fig, height_ratios=[0.85, 1, 1], hspace=0.5, wspace=0.28)
axA = fig.add_subplot(gs[0, :]); axB = fig.add_subplot(gs[1, 0]); axC = fig.add_subplot(gs[1, 1]); axD = fig.add_subplot(gs[2, 0]); axE = fig.add_subplot(gs[2, 1])

# A: limiting distance of the binders
lim = pd.read_csv(DATA / "limiting_distance.csv")["limit"].values
axA.hist(lim, bins=np.arange(0, 20.5, 0.5), color=MUTED, edgecolor="black", linewidth=1.2)
axA.set_ylim(0, 500)                                                                            # room above the bars for the labels of the cutoffs
for c in (8, 10, 12, 15):
    axA.axvline(c, color="black", lw=2.2, ls=":")
    axA.text(c - 0.12, 495, f"{c} Å", rotation=90, va="top", ha="right", fontsize=LEG_FS)
axA.set_xlim(0, 20.5)
axA.set_xlabel("Limiting distance of the peptide (4th closest passing model, Å)", fontsize=LAB_FS); axA.set_ylabel("Number of binders", fontsize=LAB_FS)

# B, C: distance cutoff at pLDDT >= 70; D, E: pLDDT cutoff at distance < 20 A
bars_and_lines(axB, axC, ["d8", "d10", "d12", "d15", "d20"], ["8", "10", "12", "15", "20"], "Distance cutoff (Å), pLDDT ≥ 70")
bars_and_lines(axD, axE, ["p60", "d20", "p80"], ["60", "70", "80"], "pLDDT cutoff, distance < 20 Å")
axC.legend(fontsize=LEG_FS, frameon=True, loc="center right")
for a in (axC, axE):
    a.set_ylim(0, 80)

for a in (axA, axB, axC, axD, axE):
    a.tick_params(axis="both", which="major", labelsize=TICK_FS, width=2, length=6)
    for sp in a.spines.values():
        sp.set_linewidth(SPINE_W)

# panel letters (A, B and D at the same x in figure coordinates)
inv = fig.transFigure.inverted()
xL = inv.transform(axB.transAxes.transform((-0.2, 1.0)))[0]
for a, letter in ((axA, "A"), (axB, "B"), (axD, "D")):
    fig.text(xL, inv.transform(a.transAxes.transform((0, 1.04)))[1], letter, fontsize=34, fontweight="bold")
axC.text(-0.2, 1.04, "C", transform=axC.transAxes, fontsize=34, fontweight="bold")
axE.text(-0.2, 1.04, "E", transform=axE.transAxes, fontsize=34, fontweight="bold")

out = FIG / "3-threshold.png"
fig.savefig(out, dpi=130, bbox_inches="tight")
print("saved", out)
