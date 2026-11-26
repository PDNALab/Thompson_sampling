"""Generates ../figures/1-baselines.png: selection policies on the BRD3 library.

  A  discovery curves of Thompson sampling (TS), UCB, epsilon-greedy and uniform random sampling (mean of 100 runs; error bars = SD across the runs at every 10,000 peptides; AUC of each method in the legend)
  B  queries needed to recover 50% of the 3,393 binders (mean +- SE over the 100 runs)
  C  queries to recover 50% of the binders on the sequence clusters (filled) and on random partitions with the same cluster-size distribution (open); dotted line = uniform random sampling (same quantity as B)
  D  queries until each of the five known ET-binding epitopes (BRG1, INO80B, CHD4, NSD3, BICRA) is found: median (dot) and interquartile range over the 100 runs

Input  (../data): library_cdhit_0.5.csv; curves_sequence/{random,ts,ucb,epsgreedy,cluster_uniform}.npz; curves_randpart/{ts,ucb,epsgreedy,cluster_uniform}.npz
        (cumulative binders after every batch of 50 queries for the 100 evaluation seeds 101-200, and found_at = queries until each of the 5 known epitopes of watchlist.csv was found; produced by ../work/run_policies.sh).
Settings of the runs: Beta(0.01, 0.99) prior, k = 50 clusters per round, equal allocation, batch 50, seed set 5,000, UCB c = 2.0, epsilon-greedy eps = 0.02 (selected on the tuning seeds 1-20).
Usage: python 1-baselines.py     (conda environment 'thompson': numpy, pandas, matplotlib; run from any directory)
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
from tsbench import metrics                               # noqa: E402  (normalised AUC and queries-to-fraction of a discovery curve)

FIG.mkdir(exist_ok=True)
N_BINDERS = 3393

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
BLUE, ORANGE, GREEN, GREY, LGREY, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984", "#b5b4ae", "#e6e5e1"   # TS, UCB, epsilon-greedy, cluster-uniform, uniform, grid lines
LAB_FS, TICK_FS, LEG_FS, SPINE_W = 30, 24, 20, 2.5                                                      # label, tick and legend font sizes; spine width


def load(f):
    """grid (queried peptides) and curves (runs x len(grid): cumulative binders found) of one .npz file"""
    z = np.load(f)
    return z["grid"], z["curves"]


POLICIES = ["random", "ts", "ucb", "epsgreedy", "cluster_uniform"]
SEQ = {k: DATA / "curves_sequence" / f"{k}.npz" for k in POLICIES}                       # on the sequence clusters (CD-HIT 0.5)
RND = {k: DATA / "curves_randpart" / f"{k}.npz" for k in POLICIES if k != "random"}      # on random partitions of the same sizes (null control)

lib = pd.read_csv(DATA / "library_cdhit_0.5.csv")
print(f"library: {len(lib):,} peptides, {lib.cluster.nunique():,} clusters, {int(lib.label.sum()):,} binders")

# ---- summary of every policy
rows = []
for key in POLICIES:
    g, c = load(SEQ[key])
    auc = metrics.auc_paper(g, c, N_BINDERS)                       # area under the discovery curve, seed excluded; uniform random = 0.5
    q50 = metrics.queries_to_fraction(g, c, N_BINDERS, 0.5)        # queries to recover 50% of the binders
    rows.append(dict(policy=key, runs=len(c), AUC=auc.mean(), AUC_sd=auc.std(ddof=1), q50=q50.mean(), q50_se=q50.std(ddof=1) / np.sqrt(len(q50)),
                     binders_at_30k=c[:, np.searchsorted(g, 30000)].mean(),
                     binders_at_30k_randpart=load(RND[key])[1][:, np.searchsorted(g, 30000)].mean() if key in RND else np.nan,
                     q50_randpart=metrics.queries_to_fraction(*load(RND[key]), N_BINDERS, 0.5).mean() if key in RND else np.nan))
summary = pd.DataFrame(rows).set_index("policy")
print(summary.round(4).to_string())

# ---- figure
fig = plt.figure(figsize=(30, 15.5))
outer = GridSpec(1, 2, figure=fig, width_ratios=[20, 7], wspace=0.14)                              # A-C on the left, the portrait panel D on the right
gs = outer[0].subgridspec(2, 2, height_ratios=[1, 1], hspace=0.42, wspace=0.2)
axA = fig.add_subplot(gs[0, :]); axB = fig.add_subplot(gs[1, 0]); axC = fig.add_subplot(gs[1, 1])
gsD = outer[1].subgridspec(5, 1, hspace=0)                                           # five panels without spacing
axD = [fig.add_subplot(gsD[i, 0]) for i in range(5)]

# A: discovery curves (error bars: SD across the 100 runs at every 10,000 peptides; small offsets keep the bars apart)
SER = [("random", "Uniform random", LGREY, ":", -1.5), ("ts", "Thompson sampling", BLUE, "-", -0.5), ("ucb", "UCB", ORANGE, "-", 0.5), ("epsgreedy", "ε-greedy", GREEN, "-.", 1.5)]
for key, lab, col, ls, off in SER:
    g, c = load(SEQ[key])
    axA.plot(g / 1000, c.mean(0), color=col, ls=ls, lw=2.4, label=f'{lab} (AUC = {summary.loc[key, "AUC"]:.3f})')
    xs = np.arange(10000, g.max(), 10000); idx = np.searchsorted(g, xs)
    axA.errorbar(xs / 1000 + off, c[:, idx].mean(0), yerr=c[:, idx].std(0, ddof=1), fmt="o", ms=12, color=col, mec="black", mew=1.6, capsize=6, lw=2.4, capthick=2.4)
axA.set_xlabel("Peptides queried (×10$^3$)", fontsize=LAB_FS); axA.set_ylabel("Cumulative binders found", fontsize=LAB_FS)
axA.set_xlim(0, 146); axA.set_ylim(0, 3600); axA.legend(fontsize=LEG_FS, frameon=True, loc="lower right")

# B: queries to recover 50% of the binders (mean +- SE)
BARS = [("ts", "TS", BLUE), ("ucb", "UCB", ORANGE), ("epsgreedy", "ε-greedy", GREEN), ("cluster_uniform", "Cluster\nuniform", GREY), ("random", "Uniform", LGREY)]
y = np.arange(len(BARS))[::-1]
for yi, (key, lab, col) in zip(y, BARS):
    m, se = summary.loc[key, "q50"], summary.loc[key, "q50_se"]
    axB.barh(yi, m / 1000, height=0.6, color=col, edgecolor="black", linewidth=2, xerr=se / 1000, error_kw=dict(capsize=4, lw=1.8))
    axB.text(m / 1000 + 2.0, yi, f"{m / 1000:.1f}k", va="center", fontsize=LEG_FS)
axB.set_yticks(y); axB.set_yticklabels([x[1] for x in BARS]); axB.set_xlabel("Queries to recover 50% of binders (×10$^3$)", fontsize=LAB_FS)
axB.set_xlim(0, 100); axB.xaxis.grid(True, color=GRID, lw=0.8); axB.set_axisbelow(True)

# C: queries to recover 50% of the binders, sequence clusters (filled) vs random partitions (open); dotted line = uniform random (same axis as B)
ROWS = [("ts", "TS", BLUE), ("ucb", "UCB", ORANGE), ("epsgreedy", "ε-greedy", GREEN), ("cluster_uniform", "Cluster\nuniform", GREY)]
rnd, rnd_se = summary.loc["random", "q50"] / 1000, summary.loc["random", "q50_se"] / 1000
axC.axvline(rnd, color=LGREY, ls=":", lw=3); axC.axvspan(rnd - rnd_se, rnd + rnd_se, color=LGREY, alpha=0.25, lw=0)
for yi, (key, lab, col) in zip(np.arange(len(ROWS))[::-1], ROWS):
    s, r = summary.loc[key, "q50"] / 1000, summary.loc[key, "q50_randpart"] / 1000
    axC.plot([s, r], [yi, yi], color=GRID, lw=7, solid_capstyle="round", zorder=1)
    axC.scatter([s], [yi], s=330, color=col, zorder=3, edgecolor="black", linewidth=1.5)
    axC.scatter([r], [yi], s=330, facecolor="white", edgecolor=col, linewidth=3.5, zorder=3)
axC.set_yticks(np.arange(len(ROWS))[::-1]); axC.set_yticklabels([r[1] for r in ROWS]); axC.set_xlabel("Queries to recover 50% of binders (×10$^3$)", fontsize=LAB_FS)
axC.set_xlim(0, 100); axC.xaxis.grid(True, color=GRID, lw=0.8); axC.set_axisbelow(True)
axC.scatter([], [], s=200, color="k", label="sequence clusters"); axC.scatter([], [], s=200, facecolor="white", edgecolor="k", linewidth=2.5, label="random partitions")
axC.legend(fontsize=LEG_FS, frameon=True, loc="lower left")

# D: known ET-binding epitopes, queries until found (median and interquartile range of the 100 runs)
EPITOPES = ["BRG1", "INO80B", "CHD4", "NSD3", "BICRA"]                                    # order of data/watchlist.csv
for b, a in enumerate(axD):
    for yi, (key, lab, col) in zip(np.arange(len(BARS))[::-1], BARS):
        f = np.load(SEQ[key])["found_at"][:, b].astype(float) / 1000
        f = f[f >= 0]
        q1, med, q3 = np.percentile(f, [25, 50, 75])
        a.plot([q1, q3], [yi, yi], color=col, lw=6, solid_capstyle="round", zorder=2)
        a.plot(med, yi, "o", color=col, ms=15, mec="black", mew=1.6, zorder=3)
    a.set_yticks(np.arange(len(BARS))[::-1]); a.set_yticklabels([]); a.set_ylim(-0.6, len(BARS) - 0.4)
    a.set_xlim(0, 130); a.set_xticks([0, 50, 100]); a.xaxis.grid(True, color=GRID, lw=0.8); a.set_axisbelow(True)
    a.text(0.97, 0.93, EPITOPES[b], transform=a.transAxes, ha="right", va="top", fontsize=LAB_FS - 6, fontweight="bold")
    if b < 4:
        a.tick_params(labelbottom=False)
axD[-1].set_xlabel("Queries until found (×10$^3$)", fontsize=LAB_FS)

for a in (axA, axB, axC, *axD):
    a.tick_params(axis="both", which="major", labelsize=TICK_FS, width=2, length=6)
    for sp in a.spines.values():
        sp.set_linewidth(SPINE_W)

# panel letters (B is placed at the same x as A, in figure coordinates)
inv = fig.transFigure.inverted()
xA, yA = inv.transform(axA.transAxes.transform((-0.125, 1.04))); yB = inv.transform(axB.transAxes.transform((0, 1.04)))[1]
fig.text(xA, yA, "A", fontsize=34, fontweight="bold"); fig.text(xA, yB, "B", fontsize=34, fontweight="bold")
axC.text(-0.16, 1.04, "C", transform=axC.transAxes, fontsize=34, fontweight="bold")
fig.text(inv.transform(axD[0].transAxes.transform((-0.07, 1.04)))[0], yA, "D", fontsize=34, fontweight="bold")

out = FIG / "1-baselines.png"
fig.savefig(out, dpi=130, bbox_inches="tight")
print("saved", out)
