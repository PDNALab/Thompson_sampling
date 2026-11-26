"""Generates ../figures/2-covid.png: clustering plus active learning on the SARS-CoV-2 peptide-array library ("Immunoreactive peptide maps of SARS-CoV-2", Mishra et al., Commun Biol 2021).

Library: 165,946 12-mers tiling the proteomes of seven coronaviruses with an 11-residue overlap; positive = reactive (signal > 10,000) in at least 10 of the 100 COVID-19 IgG samples and in none of the 10 healthy controls
(3,335 positives = 2.0 %). Peptides are clustered with CD-HIT at identity 0.5 (5,877 clusters).

  A  fraction of the CD-HIT 0.5 clusters in each size bin (cf. 1-lbtope.png A, 0-cluster_info.png A)
  B  fraction of the positives contained in the clusters with the most positives (clusters ranked by their number of positives; the x-axis is cut at 35 % of the clusters, all positives are in the first 24 %;
     cf. 1-lbtope.png B, 0-cluster_info.png B)
  C  discovery curves: RF-greedy (default random forest, 470 BRD3 features), Thompson sampling with a random pick inside the cluster, Thompson sampling with the forest as picker inside the cluster (TS + RF picker),
     uniform random sampling; lines = mean, points with error bars = mean and SD every 10,000 queries (RF-greedy and TS + RF picker: seeds 101-105; Thompson sampling and random: 100 seeds)
  D  feature-vector ablation after 50 % of the library has been queried: TS + RF picker (blue) and RF-greedy (violet) with the forest restricted to the listed features ((-) = vector left out; vector name alone =
     only that vector); points = mean, error bars = SD of seeds 101-103 (the "full" set: seeds 101-105); dashed blue line = Thompson sampling arm with a random pick, dotted grey line = uniform random

Input  (../data/covid; scripts in ../work):
        library.csv, cdhit0.5.npy (library and clusters), runs/peptide_{rf,hybrid}_<cond>_cap0.5_seed<S>.npz (orders in which the peptides were queried), arms.npz (Thompson sampling arm and uniform random, 100 seeds; ../work/covid_arms.py).
Settings: Beta(0.01, 0.99) prior, k = 50 clusters per round, equal allocation, batch 50, seed set 3.5 % of the library (counted as queries), run to 50 % of the library; the BRD3 settings, not re-tuned.
Usage: python 2-covid.py     (conda environment 'thompson': numpy, pandas, matplotlib; run from any directory; prints the numbers behind the figure)
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                                        # results/4-additional
D, FIG = ROOT / "data" / "covid", ROOT / "figures"
R = D / "runs"
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
BLUE, LGREY, RFC, INK, GRID = "#2a78d6", "#b5b4ae", "#7b4fc4", "#0b0b0b", "#e6e5e1"     # TS (and TS + RF picker), uniform random, RF-greedy, CD-HIT bars, grid lines
LAB_FS, TICK_FS, LEG_FS, SPINE_W, YT_FS = 30, 24, 20, 2.5, 18
BL = ["AAC", "DPC", "PHYS", "NTERM", "CTERM"]
BN = dict(AAC=20, DPC=400, PHYS=10, NTERM=20, CTERM=20)
CONDS = ["full"] + [f"minus_{b}" for b in BL] + [f"only_{b}" for b in BL]
SEEDS_FULL, SEEDS_ABL = (101, 102, 103, 104, 105), (101, 102, 103)

# ---- library and runs
y = pd.read_csv(D / "library.csv", usecols=["label"]).label.values.astype(np.int8)
n, P = len(y), int(y.sum())
cl = np.load(D / "cdhit0.5.npy")
arms = np.load(D / "arms.npz")
S = int(round(0.035 * n)); QM = int(round(0.5 * n))
print(f"library: {n:,} peptides, {P:,} positives ({P / n:.1%}), {len(np.unique(cl)):,} CD-HIT 0.5 clusters; seed set {S:,}, run to 50 % = {QM:,} queries")


def cum_of(f, Q):
    return np.cumsum(y[np.load(f)["order"]])[:Q]


# ---------------------------------------------------------------- figure frame (positions in inches)
W, H = 30.6, 19.2
fig = plt.figure(figsize=(W, H))


def ax_in(x0, y0, w, h):
    return fig.add_axes([x0 / W, y0 / H, w / W, h / H])


Y2, H2 = 1.95, 8.0                                         # row 2 (C): bottom, height
Y1, H1 = 11.8, 6.2                                         # row 1 (A, B)
axA = ax_in(2.5, Y1, 6.2, H1)
axB = ax_in(11.0, Y1, 11.4, H1)
axC = ax_in(2.5, Y2, 19.9, H2)
axD = ax_in(23.2, Y2, 4.8, Y1 + H1 - Y2)

# ---------------------------------------------------------------- A, B: clusters
EDGES = [1, 2, 3, 5, 9, 17, 33, 65, 129, np.inf]
LABELS = ["1", "2", "3-4", "5-8", "9-16", "17-32", "33-64", "65-128", ">128"]
sz = np.bincount(cl); sz = sz[sz > 0]
frac = np.array([((sz >= lo) & (sz < hi)).mean() for lo, hi in zip(EDGES[:-1], EDGES[1:])])
pc = np.sort(np.bincount(cl, weights=y))[::-1]
cov = np.cumsum(pc) / pc.sum() * 100
xc = np.arange(1, len(pc) + 1) / len(pc) * 100

axA.bar(np.arange(len(frac)), frac, 0.7, color=INK, edgecolor="white", lw=0.6)
axA.set_xticks(range(len(frac))); axA.set_xticklabels(LABELS, rotation=45, ha="right")
axA.set_ylim(0, 0.4); axA.set_xlim(-0.6, len(frac) - 0.4); axA.yaxis.grid(True, color=GRID, lw=0.8); axA.set_axisbelow(True)
axA.set_xlabel("Cluster size", fontsize=LAB_FS); axA.set_ylabel("Fraction of clusters", fontsize=LAB_FS)
print(f"A: {len(sz):,} clusters, size bins", dict(zip(LABELS, frac.round(3))), f"singletons {frac[0]:.1%}, median size {np.median(sz):.0f}, largest {sz.max()}")

axB.plot(np.r_[0, xc], np.r_[0, cov], color=INK, lw=4)
axB.set_xlim(0, 35); axB.set_ylim(0, 102); axB.yaxis.grid(True, color=GRID, lw=0.8); axB.set_axisbelow(True)
axB.set_xlabel("Clusters, ranked by number of positives (% of clusters)", fontsize=LAB_FS); axB.set_ylabel("Positives covered (%)", fontsize=LAB_FS)
for f in (1, 5, 10, 25, 50):
    print(f"B: top {f}% of the clusters hold {np.interp(f, xc, cov):.1f}% of the positives")
k50 = int(np.searchsorted(cov / 100, 0.5) + 1)
print(f"B: clusters without any positive {(pc == 0).mean():.1%}; {k50} clusters ({100 * k50 / len(pc):.1f}%) hold half of the positives; {(pc > 0).sum()} clusters ({100 * (pc > 0).sum() / len(pc):.1f}%) hold all of them")


# ---------------------------------------------------------------- C: discovery curves
def order_curves(files):
    return np.array([cum_of(f, QM) for f in files])


rf = order_curves([R / f"peptide_rf_full_cap0.5_seed{s}.npz" for s in SEEDS_FULL])
hy = order_curves([R / f"peptide_hybrid_full_cap0.5_seed{s}.npz" for s in SEEDS_FULL])
g = arms["grid"]
STEP = 10000
xs = np.arange(STEP, QM + 1, STEP)
q_o = np.arange(S, QM + 1)
SERIES = [("RF-greedy", rf[:, S - 1:], q_o, RFC, "-"), ("TS + RF picker", hy[:, S - 1:], q_o, BLUE, "--"),
          ("Thompson sampling", arms["ts"], g, BLUE, "-"), ("Uniform random", arms["random"], g, LGREY, ":")]
for nm, arr, q, col, ls in SERIES:
    m = arr.mean(0); gi = np.searchsorted(q, xs)
    axC.plot(q / 1000, m, color=col, ls=ls, lw=2.6, label=nm)
    axC.errorbar(q[gi] / 1000, m[gi], yerr=arr.std(0, ddof=1)[gi], fmt="o", ms=11, color=col, mec="black", mew=1.6, capsize=6, lw=2.4, capthick=2.4)
    print(f"C {nm:18s} positives at the end ({q[-1]:,} queries): {arr[:, -1].mean():8.1f}  at 10k / 20k / 40k: " + " / ".join(f"{m[np.searchsorted(q, v)]:.0f}" for v in (10000, 20000, 40000)))
axC.set_xlim(0, QM / 1000 + 1.5); axC.set_ylim(0, rf.max() * 1.04)
axC.legend(loc="lower right", fontsize=LEG_FS, frameon=True)
axC.set_xlabel("Peptides queried (×10$^3$)", fontsize=LAB_FS); axC.set_ylabel("Cumulative positives found", fontsize=LAB_FS)

# ---------------------------------------------------------------- D: feature-vector ablation, 50 % of the library
qE = int(g[-1])                                                                              # last batch end at or below 50 % of the library, the same query count for all methods
armv = arms["ts"][:, -1].mean()
lab = {"full": "full (470)", **{f"minus_{b}": f"(-) {b} ({470 - BN[b]})" for b in BL}, **{f"only_{b}": f"{b} ({BN[b]})" for b in BL}}
yp, k = {}, 0.0
for cn in CONDS:
    yp[cn] = -k; k += 1.0
    if cn in ("full", "minus_CTERM"):
        k += 0.5
axD.axvline(armv, color=BLUE, lw=3, ls="--", zorder=1); axD.axvline(qE * P / n, color=LGREY, lw=3, ls=":", zorder=1)
print(f"\nD after {qE} queries: Thompson sampling arm {armv:.0f}, random {qE * P / n:.0f}; per feature set RF-greedy | TS + RF picker | gain of the hybrid over the arm")
mx, mn = 0, 1e9
for cn in CONDS:
    seeds = SEEDS_FULL if cn == "full" else SEEDS_ABL
    vr = np.array([cum_of(R / f"peptide_rf_{cn}_cap0.5_seed{s}.npz", qE)[-1] for s in seeds])
    vh = np.array([cum_of(R / f"peptide_hybrid_{cn}_cap0.5_seed{s}.npz", qE)[-1] for s in seeds])
    axD.plot([vr.mean(), vh.mean()], [yp[cn]] * 2, color=GRID, lw=7, solid_capstyle="round", zorder=1)
    for v, col in ((vh, BLUE), (vr, RFC)):
        axD.errorbar(v.mean(), yp[cn], xerr=v.std(ddof=1), fmt="o", ms=17, color=col, mec="black", mew=1.4, ecolor=col, elinewidth=3, capsize=7, capthick=3, zorder=3)
        mx, mn = max(mx, v.mean() + v.std(ddof=1)), min(mn, v.mean() - v.std(ddof=1))
    print(f"  {cn:12s} {vr.mean():7.0f} | {vh.mean():7.0f} | {vh.mean() - armv:+5.0f}")
X0 = 1100
assert mn > X0 and mx < 3400 and qE * P / n > X0                                             # the x-axis starts at 1,100 so that the legend stays left of the points
axD.set_yticks([yp[cn] for cn in CONDS]); axD.set_yticklabels([lab[cn] for cn in CONDS], fontsize=YT_FS)
axD.yaxis.tick_right(); axD.tick_params(axis="y", which="both", right=True, left=False)
axD.set_xlim(X0, 3400); axD.set_xticks([1500, 2000, 2500, 3000]); axD.set_ylim(min(yp.values()) - 0.8, 0.8); axD.xaxis.grid(True, color=GRID, lw=0.8); axD.set_axisbelow(True)
axD.set_xlabel(f"Positives after {qE / 1000:.1f}k queries", fontsize=LAB_FS - 6)
axD.legend(handles=[plt.Line2D([], [], marker="o", ls="", color=BLUE, mec="black", ms=12, label="TS + RF picker"), plt.Line2D([], [], marker="o", ls="", color=RFC, mec="black", ms=12, label="RF-greedy (no clusters)")],
           loc="upper left", bbox_to_anchor=(0.02, 0.985), fontsize=LEG_FS - 4, frameon=True)

for a in (axA, axB, axC, axD):
    a.tick_params(axis="both", which="major", labelsize=TICK_FS, width=2, length=6)
    for s_ in a.spines.values():
        s_.set_linewidth(SPINE_W)
axD.tick_params(axis="y", labelsize=YT_FS)

# panel letters: at the same distance above every plot; B just above the left edge of its plot
for letter, x, y0 in (("A", 2.5 - 2.1, Y1 + H1), ("B", 11.0 - 0.6, Y1 + H1), ("C", 2.5 - 2.1, Y2 + H2), ("D", 23.2 - 0.5, Y1 + H1)):
    fig.text(x / W, (y0 + 0.25) / H, letter, fontsize=34, fontweight="bold")

out = FIG / "2-covid.png"
fig.savefig(out, dpi=130, bbox_inches="tight")
print("saved", out)
