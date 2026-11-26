"""Generates ../figures/1-lbtope.png: clustering plus active learning on the LBtope linear B-cell epitope library (Fixed set: 20-mers; positive = epitope).

  A  fraction of the CD-HIT 0.5 clusters in each size bin, for the unmodified library (black) and the sparse libraries (grey; mean of the five libraries; cf. 0-cluster_info.png A)
  B  fraction of the positives contained in the clusters with the most positives (clusters ranked by their number of positives; cf. 0-cluster_info.png B), unmodified library (black) and sparse libraries
     (grey; mean of the five libraries, each curve interpolated on a common grid of % of clusters)
  C  discovery curves on the unmodified library (32,628 peptides, 37 % positives): RF-greedy (default random forest, 470 BRD3 features), Thompson sampling with a random pick inside the cluster,
     Thompson sampling with the forest as picker inside the cluster (TS + RF picker), uniform random sampling; lines = mean, points with error bars = mean and SD at every 2,000 queries
     (RF-greedy and TS + RF picker: seeds 101-105; Thompson sampling and random: 100 seeds)
  D  the same on the five sparse versions of the library (21,652 peptides, 5 % positives: all negatives, positives randomly held out; each with its own CD-HIT 0.5 clusters, RF-greedy and TS + RF picker seeds 101-103,
     Thompson sampling and random 100 seeds): mean over the five libraries, error bars = SD across the five libraries, every 1,000 queries   [SPARSE_VIEW = "mean"; "ds1".."ds5" shows one library]
  E  feature-vector ablation on the unmodified library after 50 % of the library has been queried: TS + RF picker (blue) and RF-greedy (violet) with the forest restricted to the listed features
     ((-) = vector left out; vector name alone = only that vector); points = mean, error bars = SD of seeds 101-105; dashed blue line = Thompson sampling arm with a random pick, dotted grey line = uniform random

Input  (../data/lbtope; scripts in ../work):
        fixed.csv, fixed__cdhit0.5.npy (library and clusters), hybrid/lbtope_rf_<cond>_cap0.5_seed<S>.npz and hybrid_b001/lbtope_hybrid_<cond>_cap0.5_seed<S>.npz (orders in which the peptides were queried),
        sparse/ds<i>/{library.csv, cdhit0.5.npy, hybrid/sparse_{rf,hybrid}_full_cap0.5_seed<S>.npz}, arms.npz (Thompson sampling arm and uniform random, 100 seeds; ../work/lbtope_arms.py).
Settings: Beta(0.01, 0.99) prior, k = 50 clusters per round, equal allocation, batch 50, seed set 3.5 % of the library (counted as queries), run to 50 % of the library; the BRD3 settings, not re-tuned.
Usage: python 1-lbtope.py     (conda environment 'thompson': numpy, pandas, matplotlib; run from any directory; prints the numbers behind the figure)
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                                        # results/4-additional
D, FIG = ROOT / "data" / "lbtope", ROOT / "figures"
FIG.mkdir(exist_ok=True)
SPARSE_VIEW = "mean"                                       # "mean" (all five sparse libraries) or "ds1" ... "ds5"

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
SEEDS_FULL, SEEDS_SPARSE = (101, 102, 103, 104, 105), (101, 102, 103)

# ---- libraries and runs
lib = pd.read_csv(D / "fixed.csv")
y = lib.label.values.astype(np.int8)
n, P = len(y), int(y.sum())
cl = np.load(D / "fixed__cdhit0.5.npy")
arms = np.load(D / "arms.npz")
print(f"unmodified library: {n:,} peptides, {P:,} positives ({P / n:.1%}), {len(np.unique(cl)):,} CD-HIT 0.5 clusters")


def cum_of(f, yy, Q):
    return np.cumsum(yy[np.load(f)["order"]])[:Q]


# ---------------------------------------------------------------- figure frame (positions in inches)
W, H = 30.6, 19.2
fig = plt.figure(figsize=(W, H))


def ax_in(x0, y0, w, h):
    return fig.add_axes([x0 / W, y0 / H, w / W, h / H])


Y2, H2 = 1.95, 8.0                                         # row 2 (C, D): bottom, height
Y1, H1 = 11.8, 6.2                                         # row 1 (A, B)
axA = ax_in(2.5, Y1, 6.2, H1)
axB = ax_in(11.0, Y1, 11.4, H1)
axC = ax_in(2.5, Y2, 9.0, H2)
axD = ax_in(13.6, Y2, 8.8, H2)
axE = ax_in(23.2, Y2, 4.8, Y1 + H1 - Y2)

# ---------------------------------------------------------------- A, B: clusters of the unmodified library (black) and of the five sparse libraries (grey, mean)
GREY = "#8a8984"
EDGES = [1, 2, 3, 5, 9, 17, 33, 65, np.inf]
LABELS = ["1", "2", "3-4", "5-8", "9-16", "17-32", "33-64", ">64"]
XS = np.linspace(0, 100, 2001)


def size_fractions(c):
    sz = np.bincount(c)
    sz = sz[sz > 0]
    return np.array([((sz >= lo) & (sz < hi)).mean() for lo, hi in zip(EDGES[:-1], EDGES[1:])]), len(sz)


def coverage(c, yy):
    """% of the positives in the clusters with the most positives, on the grid XS (% of clusters)"""
    pc = np.sort(np.bincount(c, weights=yy))[::-1]
    cp = np.cumsum(pc) / pc.sum()
    return np.interp(XS, np.r_[0, (np.arange(len(pc)) + 1) / len(pc) * 100], np.r_[0, cp]) * 100, pc


frac, ncl = size_fractions(cl)
cov, pc = coverage(cl, y)
sp_frac, sp_cov, sp_ncl = [], [], []
for i in range(1, 6):
    lb = pd.read_csv(D / "sparse" / f"ds{i}" / "library.csv")
    ci = np.load(D / "sparse" / f"ds{i}" / "cdhit0.5.npy")
    f_, n_ = size_fractions(ci); c_, _ = coverage(ci, lb.label.values.astype(np.int8))
    sp_frac.append(f_); sp_cov.append(c_); sp_ncl.append(n_)
sp_frac, sp_cov = np.mean(sp_frac, 0), np.mean(sp_cov, 0)

axA.bar(np.arange(8) - 0.2, frac, 0.38, color=INK, edgecolor="white", lw=0.6, label="Unmodified library")
axA.bar(np.arange(8) + 0.2, sp_frac, 0.38, color=GREY, edgecolor="white", lw=0.6, label="Sparse libraries (mean)")
axA.set_xticks(range(8)); axA.set_xticklabels(LABELS, rotation=45, ha="right")
axA.set_ylim(0, 0.5); axA.set_xlim(-0.6, 7.6); axA.yaxis.grid(True, color=GRID, lw=0.8); axA.set_axisbelow(True)
axA.set_xlabel("Cluster size", fontsize=LAB_FS); axA.set_ylabel("Fraction of clusters", fontsize=LAB_FS)
axA.legend(loc="upper right", fontsize=LEG_FS - 3, frameon=True)
print(f"A unmodified: {ncl:,} clusters, size bins", dict(zip(LABELS, frac.round(3))), f"singletons {frac[0]:.1%}")
print(f"A sparse (mean of five): {np.mean(sp_ncl):,.0f} clusters, size bins", dict(zip(LABELS, sp_frac.round(3))), f"singletons {sp_frac[0]:.1%}")

axB.plot(XS, cov, color=INK, lw=4, label="Unmodified library")
axB.plot(XS, sp_cov, color=GREY, lw=4, label="Sparse libraries (mean)")
axB.set_xlim(0, 100); axB.set_ylim(0, 102); axB.yaxis.grid(True, color=GRID, lw=0.8); axB.set_axisbelow(True)
axB.set_xlabel("Clusters, ranked by number of positives (% of clusters)", fontsize=LAB_FS); axB.set_ylabel("Positives covered (%)", fontsize=LAB_FS)
axB.legend(loc="lower right", fontsize=LEG_FS, frameon=True)
for f in (1, 5, 10, 25, 50):
    j = int(f / 100 * (len(XS) - 1))
    print(f"B top {f}% of the clusters hold {cov[j]:.1f}% (unmodified) and {sp_cov[j]:.1f}% (sparse, mean) of the positives")
print(f"B unmodified: clusters without any positive {(pc == 0).mean():.1%}")


# ---------------------------------------------------------------- C, D: discovery curves
class Series:
    """lines/points of one method: arr = runs x queries cumulative positives; q = the query counts of the columns"""

    def __init__(self, arr, q, xs):
        self.arr, self.q = arr, q
        self.gi = np.searchsorted(q, xs)


def order_series(c, S, QM, xs):
    """c: runs x queries (cumulative positives after q queries, column q - 1)"""
    return Series(c[:, S - 1:QM], np.arange(S, QM + 1), xs)


def draw(ax, items, xs):
    for name, ser, col, ls, show in items:
        m = ser.arr.mean(0)
        ax.plot(ser.q / 1000, m, color=col, ls=ls, lw=2.6, label=name if show else None)
        ax.errorbar(ser.q[ser.gi] / 1000, m[ser.gi], yerr=ser.arr.std(0, ddof=1)[ser.gi], fmt="o", ms=11, color=col, mec="black", mew=1.6, capsize=6, lw=2.4, capthick=2.4)


def library_series(tag, yy, seeds, rf_f, hy_f, step):
    nn = len(yy); S = int(round(0.035 * nn)); QM = int(round(0.5 * nn)); xs = np.arange(step, QM + 1, step)
    rf = np.array([cum_of(rf_f(s), yy, QM) for s in seeds]); hy = np.array([cum_of(hy_f(s), yy, QM) for s in seeds])
    g = arms[f"{tag}__grid"]
    out = dict(rf=order_series(rf, S, QM, xs), hy=order_series(hy, S, QM, xs), ts=Series(arms[f"{tag}__ts"], g, xs), rnd=Series(arms[f"{tag}__random"], g, xs), QM=QM, xs=xs, n=nn)
    return out


full = library_series("full", y, SEEDS_FULL, lambda s: D / "hybrid" / f"lbtope_rf_full_cap0.5_seed{s}.npz", lambda s: D / "hybrid_b001" / f"lbtope_hybrid_full_cap0.5_seed{s}.npz", 2000)
STYLE = [("RF-greedy", "rf", RFC, "-"), ("TS + RF picker", "hy", BLUE, "--"), ("Thompson sampling", "ts", BLUE, "-"), ("Uniform random", "rnd", LGREY, ":")]
draw(axC, [(nm, full[k], col, ls, True) for nm, k, col, ls in STYLE], full["xs"])
axC.set_xlim(0, full["QM"] / 1000 + 0.4); axC.set_ylim(0, full["rf"].arr.max() * 1.06)
axC.legend(loc="lower right", fontsize=LEG_FS, frameon=True)
for nm, k, *_ in STYLE:
    print(f"C {nm:18s} positives at 50 % of the library: {full[k].arr[:, -1].mean():8.1f}  (random {full['rnd'].arr[:, -1].mean():.1f})")

sp = {}
for i in range(1, 6):
    lb = pd.read_csv(D / "sparse" / f"ds{i}" / "library.csv")
    yy = lb.label.values.astype(np.int8)
    sp[f"ds{i}"] = library_series(f"ds{i}", yy, SEEDS_SPARSE, lambda s, i=i: D / "sparse" / f"ds{i}" / "hybrid" / f"sparse_rf_full_cap0.5_seed{s}.npz",
                                  lambda s, i=i: D / "sparse" / f"ds{i}" / "hybrid" / f"sparse_hybrid_full_cap0.5_seed{s}.npz", 1000)
    print(f"  sparse ds{i}: {len(yy):,} peptides, {int(yy.sum()):,} positives ({yy.mean():.1%})")
ref = sp["ds1"]
if SPARSE_VIEW == "mean":
    comb = {k: Series(np.array([sp[f"ds{i}"][k].arr.mean(0) for i in range(1, 6)]), sp["ds1"][k].q, ref["xs"]) for k in ("rf", "hy", "ts", "rnd")}   # one mean curve per library, SD across the five libraries
    note = "Sparse libraries (5 % positives),\nmean of five"
else:
    comb = {k: sp[SPARSE_VIEW][k] for k in ("rf", "hy", "ts", "rnd")}
    note = f"Sparse library {SPARSE_VIEW[2:]} (5 % positives)"
draw(axD, [(nm, comb[k], col, ls, False) for nm, k, col, ls in STYLE], ref["xs"])
axD.set_xlim(0, ref["QM"] / 1000 + 0.3); axD.set_ylim(0, max(comb["rf"].arr.max(), comb["hy"].arr.max(), comb["ts"].arr.max()) * 1.06)
for nm, k, *_ in STYLE:
    print(f"D {nm:18s} positives at 50 % of the library: {comb[k].arr[:, -1].mean():7.1f} +- {comb[k].arr[:, -1].std(ddof=1):5.1f} (SD over libraries)")
for a in (axC, axD):
    a.set_xlabel("Peptides queried (×10$^3$)", fontsize=LAB_FS); a.set_ylabel("Cumulative positives found", fontsize=LAB_FS)

# ---------------------------------------------------------------- E: feature-vector ablation, unmodified library, 50 % of the library
qE = int(full["ts"].q[-1])                                                                   # last batch end at or below 50 % of the library, the same query count for all methods
armv = full["ts"].arr[:, -1].mean()
lab = {"full": "full (470)", **{f"minus_{b}": f"(-) {b} ({470 - BN[b]})" for b in BL}, **{f"only_{b}": f"{b} ({BN[b]})" for b in BL}}
yp, k = {}, 0.0
for cn in CONDS:
    yp[cn] = -k; k += 1.0
    if cn in ("full", "minus_CTERM"):
        k += 0.5
axE.axvline(armv, color=BLUE, lw=3, ls="--", zorder=1); axE.axvline(qE * P / n, color=LGREY, lw=3, ls=":", zorder=1)
print(f"\nE after {qE} queries: Thompson sampling arm {armv:.0f}, random {qE * P / n:.0f}; per feature set RF-greedy | TS + RF picker | gain of the hybrid over the arm")
mx = 0
for cn in CONDS:
    vr = np.array([cum_of(D / "hybrid" / f"lbtope_rf_{cn}_cap0.5_seed{s}.npz", y, qE)[-1] for s in SEEDS_FULL])
    vh = np.array([cum_of(D / "hybrid_b001" / f"lbtope_hybrid_{cn}_cap0.5_seed{s}.npz", y, qE)[-1] for s in SEEDS_FULL])
    axE.plot([vr.mean(), vh.mean()], [yp[cn]] * 2, color=GRID, lw=7, solid_capstyle="round", zorder=1)
    for v, col in ((vh, BLUE), (vr, RFC)):
        axE.errorbar(v.mean(), yp[cn], xerr=v.std(ddof=1), fmt="o", ms=17, color=col, mec="black", mew=1.4, ecolor=col, elinewidth=3, capsize=7, capthick=3, zorder=3)
        mx = max(mx, v.mean() + v.std(ddof=1))
    print(f"  {cn:12s} {vr.mean():7.0f} | {vh.mean():7.0f} | {vh.mean() - armv:+5.0f}")
assert mx < 10200 and qE * P / n > 5000                                                       # the x-axis starts at 5,000 (random sampling finds ~6,000)
axE.set_yticks([yp[cn] for cn in CONDS]); axE.set_yticklabels([lab[cn] for cn in CONDS], fontsize=YT_FS)
axE.yaxis.tick_right(); axE.tick_params(axis="y", which="both", right=True, left=False)
axE.set_xlim(5000, 10200); axE.set_xticks([6000, 8000, 10000]); axE.set_ylim(min(yp.values()) - 0.8, 0.8); axE.xaxis.grid(True, color=GRID, lw=0.8); axE.set_axisbelow(True)
axE.set_xlabel(f"Positives after {qE / 1000:.1f}k queries", fontsize=LAB_FS - 6)
axE.legend(handles=[plt.Line2D([], [], marker="o", ls="", color=BLUE, mec="black", ms=12, label="TS + RF picker"), plt.Line2D([], [], marker="o", ls="", color=RFC, mec="black", ms=12, label="RF-greedy (no clusters)")],
           loc="upper left", bbox_to_anchor=(0.02, 0.985), fontsize=LEG_FS - 2, frameon=True)

for a in (axA, axB, axC, axD, axE):
    a.tick_params(axis="both", which="major", labelsize=TICK_FS, width=2, length=6)
    for s_ in a.spines.values():
        s_.set_linewidth(SPINE_W)
axE.tick_params(axis="y", labelsize=YT_FS)

# panel letters: at the same distance above every plot; B and D just above the left edge of their plots
for letter, x, y0 in (("A", 2.5 - 2.1, Y1 + H1), ("B", 11.0 - 0.6, Y1 + H1), ("C", 2.5 - 2.1, Y2 + H2), ("D", 13.6 - 0.6, Y2 + H2), ("E", 23.2 - 0.5, Y1 + H1)):
    fig.text(x / W, (y0 + 0.25) / H, letter, fontsize=34, fontweight="bold")

out = FIG / "1-lbtope.png"
fig.savefig(out, dpi=130, bbox_inches="tight")
print("saved", out)
