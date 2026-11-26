"""Generates ../figures/0-hyperparam.png (panels A-E) and ../figures/0-clusters.png (the 12 clusterings): hyperparameter sensitivity of the UCB and epsilon-greedy baselines on the BRD3 library (CD-HIT 0.5 clusters).

  A  UCB: queries needed to recover 50% of the 3,393 binders for exploration weight c = 0.05 ... 5 (mean +- SE over the 100 evaluation seeds 101-200)
  B  epsilon-greedy: the same for exploration rate epsilon = 0.01 ... 0.3 (same y-axis range as A)
  C-E  mean AUC (seed excluded; uniform random = 0.5) of UCB (c = 2.0), epsilon-greedy (epsilon = 0.02) and Thompson sampling on the tuning seeds 1-20 for Beta priors with prior mean m = alpha0/(alpha0+beta0)
       (y) and prior strength s = alpha0+beta0 (x); orange box = prior with the highest mean AUC (the one used in 1-baselines.png)

Input  (../data): hp_sensitivity/seeds101-200/{ucb_c<c>,epsgreedy_e<eps>}.npz  (cumulative binders after every batch of 50 queries; produced by ../work/run_hp_sensitivity.sh; Beta(0.2, 9.8) prior, k = 50, equal allocation, batch 50, seed set 5,000)
        hp_prior/summary_prior_tuning.csv, hp_prior/selected.json                (produced by ../work/run_hp_prior.sh and ../work/select_prior.py)

0-clusters.png  discovery curves of Thompson sampling, UCB (c = 2.0), epsilon-greedy (eps = 0.02) and uniform random sampling on the 12 clusterings of the library (columns: CD-HIT, MMseqs2 easy-cluster = LINCLUST/MMseqs2,
     easy-linclust = LINCLUST; rows: identity 0.4, 0.5, 0.7, 0.9; mean of the 100 evaluation seeds 101-200; error bars = SD across the runs at every 10,000 peptides; number of clusters in each panel)
  Input: curves_clusterings/<method>_<thr>/{ts,ucb,epsgreedy}.npz, cluster_counts.csv, curves_sequence/random.npz   (../work/run_clusterings.sh, ../work/make_cluster_inputs.py, ../work/run_policies.sh)
  The settings were selected on CD-HIT 0.5 with the tuning seeds 1-20 and not re-tuned for the other clusterings; Beta(0.01, 0.99) prior for all methods.
Panels A and B are labelled in the legend by the method (evaluation seeds 101-200).
Usage: python 0-hyperparam.py     (makes both figures; conda environment 'thompson': numpy, pandas, matplotlib; run from any directory)
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                                        # results/1-policies
DATA, FIG = ROOT / "data", ROOT / "figures"
sys.path.insert(0, str(ROOT.parent / "scripts"))          # results/scripts/tsbench
from tsbench import metrics                               # noqa: E402

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
ORANGE, GREEN, GRID = "#eb6834", "#1baf7a", "#e6e5e1"        # UCB, epsilon-greedy (as in 1-baselines.png), grid lines
BLUE, LGREY = "#2a78d6", "#b5b4ae"                           # Thompson sampling, uniform random (as in 1-baselines.png)
LAB_FS, TICK_FS, LEG_FS, SPINE_W = 30, 24, 20, 2.5          # label, tick and legend font sizes; spine width
CELL_FS = 17                                                # font size of the numbers in the heatmaps


def q50_of(f):
    """queries to recover 50% of the binders of every run of one .npz file -> mean, SE"""
    z = np.load(f)
    q = metrics.queries_to_fraction(z["grid"], z["curves"], N_BINDERS, 0.5)
    return np.nanmean(q), np.nanstd(q, ddof=1) / np.sqrt(len(q))


# ---- top row: sensitivity of the hyperparameter, evaluation seeds 101-200
SENS = DATA / "hp_sensitivity" / "seeds101-200"
UCB_C = ["0.05", "0.1", "0.25", "0.5", "1.0", "1.41", "2.0", "5.0"]
EPS = ["0.01", "0.02", "0.05", "0.1", "0.2", "0.3"]
TOP = [("ucb", "ucb_c", UCB_C, ORANGE, "Exploration weight $c$"), ("epsgreedy", "epsgreedy_e", EPS, GREEN, "Exploration rate ε")]
sens = {}
for key, stem, vals, _, _ in TOP:
    sens[key] = [q50_of(SENS / f"{stem}{v}.npz") for v in vals]
    print(key, " ".join(f"{v}:{m / 1000:.2f}±{s / 1000:.2f}" for v, (m, s) in zip(vals, sens[key])))

# ---- bottom row: Beta-prior sweep, tuning seeds 1-20
D = pd.read_csv(DATA / "hp_prior" / "summary_prior_tuning.csv")
sel = json.load(open(DATA / "hp_prior" / "selected.json"))
D = D[D.m < 0.4]                                             # the 6 x 6 grid; the two uninformative priors Beta(1,1) and Beta(0.5,0.5) are left out
ms, ss = sorted(D.m.unique()), sorted(D.s.unique())
vmin, vmax = D.auc_mean.min(), D.auc_mean.max()              # one colour scale for the three methods
BOTTOM = [("ucb", "ucb"), ("epsgreedy", "epsgreedy"), ("ts", "ts")]

# ---- figure (all sizes in inches, converted to figure fractions)
W, H = 21, 14.4
fy = lambda y: 1 - y / H                                     # distance from the top (inches) -> figure fraction
fig = plt.figure(figsize=(W, H))
LEFT, RIGHT = 0.10, 0.96
gsT = GridSpec(1, 2, figure=fig, left=LEFT, right=RIGHT, top=fy(0.45), bottom=fy(6.0), wspace=0.28)
gsB = GridSpec(1, 4, figure=fig, left=LEFT, right=RIGHT, top=fy(8.1), bottom=fy(13.95), wspace=0.30, width_ratios=[1, 1, 1, 0.06])
axes = []

YLIM = (17, 20)                                              # same y-axis range for A and B (bars are cut at 17,000 queries)
for j, (key, stem, vals, col, xl) in enumerate(TOP):
    a = fig.add_subplot(gsT[0, j]); axes.append(a)
    x = np.arange(len(vals)); m = np.array([s[0] for s in sens[key]]) / 1000; se = np.array([s[1] for s in sens[key]]) / 1000
    a.bar(x, m, width=0.7, color=col, edgecolor="black", linewidth=2, yerr=se, error_kw=dict(capsize=5, lw=2, capthick=2))
    a.set_xticks(x); a.set_xticklabels(vals); a.set_xlabel(xl, fontsize=LAB_FS); a.set_ylabel("Queries to 50% recovery (×10$^3$)", fontsize=LAB_FS)
    a.set_ylim(*YLIM); a.set_yticks(range(17, 21)); a.yaxis.grid(True, color=GRID, lw=0.8); a.set_axisbelow(True)
    a.legend(handles=[Patch(facecolor=col, edgecolor="black", linewidth=2, label={"ucb": "UCB", "epsgreedy": "ε-greedy"}[key])], fontsize=LEG_FS, frameon=True, loc="upper right")

for j, (meth, skey) in enumerate(BOTTOM):
    a = fig.add_subplot(gsB[0, j]); axes.append(a)
    M = np.full((len(ms), len(ss)), np.nan)
    for _, r in D[D.method == meth].iterrows():
        M[ms.index(r.m), ss.index(r.s)] = r.auc_mean
    im = a.imshow(M, origin="lower", aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
    for i in range(len(ms)):
        for k in range(len(ss)):
            a.text(k, i, f"{M[i, k]:.3f}", ha="center", va="center", fontsize=CELL_FS, color="white" if M[i, k] < (vmin + vmax) / 2 else "black")
    b = sel[skey]                                            # prior with the highest mean AUC
    a.add_patch(plt.Rectangle((ss.index(b["s"]) - 0.5, ms.index(b["m"]) - 0.5), 1, 1, fill=False, ec=ORANGE, lw=4.5))
    a.set_xticks(range(len(ss))); a.set_xticklabels([f"{s:g}" for s in ss]); a.set_yticks(range(len(ms))); a.set_yticklabels([f"{m:g}" for m in ms])
    a.set_xlabel("Prior strength", fontsize=LAB_FS)
    if j == 0:
        a.set_ylabel("Prior mean", fontsize=LAB_FS)
    a.tick_params(which="minor", length=0)
cax = fig.add_subplot(gsB[0, 3])
cb = fig.colorbar(im, cax=cax); cb.set_label("Mean AUC", fontsize=LAB_FS, labelpad=18); cb.ax.tick_params(labelsize=TICK_FS, width=2, length=6); cb.outline.set_linewidth(SPINE_W)

for a in axes:
    a.tick_params(axis="both", which="major", labelsize=TICK_FS, width=2, length=6)
    for sp in a.spines.values():
        sp.set_linewidth(SPINE_W)

# panel letters
for a, letter, dx in zip(axes, "ABCDE", (-0.17, -0.17, -0.27, -0.12, -0.12)):
    a.text(dx, 1.04, letter, transform=a.transAxes, fontsize=34, fontweight="bold")

out = FIG / "0-hyperparam.png"
fig.savefig(out, dpi=130, bbox_inches="tight")
print("saved", out)
plt.close(fig)


# ---- discovery curves on the 12 clusterings
CL = DATA / "curves_clusterings"
counts = pd.read_csv(DATA / "cluster_counts.csv").set_index(["method", "thr"]).n_clusters
COLS = [("cdhit", "CD-HIT"), ("easycluster", "LINCLUST/MMseqs2"), ("easylinclust", "LINCLUST")]
THR = ["0.4", "0.5", "0.7", "0.9"]
SER = [("random", "Uniform random", LGREY, ":", -1.5), ("ts", "Thompson sampling", BLUE, "-", -0.5), ("ucb", "UCB", ORANGE, "-", 0.5), ("epsgreedy", "ε-greedy", GREEN, "-.", 1.5)]


def load(f):
    z = np.load(f)
    return z["grid"], z["curves"]


# ---- second figure: 0-clusters.png
W, H = 21, 16.2
fy = lambda y: 1 - y / H
fig = plt.figure(figsize=(W, H))
LEFT, RIGHT = 0.10, 0.96
gsF = GridSpec(4, 3, figure=fig, left=LEFT, right=RIGHT, top=fy(0.8), bottom=fy(0.8 + 4 * 3.6), wspace=0, hspace=0)   # no spacing between the panels
axes = []

# 4 x 3 panels without spacing; tick labels only on the outer edges
axF = np.empty((4, 3), dtype=object)
for i, thr in enumerate(THR):
    for j, (m, mname) in enumerate(COLS):
        a = axF[i, j] = fig.add_subplot(gsF[i, j]); axes.append(a)
        for key, lab, col, ls, off in SER:
            g, c = load(DATA / "curves_sequence" / "random.npz" if key == "random" else CL / f"{m}_{thr}" / f"{key}.npz")
            a.plot(g / 1000, c.mean(0), color=col, ls=ls, lw=2.0, label=lab if (i == 0 and j == 0) else None)
            xs = np.arange(10000, g.max(), 10000); idx = np.searchsorted(g, xs)
            a.errorbar(xs / 1000 + off, c[:, idx].mean(0), yerr=c[:, idx].std(0, ddof=1), fmt="o", ms=7, color=col, mec="black", mew=1.0, capsize=3.5, lw=1.5, capthick=1.5)
        a.text(0.96, 0.05, f"{counts[(m, float(thr))]:,} clusters", transform=a.transAxes, ha="right", fontsize=LEG_FS)
        a.set_xlim(0, 146); a.set_ylim(0, 3700); a.set_xticks([0, 40, 80, 120]); a.set_yticks([0, 1000, 2000, 3000])
        a.yaxis.grid(True, color=GRID, lw=0.8); a.set_axisbelow(True)
        if i == 0:
            a.set_title(mname, fontsize=LAB_FS - 4, fontweight="bold", pad=10)
        if j == 0:
            a.set_ylabel(f"Identity {thr}", fontsize=LAB_FS - 4)
        else:
            a.tick_params(labelleft=False)
        if i < 3:
            a.tick_params(labelbottom=False)
axF[0, 0].legend(fontsize=LEG_FS, frameon=True, loc="lower right", bbox_to_anchor=(1.0, 0.12))
axF[3, 1].set_xlabel("Peptides queried (×10$^3$)", fontsize=LAB_FS)
fig.text(0.012, fy(0.8 + 2 * 3.6), "Cumulative binders found", fontsize=LAB_FS, rotation=90, va="center", ha="center")

for a in axes:
    a.tick_params(axis="both", which="major", labelsize=TICK_FS, width=2, length=6)
    for sp in a.spines.values():
        sp.set_linewidth(SPINE_W)

out = FIG / "0-clusters.png"
fig.savefig(out, dpi=130, bbox_inches="tight")
print("saved", out)
