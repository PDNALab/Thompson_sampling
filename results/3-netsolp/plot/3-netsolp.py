"""Generates ../figures/3-netsolp.png: the selection policies on a second, model-derived label, NetSolP predicted solubility (same 142,338 peptides and CD-HIT 0.5 clusters as the BRD3 library).

  A, B, C  discovery curves (mean of 100 runs; points with error bars = mean and SD across the runs at every 10,000 peptides, as in 1-baselines A) of Thompson sampling (TS), UCB, epsilon-greedy and uniform random sampling when a peptide counts as a "binder" if its
           predicted solubility is above 0.865 (A, 2,757 peptides, ~2% of the library), 0.82 (B, 7,382, ~5%) or 0.77 (C, 14,532, ~10%); the AUC of each method is in the legend

Input  (../data): curves/{top2,top5,top10}/{random,ts,ucb,epsgreedy}.npz (cumulative positives after every batch of 50 queries, evaluation seeds 101-200; ../work/run_netsolp.sh),
        si_top{2,5,10}.csv (labels; number of positives).
Settings of the runs: those of 1-policies/figures/1-baselines.png (A), not re-tuned for NetSolP: Beta(0.01, 0.99) prior, k = 50 clusters per round, equal allocation, batch 50, seed set 5,000, UCB c = 2.0, epsilon-greedy eps = 0.02.
Usage: python 3-netsolp.py     (conda environment 'thompson': numpy, pandas, matplotlib; run from any directory)
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                                        # results/3-netsolp
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
BLUE, ORANGE, GREEN, LGREY = "#2a78d6", "#eb6834", "#1baf7a", "#b5b4ae"          # TS, UCB, epsilon-greedy, uniform random (as in 1-baselines)
LAB_FS, TICK_FS, LEG_FS, SPINE_W = 30, 24, 20, 2.5

SER = [("random", "Uniform random", LGREY, ":", -1.5), ("ts", "Thompson sampling", BLUE, "-", -0.5), ("ucb", "UCB", ORANGE, "-", 0.5), ("epsgreedy", "ε-greedy", GREEN, "-.", 1.5)]   # last = small x offset of the error bars
LABS = [("top2", "Predicted solubility > 0.865 (~2%)"), ("top5", "Predicted solubility > 0.82 (~5%)"), ("top10", "Predicted solubility > 0.77 (~10%)")]

fig, axes = plt.subplots(1, 3, figsize=(30, 8.6), gridspec_kw=dict(wspace=0.27))
for ax, (lab, title) in zip(axes, LABS):
    n = int(pd.read_csv(DATA / f"si_{lab}.csv")["label"].sum())
    for key, name, col, ls, off in SER:
        z = np.load(DATA / "curves" / lab / f"{key}.npz")
        g, c = z["grid"], z["curves"]
        auc = metrics.auc_paper(g, c, n); q50 = metrics.queries_to_fraction(g, c, n, 0.5)
        print(f"{lab:6s} {key:10s} n_pos={n:6d} runs={len(c)} AUC={auc.mean():.4f}+-{auc.std(ddof=1):.4f} q50={np.nanmean(q50):8.0f}")
        ax.plot(g / 1000, c.mean(0), color=col, ls=ls, lw=2.4, label=f"{name} (AUC = {auc.mean():.3f})")
        xs = np.arange(10000, g.max(), 10000); idx = np.searchsorted(g, xs)                        # error bars: SD across the 100 runs at every 10,000 peptides
        ax.errorbar(xs / 1000 + off, c[:, idx].mean(0), yerr=c[:, idx].std(0, ddof=1), fmt="o", ms=12, color=col, mec="black", mew=1.6, capsize=6, lw=2.4, capthick=2.4)
    ax.set_title(title, fontsize=LAB_FS - 4, pad=14)
    ax.set_xlabel("Peptides queried (×10$^3$)", fontsize=LAB_FS); ax.set_ylabel("Cumulative positives found", fontsize=LAB_FS)
    ax.set_xlim(0, 146); ax.set_ylim(0, n * 1.03)
    ax.legend(fontsize=LEG_FS - 2, frameon=True, loc="lower right")
    ax.tick_params(axis="both", which="major", labelsize=TICK_FS, width=2, length=6)
    for sp in ax.spines.values():
        sp.set_linewidth(SPINE_W)

inv = fig.transFigure.inverted()
for ax, letter in zip(axes, "ABC"):
    x, y = inv.transform(ax.transAxes.transform((-0.2, 1.09)))
    fig.text(x, y, letter, fontsize=34, fontweight="bold")

out = FIG / "3-netsolp.png"
fig.savefig(out, dpi=130, bbox_inches="tight")
print("saved", out)
