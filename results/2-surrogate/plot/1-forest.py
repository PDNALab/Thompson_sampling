"""Generates ../figures/1-forest.png: a random-forest surrogate versus the cluster-based bandits on the BRD3 library (142,338 peptides, 3,393 binders, CD-HIT 0.5 clusters).

  A  cumulative binders found against peptides queried: greedy selection with a default random forest (RF-greedy), greedy logistic regression on unscaled and standardized (scaled) features,
     UCB over the clusters with a random pick inside the cluster (the paper arm), uniform random sampling; lines = mean, points with error bars = mean and SD at every 10,000 peptides
     (RF and LR: seeds 101-103; UCB and random: 100 seeds), the format of 1-policies/figures/1-baselines.png A
  B  feature-vector ablation of RF-greedy at 30,000 queries (the format of 1-baselines.png C): each of the five feature vectors (AAC, DPC, PHYS, NTERM, CTERM; 470 features) left out, (-) (open circle),
     replaced by noise, (+) ~  of the same width (every column independently permuted across peptides; light circle) or used alone (filled circle); points = mean, error bars = SD of seeds 101-103;
     violet dotted line = full forest, grey dotted line = uniform random (explained in the caption, not labelled)
  C  what a smarter pick inside the cluster adds to each bandit (Thompson sampling, UCB, epsilon-greedy), two plots stacked without spacing: binders found by the hybrid minus binders found by the same
     bandit with the paper's random pick, paired by seed (line = mean of seeds 101-103, band = min-max); top: nearest-known-binder pick (3-mer Tanimoto), bottom: the forest's highest-probability peptide
  D  as B (points with error bars: mean and SD of seeds 101-103) at 30,000 queries for UCB with a forest picker (the forest ranks the peptides inside the clusters UCB chooses) and for RF-greedy,
     the forest restricted to the listed features; (-) = vector left out, plain name = vector only; dashed line = UCB arm with a random pick, dotted line = uniform random (explained in the caption, not labelled)
     (long portrait panel on the right, as D of 1-baselines.png)

Input  (../data; produced by the scripts in ../work): orders in which the peptides were queried (library rows, seed set first), seeds 101-103:
        rf_greedy_default_refit1, lr_greedy_default_refit1, lr_greedy_scaled_refit1, rf_ablation_{minus,only}_<vector>, rf_noise_<vector>,
        bandit_{ts,ucb,eps_greedy}_{nn3,rf}, bandit_ucb_rf_{minus,only}_<vector>;
        ../../1-policies/data/library_cdhit_0.5.csv, curves_sequence/{ucb,random}.npz (the paper arms, 100 evaluation seeds 101-200).
The UCB / Thompson sampling / epsilon-greedy arms with a random pick on seeds 101-103 (reference for C-D) are the runs of seeds 101-103 in curves_sequence/{ts,ucb,epsgreedy}.npz (nothing is simulated here).
Settings of every bandit: Beta(0.01, 0.99) prior, k = 50 clusters per round, equal allocation (one peptide per cluster), batch 50, seed set 5,000, UCB c = 2.0, epsilon-greedy eps = 0.02.
Usage: python 1-forest.py     (conda environment 'thompson': numpy, pandas, matplotlib; run from any directory; prints the numbers behind the figure)
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgba
from matplotlib.gridspec import GridSpec

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                                        # results/2-surrogate
DATA, FIG = ROOT / "data", ROOT / "figures"
POL_DATA = ROOT.parent / "1-policies" / "data"
sys.path.insert(0, str(ROOT.parent / "scripts"))          # results/scripts/tsbench
from tsbench import Library                              # noqa: E402

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
BLUE, ORANGE, GREEN, LGREY, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#b5b4ae", "#e6e5e1"          # TS, UCB, epsilon-greedy (as in all the other figures), uniform random, grid lines
DARK = "#333333"
RFC, LRU, LRS = "#7b4fc4", "#d6409f", "#8b5a2b"                                                   # RF-greedy, LR-greedy unscaled, LR-greedy scaled
LAB_FS, TICK_FS, LEG_FS, SPINE_W = 30, 24, 20, 2.5                                          # label, tick and legend font sizes; spine width
YT_FS = 18                                                                                    # row labels of the dot plots

# ---- library and metrics
ALPHA0, BETA0 = 0.01, 0.99
POLICY_KW = {"ts": {}, "ucb": {"c": 2.0}, "eps_greedy": {"eps": 0.02}}
SEEDS = (101, 102, 103)
BLOCKS = ["AAC", "DPC", "PHYS", "NTERM", "CTERM"]
BLOCK_N = dict(AAC=20, DPC=400, PHYS=10, NTERM=20, CTERM=20)
lib_df = pd.read_csv(POL_DATA / "library_cdhit_0.5.csv")
lib = Library(lib_df.label.values, lib_df.cluster.values)
y = lib.labels
NB = int(y.sum())
print(f"library: {len(y):,} peptides, {NB:,} binders")


def order(prefix, s):
    return np.load(DATA / f"{prefix}_seed{s}.npz")["order"]


def cum(o, n=None):
    """cumulative binders after q = 1..n queries of the order o"""
    return np.cumsum(y[o[:n]])


def f30(o_of):
    """binders found at 30,000 queries for the three seeds; o_of(seed) -> order"""
    return np.array([y[o_of(s)[:30000]].sum() for s in SEEDS])


def tidy(ax, axis="y"):
    ax.grid(axis=axis, color=GRID, lw=0.8); ax.set_axisbelow(True)


# ---- paper arms with a random pick on seeds 101-103: the reference of C-F, read from the runs of 1-policies (curves every 50 queries; same simulator, settings and seeds as the hybrids)
Q70 = 70000
ARM_GRID = np.arange(5000, Q70 + 1, 250)
arm = {}
for p, f in (("ts", "ts"), ("ucb", "ucb"), ("eps_greedy", "epsgreedy")):
    z = np.load(POL_DATA / "curves_sequence" / f"{f}.npz", allow_pickle=True)
    seeds_all = [int(x) for x in z["seeds"]]
    qidx = np.searchsorted(z["grid"], np.arange(1, Q70 + 1), side="right") - 1          # step function: binders after q queries = value at the last grid point <= q
    for s in SEEDS:
        arm[(p, s)] = z["curves"][seeds_all.index(s)][qidx]
arm_ucb = np.array([arm[("ucb", s)] for s in SEEDS])                      # seeds x 70,000 (exact at multiples of 50 queries)
arm30 = arm_ucb[:, 29999]
rnd = np.load(POL_DATA / "curves_sequence" / "random.npz", allow_pickle=True)
rnd30 = rnd["curves"][:, np.searchsorted(rnd["grid"], 30000)].mean()
rnd_sel = rnd["grid"] <= 30000
print(f"UCB arm (random pick), seeds {SEEDS}: binders at 30k {arm30.round(0)} mean {arm30.mean():.0f}; uniform random {rnd30:.0f}")

# ---- figure
fig = plt.figure(figsize=(30, 21))
outer = GridSpec(1, 2, figure=fig, width_ratios=[24, 7], wspace=0.06)                                      # A-C on the left, the long portrait panel D on the right (as in 1-baselines)
gsL = outer[0].subgridspec(2, 2, height_ratios=[0.8, 1.1], hspace=0.24, wspace=0.24)
axA = fig.add_subplot(gsL[0, :]); axB = fig.add_subplot(gsL[1, 0])
gsC = gsL[1, 1].subgridspec(2, 1, hspace=0)                                                                  # C: two plots stacked without spacing
axC, axD = fig.add_subplot(gsC[0, 0]), fig.add_subplot(gsC[1, 0])
axDD = fig.add_subplot(outer[1])


def pts(ax, x, curves, xs, col, off, ms):
    """points with error bars (mean and SD across the runs) at the query counts xs; curves: runs x (queries) cumulative binders, curve[q - 1] after q queries"""
    ax.errorbar(xs / 1000 + off, curves[:, xs - 1].mean(0), yerr=curves[:, xs - 1].std(0, ddof=1), fmt="o", ms=ms, color=col, mec="black", mew=1.6, capsize=6, lw=2.4, capthick=2.4)


def grid_curves(z, xs):
    """curves of a paper-arm .npz (every 50 queries) at the query counts xs -> runs x len(xs)"""
    return z["curves"][:, np.searchsorted(z["grid"], xs)]


# ---------------------------------------------------------------- A: discovery curves (the format of 1-baselines A)
Q = np.arange(1, Q70 + 1)
XA = np.arange(10000, Q70 + 1, 10000)
OFFA = (0.0,) * 5                                                       # points sit exactly on the curves
SERA = [("RF-greedy", "rf_greedy_default_refit1", RFC, "-"), ("LR-greedy (unscaled)", "lr_greedy_default_refit1", LRU, "-"), ("LR-greedy (scaled)", "lr_greedy_scaled_refit1", LRS, "-")]
for (name, pre, col, ls), off in zip(SERA, OFFA[:3]):
    c = np.array([cum(order(pre, s), Q70) for s in SEEDS])                  # seeds x 70,000 (cumulative binders)
    axA.plot(Q[4999:] / 1000, c.mean(0)[4999:], color=col, lw=2.4, ls=ls, label=name)
    pts(axA, Q, c, XA, col, off, 11)
    at = lambda q: c.mean(0)[q - 1]                              # noqa: E731
    print(f"A {name:24s} binders at 10k {at(10000):6.0f}  30k {at(30000):6.0f}  70k {at(70000):6.0f}")
for (name, f, col, ls), off in zip((("UCB", "ucb", ORANGE, "-"), ("Uniform random", "random", LGREY, ":")), OFFA[3:]):
    z = np.load(POL_DATA / "curves_sequence" / f"{f}.npz", allow_pickle=True)
    sel = z["grid"] <= Q70
    axA.plot(z["grid"][sel] / 1000, z["curves"].mean(0)[sel], color=col, lw=2.4, ls=ls, label=name)
    axA.errorbar(XA / 1000 + off, grid_curves(z, XA).mean(0), yerr=grid_curves(z, XA).std(0, ddof=1), fmt="o", ms=11, color=col, mec="black", mew=1.6, capsize=6, lw=2.4, capthick=2.4)
    m = z["curves"].mean(0)
    print(f"A {name:24s} binders at 10k {m[np.searchsorted(z['grid'], 10000)]:6.0f}  30k {m[np.searchsorted(z['grid'], 30000)]:6.0f}  70k {m[np.searchsorted(z['grid'], 70000)]:6.0f}")
ts = np.load(POL_DATA / "curves_sequence" / "ts.npz", allow_pickle=True)
print("A Thompson sampling (paper arm)  at 10k/30k/70k:", [round(float(ts["curves"].mean(0)[np.searchsorted(ts["grid"], q)])) for q in (10000, 30000, 70000)])
axA.set_xlim(0, 74); axA.set_ylim(0, 3600); axA.set_xlabel("Peptides queried (×10$^3$)", fontsize=LAB_FS); axA.set_ylabel("Cumulative binders found", fontsize=LAB_FS)
axA.legend(loc="lower right", fontsize=LEG_FS - 2, frameon=True)

# ---------------------------------------------------------------- B: feature-vector ablation of RF-greedy (the format of 1-baselines C)
def order_rf(cond, s):
    if cond == "full":
        return order("rf_greedy_default_refit1", s)
    return order(f"rf_noise_{cond.split('_')[1]}" if cond.startswith("noise") else f"rf_ablation_{cond}", s)


FACE = dict(full=RFC, only=RFC, minus="white", noise=to_rgba(RFC, 0.3))                       # filled = only / full, light = replaced by noise, open = left out
full = f30(lambda s: order_rf("full", s))
rows = [("full", "full (470)", "full")]
for b in BLOCKS:
    rows += [(f"minus_{b}", f"(-) {b} ({470 - BLOCK_N[b]})", "minus"), (f"noise_{b}", f"(+) ~{b} (470)", "noise"), (f"only_{b}", f"{b} ({BLOCK_N[b]})", "only")]
ypos, k = {}, 0.0
for cn, *_ in rows:
    ypos[cn] = -k; k += 1.0
    if cn == "full" or cn.startswith("only"):
        k += 0.45
axB.axvline(full.mean(), color=RFC, lw=3, ls=":", zorder=1); axB.axvline(rnd30, color=LGREY, lw=3, ls=":", zorder=1)
print("\n== B  RF-greedy at 30k (binders): mean [seeds]   difference to full")
for cn, lab, kind in rows:
    v = f30(lambda s: order_rf(cn, s))
    if kind != "full":
        axB.plot([full.mean(), v.mean()], [ypos[cn]] * 2, color=GRID, lw=7, solid_capstyle="round", zorder=1)       # drop relative to the full forest
    axB.errorbar(v.mean(), ypos[cn], xerr=v.std(ddof=1), fmt="o", ms=18, mfc=FACE[kind], mec=RFC, mew=2.8, ecolor=RFC, elinewidth=3, capsize=7, capthick=3, zorder=3)     # mean and SD of the 3 seeds
    print(f"{cn:12s} {v.mean():6.0f} {v.tolist()}   {v.mean() - full.mean():+5.0f}")
axB.set_yticks([ypos[cn] for cn, *_ in rows]); axB.set_yticklabels([r[1] for r in rows], fontsize=YT_FS)
axB.set_xlim(0, 3050); axB.set_ylim(min(ypos.values()) - 0.8, 0.8); axB.set_xlabel("Binders after 30k queries", fontsize=LAB_FS); tidy(axB, "x")

# ---------------------------------------------------------------- C (two plots): what the pick inside the cluster adds to each bandit
POL = [("ts", "Thompson sampling", BLUE), ("ucb", "UCB", ORANGE), ("eps_greedy", "ε-greedy", GREEN)]
print("\n== C (top: nearest binder, bottom: forest)  gain of the hybrid over the same bandit with a random pick (binders, mean of paired seeds)")
for ax, w, nm in ((axC, "nn3", "nearest binder"), (axD, "rf", "forest")):
    for p, name, col in POL:
        d = np.array([np.cumsum(y[order(f"bandit_{p}_{w}", s)[:Q70]])[ARM_GRID - 1] - arm[(p, s)][ARM_GRID - 1] for s in SEEDS])
        ax.fill_between(ARM_GRID / 1000, d.min(0), d.max(0), color=col, alpha=0.12, lw=0)
        ax.plot(ARM_GRID / 1000, d.mean(0), color=col, lw=3.2, label=name)
        at = lambda q: d.mean(0)[np.searchsorted(ARM_GRID, q)]  # noqa: E731
        print(f"{nm:14s} {p:11s} @10k {at(10000):5.0f} @20k {at(20000):5.0f} @30k {at(30000):5.0f} @50k {at(50000):5.0f} @70k {at(70000):5.0f}   peak {d.mean(0).max():4.0f} at {ARM_GRID[d.mean(0).argmax()] / 1000:.1f}k")
    ax.axhline(0, color=DARK, lw=2)
    ax.set_xlim(5, 70); tidy(ax)
axD.set_xlabel("Peptides queried (×10$^3$)", fontsize=LAB_FS); axC.tick_params(labelbottom=False)
axC.legend(loc="upper right", fontsize=LEG_FS)
lim = (min(axC.get_ylim()[0], axD.get_ylim()[0]), max(axC.get_ylim()[1], axD.get_ylim()[1]))
axC.set_ylim(*lim); axD.set_ylim(*lim)
for ax in (axC, axD):
    ax.set_yticks([0, 100, 200, 300])
axC.text(0.03, 0.04, "Nearest-binder pick (3-mer Tanimoto)", transform=axC.transAxes, fontsize=LEG_FS, va="bottom")
axD.text(0.03, 0.04, "Forest pick", transform=axD.transAxes, fontsize=LEG_FS, va="bottom")

# ---------------------------------------------------------------- D: UCB + forest picker vs RF-greedy under the same feature sets
conds = ["full"] + [f"minus_{x}" for x in BLOCKS] + [f"only_{x}" for x in BLOCKS]
lab = {"full": "full (470)", **{f"minus_{x}": f"(-) {x} ({470 - BLOCK_N[x]})" for x in BLOCKS}, **{f"only_{x}": f"{x} ({BLOCK_N[x]})" for x in BLOCKS}}
yp, k = {}, 0.0
for cn in conds:
    yp[cn] = -k; k += 1.0
    if cn in ("full", "minus_CTERM"):
        k += 0.5
order_hy = lambda cn, s: order("bandit_ucb_rf" if cn == "full" else f"bandit_ucb_rf_{cn}", s)       # noqa: E731
axDD.axvline(arm30.mean(), color=ORANGE, lw=3, ls="--", zorder=1); axDD.axvline(rnd30, color=LGREY, lw=3, ls=":", zorder=1)
print("\n== D  30k binders: RF-greedy | UCB + forest picker | gain of the hybrid over the UCB arm")
for cn in conds:
    vr, vh = f30(lambda s: order_rf(cn, s)), f30(lambda s: order_hy(cn, s))
    axDD.plot([vr.mean(), vh.mean()], [yp[cn]] * 2, color=GRID, lw=7, solid_capstyle="round", zorder=1)
    for v, col in ((vh, ORANGE), (vr, RFC)):                                               # both on the row, so that the grey line connects the two dots
        axDD.errorbar(v.mean(), yp[cn], xerr=v.std(ddof=1), fmt="o", ms=17, color=col, mec="black", mew=1.4, ecolor=col, elinewidth=3, capsize=7, capthick=3, zorder=3)      # mean and SD of the 3 seeds
    print(f"{cn:12s} RF-greedy {vr.mean():6.0f} | UCB+forest {vh.mean():6.0f} | vs UCB arm {vh.mean() - arm30.mean():+5.0f}")
axDD.set_yticks([yp[cn] for cn in conds]); axDD.set_yticklabels([lab[cn] for cn in conds], fontsize=YT_FS)
axDD.yaxis.tick_right(); axDD.tick_params(axis="y", which="both", right=True, left=False)
axDD.set_xlim(0, 3050); axDD.set_ylim(min(yp.values()) - 0.8, 0.8); axDD.set_xlabel("Binders after 30k queries", fontsize=LAB_FS); tidy(axDD, "x")
axDD.legend(handles=[plt.Line2D([], [], marker="o", ls="", color=ORANGE, mec="black", ms=12, label="UCB + forest picker"), plt.Line2D([], [], marker="o", ls="", color=RFC, mec="black", ms=12, label="RF-greedy (no clusters)")],
           loc="upper left", bbox_to_anchor=(0.02, 0.985), fontsize=LEG_FS - 2, frameon=True)

for a in (axA, axB, axC, axD, axDD):
    a.tick_params(axis="both", which="major", labelsize=TICK_FS, width=2, length=6)
    for sp in a.spines.values():
        sp.set_linewidth(SPINE_W)
for a in (axB, axDD):
    a.tick_params(axis="y", labelsize=YT_FS)

# y label of C, centred on the two stacked plots
inv = fig.transFigure.inverted()
FW = fig.get_figwidth()
p1, p2 = axC.get_position(), axD.get_position()
fig.text(p1.x0 - 1.15 / FW, (p1.y1 + p2.y0) / 2, "Gain in binders found", fontsize=LAB_FS, rotation=90, va="center", ha="center")

# panel letters: A and B at one x (left of the longest row labels, those of B); C and D above their plots
xL = axA.get_position().x0 - 2.3 / FW
FH = fig.get_figheight()
yT = lambda a: a.get_position().y1 + 0.3 / FH                                                  # noqa: E731  (the same distance above every plot)
for letter, x, a in (("A", xL, axA), ("B", xL, axB), ("C", p1.x0 - 0.35 / FW, axC), ("D", axDD.get_position().x0 - 0.35 / FW, axDD)):
    fig.text(x, yT(a), letter, fontsize=34, fontweight="bold")

out = FIG / "1-forest.png"
fig.savefig(out, dpi=130, bbox_inches="tight")
print("saved", out)
