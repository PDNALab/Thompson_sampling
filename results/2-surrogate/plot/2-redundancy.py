"""Generates ../figures/2-redundancy.png: how redundant are the 50 peptides picked in one batch? (BRD3 library, first 25,000 queries, 5,000-peptide seed set, batch of 50)

  A  fraction of the 50 picks of a batch that have a sliding-window neighbour (shift <= 3 residues) among the other picks of the same batch, against the peptides queried (mean over the batches of a
     window of 10 rounds, one refit cycle of the forest). Broken y-axis: the forest (top) picks 10-30% redundant peptides, the bandits and uniform random (bottom, zoomed) pick almost none.
     RF-greedy: seeds 101-103; Thompson sampling, UCB, epsilon-greedy and uniform random: 100 seeds (101-200).
  C  one example batch (queries 10,000-10,050, seed 101): pairwise similarity of its 50 picks (best ungapped shift, identical residues / overlap), ordered by group, for RF-greedy (left) and UCB (right)

Input  (../data): batch_similarity.npz (RF-greedy, seeds 101-103; its picks give the example batch; ../work/batch_similarity_forest.py),
        batch_similarity_arms.npz (arms and uniform random, seeds 101-200; ../work/batch_similarity_arms.py), sequences.csv, ../../1-policies/data/library_cdhit_0.5.csv (row order of the picks).
Settings: Beta(0.01, 0.99) prior, k = 50 clusters per round, equal allocation, UCB c = 2.0, epsilon-greedy eps = 0.02 (1-baselines.png); forest (labelled RF-greedy): the greedy forest of the earlier ablation / stress tests
(100 trees, balanced_subsample class weights, max_samples 0.3, refit every 10 rounds, 470 BRD3 features), NOT the default forest refit every round of 1-forest.png.
Usage: python 2-redundancy.py     (conda environment 'thompson': numpy, pandas, matplotlib; run from any directory; prints the numbers behind the figure)
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                                        # results/2-surrogate
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
BLUE, ORANGE, GREEN, LGREY, RFC, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#b5b4ae", "#7b4fc4", "#e6e5e1"   # TS, UCB, epsilon-greedy, uniform random, RF-greedy, grid lines
LAB_FS, TICK_FS, LEG_FS, SPINE_W = 30, 24, 20, 2.5

zf = np.load(DATA / "batch_similarity.npz")
za = np.load(DATA / "batch_similarity_arms.npz")
S, B = int(zf["S"]), int(zf["B"])
nb = zf["forest__stats"].shape[1]
q = (S + B * np.arange(nb) + B) / 1000                                                      # peptides queried (x1000) after each batch


def smooth(y, w=10):
    """moving average over w rounds (the forest is refit every 10 rounds, which gives a saw-tooth)"""
    return np.convolve(np.pad(y, (w // 2, w // 2 - 1), mode="edge"), np.ones(w) / w, mode="valid")


SER = [("forest", "RF-greedy", RFC, "-", zf), ("random", "Uniform random", LGREY, ":", za), ("ts", "Thompson sampling", BLUE, "-", za), ("ucb", "UCB", ORANGE, "-", za), ("epsgreedy", "ε-greedy", GREEN, "-.", za)]
curve = {k: 100 * z[f"{k}__stats"][:, :, 0].mean(0) for k, _, _, _, z in SER}              # similar picks (%) per batch, mean over the seeds
for k, name, _, _, z in SER:
    st = z[f"{k}__stats"]
    print(f"{name:18s} seeds {st.shape[0]:3d}  similar picks {100 * st[:, :, 0].mean():6.2f}%  (first 10k queries {100 * st[:, :100, 0].mean():6.2f}%, last 5k {100 * st[:, -100:, 0].mean():6.2f}%)  groups {st[:, :, 1].mean():5.1f}  clusters {st[:, :, 2].mean():5.1f}")

# ---- pairwise similarity of the picks of one batch: best ungapped shift (|d| <= 12), identical residues / overlap
lib_df = pd.read_csv(ROOT.parent / "1-policies" / "data" / "library_cdhit_0.5.csv")
seq = pd.read_csv(DATA / "sequences.csv").set_index("id")["sequence"]
seqs = seq.loc[lib_df["id"]].values


def sim(a, b):
    best = 0.0
    for d in range(-12, 13):
        x, y = (a[d:], b[:len(b) - d]) if d >= 0 else (a[:len(a) + d], b[-d:])
        best = max(best, sum(u == v for u, v in zip(x, y)) / len(x))
    return best


def matrix(picks):
    m = len(picks)
    M = np.eye(m)
    for i in range(m):
        for j in range(i + 1, m):
            M[i, j] = M[j, i] = sim(seqs[picks[i]], seqs[picks[j]])
    par = list(range(m))                                                                 # order: single-linkage groups of pairs with similarity >= 0.8, largest group first

    def f(a):
        while par[a] != a:
            par[a] = par[par[a]]; a = par[a]
        return a
    for i in range(m):
        for j in range(i + 1, m):
            if M[i, j] >= 0.8:
                par[f(i)] = f(j)
    g = np.array([f(i) for i in range(m)])
    sizes = {k: (g == k).sum() for k in set(g)}
    order = sorted(range(m), key=lambda i: (-sizes[g[i]], g[i]))
    return M[np.ix_(order, order)], sorted(sizes.values(), reverse=True)


T = int(round((10000 - S) / B))
mats = {}
for k in ("forest", "ucb"):
    mats[k] = matrix(zf[f"{k}__picks"][0, T])
    print(k, "example batch (seed 101, queries 10,000-10,050): group sizes", mats[k][1][:12], "number of groups", len(mats[k][1]))

# ---- figure: A (narrow, broken y-axis) and the two heatmaps of B, all of the same height; positions in inches
W, H = 31.0, 10.0
fig = plt.figure(figsize=(W, H))


def ax_in(x0, y0, w, h, **kw):
    return fig.add_axes([x0 / W, y0 / H, w / W, h / H], **kw)


Y0, HT = 1.4, 7.6                                          # bottom and height (inches) of A and of the heatmaps
GAPA, A_BOT = 0.12, 3.2                                    # gap of the axis break, height of the zoomed (bottom) part of A
XA, WA = 2.4, 6.6                                          # left edge and width of A
axL = ax_in(XA, Y0, WA, A_BOT)
axT = ax_in(XA, Y0 + A_BOT + GAPA, WA, HT - A_BOT - GAPA)
XC1 = XA + WA + 2.7
axC1 = ax_in(XC1, Y0, HT, HT)
axC2 = ax_in(XC1 + HT + 1.6, Y0, HT, HT)
cax = ax_in(XC1 + 2 * HT + 1.6 + 0.8, Y0, 0.35, HT)

axT.plot(q, smooth(curve["forest"]), color=RFC, lw=4, label="RF-greedy")
for k, name, col, ls, _ in SER[1:]:
    axL.plot(q, smooth(curve[k]), color=col, ls=ls, lw=3.5, label=name)
top_max = max(smooth(curve[k]).max() for k, *_ in SER[1:])
lo_top = 5 * np.floor(smooth(curve["forest"]).min() / 5)
axT.set_ylim(lo_top, 72); axT.set_yticks(range(int(lo_top), 71, 10) if lo_top % 10 == 0 else range(int(lo_top) + 5, 71, 10))
ymax = np.ceil(np.max([curve[k].max() for k, *_ in SER[1:]]) * 1.3 * 20) / 20                # the raw curves of the bandits and random stay below
axL.set_ylim(0, 0.42); axL.set_yticks([0, 0.1, 0.2, 0.3, 0.4])
for a in (axT, axL):
    a.set_xlim(5, 25); a.set_xticks([5, 10, 15, 20, 25]); a.yaxis.grid(True, color=GRID, lw=0.8); a.set_axisbelow(True)
axT.tick_params(labelbottom=False, bottom=False); axT.spines["bottom"].set_visible(False); axL.spines["top"].set_visible(False)
axT.tick_params(axis="x", which="both", bottom=False); axL.tick_params(axis="x", which="both", top=False)
axL.set_xlabel("Peptides queried (×10$^3$)", fontsize=LAB_FS)
h1, l1 = axT.get_legend_handles_labels(); h2, l2 = axL.get_legend_handles_labels()
axT.legend(h1 + h2, l1 + l2, fontsize=LEG_FS, frameon=True, loc="upper right")                # one legend for all five methods, in the empty part of the top panel
for a, sp in ((axT, "bottom"), (axL, "top")):                                               # axis-break marks on both spines
    kw = dict(transform=a.transAxes, color="black", clip_on=False, lw=SPINE_W)
    y0 = 0.0 if sp == "bottom" else 1.0
    a.plot((-0.012, 0.012), (y0 - 0.025, y0 + 0.025), **kw); a.plot((0.988, 1.012), (y0 - 0.025, y0 + 0.025), **kw)

for ax, k, t in ((axC1, "forest", "RF-greedy"), (axC2, "ucb", "UCB")):
    im = ax.imshow(mats[k][0], cmap="viridis", vmin=0.1, vmax=1.0, origin="upper", aspect="equal")
    ax.set_title(t, fontsize=LAB_FS - 4, pad=10)
    ax.set_xticks([0, 24, 49]); ax.set_xticklabels([1, 25, 50]); ax.set_yticks([0, 24, 49]); ax.set_yticklabels([1, 25, 50])
    ax.set_xlabel("Pick in batch", fontsize=LAB_FS - 4)
axC1.set_ylabel("Pick in batch", fontsize=LAB_FS - 4)
cb = fig.colorbar(im, cax=cax)
cb.set_label("Sequence similarity", fontsize=LAB_FS - 4, labelpad=18); cb.ax.tick_params(labelsize=TICK_FS - 4, width=2, length=6); cb.outline.set_linewidth(SPINE_W)

for a in (axT, axL, axC1, axC2):
    a.tick_params(axis="both", which="major", labelsize=TICK_FS, width=2, length=6)
    for sp in a.spines.values():
        sp.set_linewidth(SPINE_W)
axT.spines["bottom"].set_visible(False); axL.spines["top"].set_visible(False)

# shared y label of A, panel letters
FW, FH = fig.get_figwidth(), fig.get_figheight()
fig.text((XA - 1.35) / FW, (Y0 + HT / 2) / FH, "Similar picks in batch (%)", fontsize=LAB_FS, rotation=90, va="center", ha="center")
yL = (Y0 + HT + 0.3) / FH
fig.text((XA - 2.0) / FW, yL, "A", fontsize=34, fontweight="bold")
fig.text((XC1 - 1.9) / FW, yL, "B", fontsize=34, fontweight="bold")

out = FIG / "2-redundancy.png"
fig.savefig(out, dpi=130, bbox_inches="tight")
print("saved", out)
