"""Analysis of the Beta-prior sweep (run_hp_prior.sh, tuning seeds 1-20): normalised AUC (seed excluded) of every (method, prior mean m, prior strength s); selects the prior with the highest mean AUC for each method.

Output ../data/hp_prior/summary_prior_tuning.csv (method, m, s, alpha0, beta0, n, auc_mean, auc_se, q50_mean, q50_se) and ../data/hp_prior/selected.json (the selected prior of ucb, epsgreedy and ts, with the number of priors
within one standard error of the best). Usage: python select_prior.py [data_dir]      (default ../data; conda environment 'thompson')
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "scripts"))        # results/scripts/tsbench
from tsbench import metrics                                 # noqa: E402

RES = (Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parent / "data") / "hp_prior"
N_BINDERS = 3393
rows = []
for f in sorted((RES / "seeds1-20").glob("*.npz")):
    z = np.load(f)
    g, c = z["grid"], z["curves"]
    auc = metrics.auc_paper(g, c, N_BINDERS)
    q50 = metrics.queries_to_fraction(g, c, N_BINDERS, 0.5)
    meth, tag = f.stem.split("_", 1)
    m, s = (float(x) for x in re.match(r"m([0-9.]+)_s([0-9.]+)", tag).groups())
    rows.append(dict(method=meth, m=m, s=s, alpha0=m * s, beta0=(1 - m) * s, n=len(c), auc_mean=auc.mean(), auc_se=auc.std(ddof=1) / np.sqrt(len(c)),
                     q50_mean=np.nanmean(q50), q50_se=np.nanstd(q50, ddof=1) / np.sqrt(len(c))))
D = pd.DataFrame(rows)
D.to_csv(RES / "summary_prior_tuning.csv", index=False)

sel = {}
for meth, g in D.groupby("method"):
    b = g.loc[g.auc_mean.idxmax()]
    within = int((g.auc_mean >= b.auc_mean - b.auc_se).sum())
    sel[meth] = dict(alpha0=float(b.alpha0), beta0=float(b.beta0), m=float(b.m), s=float(b.s), auc=float(b.auc_mean), se=float(b.auc_se), n_priors=int(len(g)), n_within_1se=within)
    print(f"{meth:10s} best prior Beta({b.alpha0:.4g}, {b.beta0:.4g}) (m {b.m}, s {b.s}): AUC {b.auc_mean:.4f} (SE {b.auc_se:.4f}); {within} of {len(g)} priors within 1 SE; worst {g.auc_mean.min():.4f}")
json.dump(sel, open(RES / "selected.json", "w"), indent=1)
