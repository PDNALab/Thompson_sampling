"""Cluster diagnostics for a labelled library: how much label information do the clusters carry?
stats(): number of clusters, singleton share, permutation-corrected mutual information between cluster and label, clusters holding 50 % of the positives vs a size-matched
random partition, neighbour label agreement (leave-one-out) and an oracle that ranks clusters by their true precision."""
import numpy as np, pandas as pd


def H(p): p = np.clip(p, 1e-12, 1-1e-12); return -(p*np.log2(p)+(1-p)*np.log2(1-p))
def mi(g, y):
    cnt = np.bincount(g); b = np.bincount(g, weights=y); k = cnt > 0; return H(y.mean()) - (cnt[k]/len(y)*H(b[k]/cnt[k])).sum()
def stats(lib, scheme, g, y, rng):
    """cluster diagnostics; permutation-corrected excess mutual information"""
    g = pd.factorize(g)[0]; cnt = np.bincount(g); b = np.bincount(g, weights=y); n = len(y); nb = int(y.sum())
    k50 = lambda gg: int(np.searchsorted(np.cumsum(np.sort(np.bincount(gg, weights=y))[::-1])/nb, 0.5)+1)
    m = mi(g, y); mp = np.mean([mi(g, rng.permutation(y)) for _ in range(5)])
    k50r = np.mean([k50(rng.permutation(g)) for _ in range(3)])
    sz = cnt[g]; ns = sz >= 2; loo = np.where(ns, (b[g]-y)/np.maximum(sz-1, 1), np.nan)
    prec = b/cnt; o = np.argsort(-prec, kind='stable'); cp = np.cumsum(cnt[o])/n; cy = np.cumsum(b[o])/nb      # oracle: arms ranked by TRUE precision
    orc = lambda f: float(np.interp(f, cp, cy))
    r = dict(lib=lib, scheme=scheme, N=n, positives=nb, base_rate=nb/n, n_arms=len(cnt), singleton_frac=float((cnt == 1).mean()),
        size_min=int(cnt.min()), size_median=float(np.median(cnt)), size_p90=float(np.percentile(cnt, 90)), size_max=int(cnt.max()),
        arms_with_positive=int((b > 0).sum()), arms_for_50pct_pos=k50(g), arms_for_50pct_pos_randpart=k50r, mi_bits=m, mi_perm_bits=mp,
        excess_mi_frac_of_H=(m-mp)/H(nb/n), loo_rate_given_pos=float(np.nanmean(loo[y == 1])), loo_rate_given_neg=float(np.nanmean(loo[y == 0])),
        max_arm_precision=float(prec.max()), pos_share_in_arms_prec_ge_50=float(b[prec >= .5].sum()/nb), pos_share_in_arms_prec_ge_20=float(b[prec >= .2].sum()/nb),
        oracle_pos_at_1pct=orc(.01), oracle_pos_at_5pct=orc(.05), oracle_pos_at_10pct=orc(.10), oracle_pos_at_25pct=orc(.25))
    return r, cnt, (cp, cy)
