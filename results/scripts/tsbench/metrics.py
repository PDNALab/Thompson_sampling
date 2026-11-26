import numpy as np

_trapz = getattr(np, "trapezoid", None) or np.trapz  # np.trapz was removed in NumPy 2


def auc_paper(queried, cum, total_binders):
    """Normalized AUC as defined in the paper (seed excluded).

    AUC = int (cum - y0) dx / ((x_end - x_0) * (B - y0)); random sampling ~ 0.5.
    """
    queried = np.asarray(queried, dtype=float)
    cum = np.asarray(cum, dtype=float)
    y0 = cum[..., 0:1]
    area = _trapz(cum - y0, queried, axis=-1)
    return area / ((queried[-1] - queried[0]) * (total_binders - y0[..., 0]))


def binders_at(grid, curves, q):
    """Cumulative binders at q queries (last grid point <= q)."""
    j = np.searchsorted(grid, q, side="right") - 1
    return curves[..., j]


def queries_to_fraction(grid, curves, total_binders, frac):
    """First grid point where cumulative binders >= frac * total (nan if never)."""
    need = frac * total_binders
    hit = curves >= need
    first = hit.argmax(axis=-1)
    out = grid[first].astype(float)
    out[~hit.any(axis=-1)] = np.nan
    return out


def summarize(grid, curves, total_binders, budgets=(30000, 50000, 70000), fracs=(0.25, 0.5, 0.75)):
    """Per-replicate metrics for a (R, len(grid)) matrix of cumulative binders."""
    res = {"auc": auc_paper(grid, curves, total_binders)}
    for q in budgets:
        res[f"binders_{q // 1000}k"] = binders_at(grid, curves, q)
    for f in fracs:
        res[f"q{int(round(f * 100))}"] = queries_to_fraction(grid, curves, total_binders, f)
    return res
