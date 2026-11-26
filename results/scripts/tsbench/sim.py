"""Vectorized replay simulator for cluster-guided batch selection.

Mirrors `ts.py` (seed -> repeated rounds of: score arms, take top-k, allocate
the batch over them, pull random unqueried peptides from each, refill from the
next-best arms if some arm cannot fill its quota, update posteriors), but keeps
everything in memory and replaces per-round shuffling with a per-arm random
queue built once per replicate. Sampling inside an arm is still uniform without
replacement.

RNG call order differs from `ts.py`, so trajectories are not bit-identical to
the published runs; agreement is statistical (see scripts/validate.py).
"""
import numpy as np

from .policies import State


class Library:
    def __init__(self, labels, groups):
        labels = np.asarray(labels).astype(np.int8)
        _, inv = np.unique(np.asarray(groups).astype(str), return_inverse=True)
        self.labels = labels
        self.groups = inv.astype(np.int32)
        self.n = len(labels)
        self.n_groups = int(inv.max()) + 1
        self.total_binders = int(labels.sum())

    def random_partition(self, seed):
        """Same group-size distribution, group membership shuffled at random."""
        perm = np.random.default_rng(seed).permutation(self.n)
        new = object.__new__(Library)
        new.labels = self.labels
        new.groups = self.groups[perm]
        new.n = self.n
        new.n_groups = self.n_groups
        new.total_binders = self.total_binders
        return new


def make_grid(S, N, B):
    """Common query-count grid: S, S+B, ..., then N."""
    return np.append(np.arange(S, N, B), N)


def to_grid(queried, cum, grid):
    """Step-function resampling: cumulative binders at the last point <= each grid value."""
    idx = np.searchsorted(queried, grid, side="right") - 1
    return np.asarray(cum)[idx]


def _alloc(thetas, total, mode):
    """Port of ts.py alloc_quota (equal / proportional)."""
    m = len(thetas)
    if mode == "equal" and m <= total:
        q = np.full(m, total // m, dtype=np.int64)
        q[int(np.argmax(thetas))] += total - q.sum()
        return q
    if mode == "equal":
        q = np.full(m, total // m, dtype=np.int64)
    elif mode == "proportional":
        w = np.clip(thetas, 1e-9, None)
        q = np.floor(w / w.sum() * total).astype(np.int64)
    else:
        raise ValueError("alloc must be 'equal' or 'proportional'")
    for i in range(m):
        if q[i] == 0 and q.sum() < total:
            q[i] += 1
    while q.sum() < total:
        q[int(np.argmax(thetas))] += 1
    while q.sum() > total:
        q[int(np.argmin(thetas))] -= 1
    return q


def _watch_lookup(n, watch):
    if watch is None or len(watch) == 0:
        return None, np.zeros(0, dtype=np.int32)
    lookup = np.full(n, -1, dtype=np.int32)
    lookup[np.asarray(watch)] = np.arange(len(watch), dtype=np.int32)
    return lookup, np.full(len(watch), -1, dtype=np.int32)


def run(lib, policy, seed, S=5000, B=50, k=50, alloc="equal", alpha0=0.2, beta0=9.8,
        max_queries=None, watch=None, return_order=False, seed_exclude=None):
    """One replicate of an arm-based policy. Returns a dict of arrays."""
    rng = np.random.default_rng(seed)
    n, G = lib.n, lib.n_groups
    y, grp = lib.labels, lib.groups

    perm = rng.permutation(n)
    if seed_exclude is not None:   # rows that may not be in the seed (they stay in the pool); default None = unchanged behaviour
        ex = seed_exclude[perm]
        perm = np.concatenate([perm[~ex], perm[ex]])
    seed_idx, rest = perm[:S], perm[S:]

    ys, gs = y[seed_idx], grp[seed_idx]
    pos = np.bincount(gs, weights=ys, minlength=G)
    tot = np.bincount(gs, minlength=G)
    alpha = alpha0 + pos
    beta = beta0 + (tot - pos)

    g_rest = grp[rest]
    queue = rest[np.argsort(g_rest, kind="stable")]  # per-arm random order
    counts = np.bincount(g_rest, minlength=G)
    end = np.cumsum(counts)
    ptr = end - counts

    lookup, found = _watch_lookup(n, watch)
    if lookup is not None:
        w = lookup[seed_idx]
        found[w[w >= 0]] = S

    cum, queried = int(ys.sum()), len(seed_idx)
    Q, C = [queried], [cum]
    order_log = [seed_idx] if return_order else None
    t = 0
    while max_queries is None or queried < max_queries:
        act = np.flatnonzero(ptr < end)
        if len(act) == 0:
            break
        b = int(min(B, (end - ptr).sum()))
        m = min(k, len(act), b)
        st = State(alpha[act], beta[act], act, rng, t)
        sc = policy.scores(st)
        sel = policy.choose(sc, m, st)
        chosen = act[sel]
        quota = _alloc(sc[sel], b, alloc)

        picks, leftover = [], 0
        for c, qc in zip(chosen, quota):
            avail = end[c] - ptr[c]
            take = int(min(qc, avail))
            if take > 0:
                picks.append(queue[ptr[c]:ptr[c] + take])
                ptr[c] += take
            leftover += int(qc) - take
        if leftover > 0:
            is_ch = np.zeros(len(act), dtype=bool)
            is_ch[sel] = True
            for p in np.argsort(-sc):
                if is_ch[p]:
                    continue
                c = act[p]
                avail = end[c] - ptr[c]
                if avail <= 0:
                    continue
                take = int(min(leftover, avail))
                picks.append(queue[ptr[c]:ptr[c] + take])
                ptr[c] += take
                leftover -= take
                if leftover == 0:
                    break
        if not picks:
            break
        picked = np.concatenate(picks)
        yp = y[picked].astype(np.float64)
        gp = grp[picked]
        np.add.at(alpha, gp, yp)
        np.add.at(beta, gp, 1.0 - yp)
        cum += int(yp.sum())
        queried += len(picked)
        Q.append(queried)
        C.append(cum)
        if lookup is not None:
            w = lookup[picked]
            found[w[w >= 0]] = queried
        if return_order:
            order_log.append(picked)
        t += 1

    out = {"queried": np.asarray(Q, dtype=np.int64), "cum": np.asarray(C, dtype=np.int64),
           "found_at": found, "seed_idx": seed_idx}
    if return_order:
        out["order"] = np.concatenate(order_log)
    return out


def run_random(lib, seed, S=5000, B=50, max_queries=None, watch=None, return_order=False):
    """Uniform random sampling without replacement (same seed set as `run` for equal seed)."""
    rng = np.random.default_rng(seed)
    n = lib.n
    perm = rng.permutation(n)
    seed_idx, rest = perm[:S], perm[S:]
    y = lib.labels

    ycum = np.concatenate([[0], np.cumsum(y[rest], dtype=np.int64)])
    steps = np.arange(0, len(rest) + 1, B)
    if steps[-1] != len(rest):
        steps = np.append(steps, len(rest))
    if max_queries is not None:
        steps = steps[S + steps <= max(max_queries, S)]
    Q = S + steps
    C = int(y[seed_idx].sum()) + ycum[steps]

    found = np.zeros(0, dtype=np.int32)
    if watch is not None and len(watch):
        where = np.empty(n, dtype=np.int64)
        where[perm] = np.arange(n)
        pos = where[np.asarray(watch)]
        found = np.where(pos < S, S, S + (np.ceil((pos - S + 1) / B) * B).astype(np.int64))
        found = np.minimum(found, n).astype(np.int32)
        if max_queries is not None:
            found = np.where(found <= Q[-1], found, -1).astype(np.int32)

    out = {"queried": Q.astype(np.int64), "cum": C.astype(np.int64), "found_at": found,
           "seed_idx": seed_idx}
    if return_order:
        out["order"] = perm
    return out
