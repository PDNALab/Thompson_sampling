"""A bandit picks the clusters, a default random forest ranks the peptides inside them.

Per round (batch B = 50, k = 50 clusters, equal allocation -> one peptide per chosen cluster):
  1. the bandit (TS / UCB / eps-greedy) scores every non-exhausted CD-HIT 0.5 cluster from its Beta posterior and chooses the top k
     -- exactly the settings of results/1-policies/figures/1-baselines.png (prior Beta(0.01, 0.99), UCB c = 2.0, eps-greedy eps = 0.02);
  2. a RandomForestClassifier() with sklearn defaults is refit on EVERY labelled peptide (470 features, as in rf_greedy.py);
  3. inside each chosen cluster the unqueried peptide with the highest forest P(binder) is queried (ties broken at random);
  4. the cluster posteriors are updated with the labels (alpha += hit, beta += miss), the forest is refit, repeat.
The forest never influences which clusters are chosen; the bandit never influences which peptide is chosen in a cluster.
The only difference from the arm policies of 1-baselines is step 3 (they pick a random unqueried peptide in the chosen cluster).

--within random replaces step 3 by a random pick (no forest, fast): this is the control that must reproduce the arm policies of
results/1-policies, and it tests this loop against the paper's simulator.

Output (OUT/bandit_<policy>_<within>_seed<S>.npz): order (queried library rows, seed set first), hits (binders per round), fit_s, score_s,
  top_p (mean forest P of the picks; nan for --within random), n_cand (peptides scored per round).
Checkpoints every 100 rounds (OUT/*.ckpt, RNG state included): rerun the same command to resume.
--S 0 = cold start (no seed set): while the labelled set lacks a binder or a non-binder no forest can be fit, so the pick inside the chosen clusters is random
(the paper's arm behaviour, `phase` = 0); afterwards the forest ranks as usual (`phase` = 1). Output name gets _S<S> when S != 5000 and _<cond> when --cond != full (feature-vector ablation of the forest, same conditions as rf_ablation.py).

Usage: python bandit_rf.py --policy ts|ucb|eps_greedy --seed 101 [--within rf|random] [--n_jobs 4] [--max_queries 70000] [--out DIR]
"""
import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "results" / "scripts"))
from tsbench import Library, make_policy        # noqa: E402  (the paper's final arm code)
from tsbench.policies import State              # noqa: E402
from tsbench.sim import _alloc                  # noqa: E402
from rf_greedy import DATA, load_blocks, load_features   # noqa: E402

ALPHA0, BETA0, K = 0.01, 0.99, 50
POLICY_KW = {"ts": {}, "ucb": {"c": 2.0}, "eps_greedy": {"eps": 0.02}}


def load_library():
    lib_df = pd.read_csv(REPO / "results" / "1-policies" / "data" / "library_cdhit_0.5.csv")
    return Library(lib_df.label.values, lib_df.cluster.values)


def run(lib, X, policy_name, seed, within="rf", S=5000, B=50, k=K, max_queries=70000, n_jobs=1, ckpt=None, ckpt_every=100, log_every=100):
    from sklearn.ensemble import RandomForestClassifier

    policy = make_policy(policy_name, **POLICY_KW[policy_name])
    rng = np.random.default_rng(seed)
    n, G, y, grp = lib.n, lib.n_groups, lib.labels, lib.groups
    perm = rng.permutation(n)                                   # same first RNG call as tsbench / rf_greedy -> same seed set
    seed_idx = perm[:S]
    members_all = np.argsort(grp, kind="stable")
    size = np.bincount(grp, minlength=G)
    end = np.cumsum(size)
    members = [members_all[e - s:e] for s, e in zip(size, end)]   # members[c] = library rows of cluster c

    avail = np.ones(n, dtype=bool)
    avail[seed_idx] = False
    order = [seed_idx.astype(np.int32)]
    fit_s, score_s, top_p, hits, n_cand, phase = [], [], [], [], [], []
    t = 0
    if ckpt is not None and Path(ckpt).exists():
        c = pickle.load(open(ckpt, "rb"))
        rng.bit_generator.state = c["rng"]
        order, fit_s, score_s, top_p, hits, n_cand, t = c["order"], c["fit_s"], c["score_s"], c["top_p"], c["hits"], c["n_cand"], c["t"]
        phase = c.get("phase", [1] * t)
        avail[np.concatenate(order)] = False
        print(f"  resumed from {ckpt} at round {t}", flush=True)
    labelled = np.concatenate(order)
    yl, gl = y[labelled].astype(np.float64), grp[labelled]
    alpha = ALPHA0 + np.bincount(gl, weights=yl, minlength=G)         # posterior of every cluster from the labels so far
    beta = BETA0 + np.bincount(gl, weights=1 - yl, minlength=G)
    rem = size - np.bincount(gl, minlength=G)                          # unqueried peptides per cluster
    queried, cum = len(labelled), int(y[labelled].sum())
    print(f"{policy_name} + {within}, seed {seed}: seed set {S} ({int(y[seed_idx].sum())} binders); start at {queried} queries, {cum} binders", flush=True)

    last_ck = t
    while queried < max_queries and rem.sum() > 0:
        if ckpt is not None and t % ckpt_every == 0 and t > last_ck:
            tmp = Path(str(ckpt) + ".tmp")
            with open(tmp, "wb") as fh:
                pickle.dump(dict(rng=rng.bit_generator.state, order=order, fit_s=fit_s, score_s=score_s, top_p=top_p, hits=hits, n_cand=n_cand, phase=phase, t=t), fh)
            tmp.replace(ckpt)
            last_ck = t
        act = np.flatnonzero(rem > 0)
        b = int(min(B, rem.sum()))
        m = min(k, len(act), b)
        st = State(alpha[act], beta[act], act, rng, t)
        sc = policy.scores(st)
        sel = policy.choose(sc, m, st)
        chosen = act[sel]
        quota = _alloc(sc[sel], b, "equal")

        rf = None
        ready = within == "rf" and 0 < cum < queried        # a forest needs both classes; until then the pick inside a cluster is random (= the paper's arms)
        phase.append(int(ready))
        if within == "rf" and not ready:
            fit_s.append(0.0)
        if ready:
            t0 = time.time()
            rf = RandomForestClassifier(random_state=seed * 100000 + t, n_jobs=n_jobs).fit(X[labelled], y[labelled])
            fit_s.append(time.time() - t0)
        t0 = time.time()
        cache = {}

        def ranked(c):
            """unqueried members of cluster c, best first (forest P(binder), random tie-break); (rows, P)."""
            if c not in cache:
                cand = members[c][avail[members[c]]]
                p = rf.predict_proba(X[cand])[:, 1] if rf is not None else np.zeros(len(cand))
                o = np.argsort(-(p + 1e-7 * rng.random(len(cand))), kind="stable")
                cache[c] = (cand[o], p[o])
            return cache[c]

        picks, ps, leftover = [], [], 0
        for c, qc in zip(chosen, quota):
            rows, p = ranked(c)
            take = int(min(qc, len(rows)))
            picks.append(rows[:take]); ps.append(p[:take])
            leftover += int(qc) - take
        if leftover > 0:                                               # only possible when < k clusters are left
            is_ch = np.zeros(len(act), dtype=bool)
            is_ch[sel] = True
            for pos in np.argsort(-sc):
                if is_ch[pos]:
                    continue
                rows, p = ranked(act[pos])
                take = int(min(leftover, len(rows)))
                if take > 0:
                    picks.append(rows[:take]); ps.append(p[:take]); leftover -= take
                if leftover == 0:
                    break
        picked = np.concatenate(picks)
        score_s.append(time.time() - t0)
        n_cand.append(sum(len(v[0]) for v in cache.values()))
        top_p.append(float(np.concatenate(ps).mean()) if ready else float("nan"))

        avail[picked] = False
        yp = y[picked].astype(np.float64)
        np.add.at(alpha, grp[picked], yp)
        np.add.at(beta, grp[picked], 1.0 - yp)
        np.add.at(rem, grp[picked], -1)
        labelled = np.concatenate([labelled, picked])
        order.append(picked.astype(np.int32))
        hits.append(int(yp.sum()))
        queried += len(picked); cum += hits[-1]; t += 1
        if t % log_every == 0:
            print(f"  round {t} queried {queried} binders {cum} ({cum / lib.total_binders:.1%}) hit rate of the last {log_every} rounds "
                  f"{np.sum(hits[-log_every:]) / (B * log_every):.3f} fit+score {sum(fit_s) + sum(score_s):.0f}s", flush=True)
    return dict(order=np.concatenate(order), hits=np.array(hits, np.int8), fit_s=np.array(fit_s, np.float32), score_s=np.array(score_s, np.float32),
                top_p=np.array(top_p, np.float32), n_cand=np.array(n_cand, np.int32), phase=np.array(phase, np.int8))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True, choices=sorted(POLICY_KW))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--within", default="rf", choices=["rf", "random"])
    ap.add_argument("--S", type=int, default=5000)                    # size of the random seed set; 0 = cold start
    ap.add_argument("--cond", default="full", help="feature-vector ablation of the forest: full | minus_<B> | only_<B>, B in AAC DPC PHYS NTERM CTERM (see rf_ablation.py)")
    ap.add_argument("--n_jobs", type=int, default=1)
    ap.add_argument("--max_queries", type=int, default=70000)
    ap.add_argument("--out", type=Path, default=DATA)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    lib = load_library()
    X, y = load_features()
    assert (y == lib.labels).all()
    if a.cond != "full":
        from rf_ablation import columns
        z = load_blocks()
        X = np.hstack([z[b] for b in columns(a.cond)]).astype(np.float32)
        print(f"feature ablation {a.cond}: {X.shape[1]} features", flush=True)
    tag = ("" if a.S == 5000 else f"_S{a.S}") + ("" if a.cond == "full" else f"_{a.cond}")
    f = a.out / f"bandit_{a.policy}_{a.within}{tag}_seed{a.seed}.npz"
    res = run(lib, X, a.policy, a.seed, within=a.within, S=a.S, max_queries=a.max_queries, n_jobs=a.n_jobs, ckpt=f.with_suffix(".ckpt"))
    np.savez_compressed(f, **res, seed=a.seed, policy=a.policy, within=a.within, cond=a.cond)
    print("saved", f)
