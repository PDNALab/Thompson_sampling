"""Run one selection policy for a range of seeds on one cluster partition of a labeled peptide library and save compact discovery curves.

Policies (tsbench/policies.py): ts (Thompson sampling), ucb, eps_greedy, greedy, cluster_uniform (random cluster, then random member; no learning) and random (uniform sampling of peptides).
All arm policies share the same batch loop (tsbench/sim.py): a random seed set of S peptides is labeled first, then each round the policy scores every non-exhausted cluster (Beta posterior with prior
Beta(alpha0, beta0) updated by the labels revealed so far), the top-k clusters are chosen, the batch of B queries is allocated over them ('equal' or 'proportional') and random members are queried.

Example (the BRD3 library, CD-HIT 0.5 clusters; the policies of the paper figure):
  python run_experiment.py --data ../1-policies/data/library_cdhit_0.5.csv --method ucb --c 2.0 --k 50 --alloc equal --alpha0 0.01 --beta0 0.99 \
      --seeds 101-200 --out ../1-policies/data/curves_sequence/ucb.npz

Input CSV: id, cluster, label (1 = binder). Output .npz: grid (queried peptides, S, S+B, ...), curves (runs x len(grid), cumulative binders), found_at (runs x watchlist peptides: number of queries at which each peptide of --watchlist was found; empty without --watchlist), seeds, seconds, fit_seconds, config (json).
--partition random replaces the clusters by a random partition with the same cluster-size distribution (a null control; partition id = seed % n_partitions).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))          # tsbench lives next to this script
from tsbench import Library, make_policy, make_grid, run, run_random  # noqa: E402
from tsbench.sim import to_grid  # noqa: E402

PARTITION_SEED0 = 9001


def parse_seeds(text):
    if "-" in text and "," not in text:
        lo, hi = text.split("-")
        return list(range(int(lo), int(hi) + 1))
    return [int(x) for x in text.split(",")]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True, help="CSV with id,cluster,label")
    p.add_argument("--method", required=True, choices=["random", "ts", "greedy", "ucb", "eps_greedy", "cluster_uniform"])
    p.add_argument("--partition", default="sequence", choices=["sequence", "random"])
    p.add_argument("--n_partitions", type=int, default=10)
    p.add_argument("--alloc", default="equal", choices=["equal", "proportional"])
    p.add_argument("--S", type=int, default=5000, help="size of the random seed set (counted as queries)")
    p.add_argument("--B", type=int, default=50, help="batch size (queries per round)")
    p.add_argument("--k", type=int, default=50, help="clusters selected per round")
    p.add_argument("--alpha0", type=float, default=0.2)
    p.add_argument("--beta0", type=float, default=9.8)
    p.add_argument("--c", type=float, default=1.0, help="UCB exploration weight")
    p.add_argument("--eps", type=float, default=0.1, help="epsilon-greedy exploration rate")
    p.add_argument("--tau", type=float, default=None, help="epsilon-greedy decay time constant (rounds)")
    p.add_argument("--seeds", required=True, help="e.g. 101-200 or 1,2,3")
    p.add_argument("--max_queries", type=int, default=None)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--watchlist", type=Path, default=None, help="CSV name,id of peptides (ids of --data) whose discovery time is recorded in found_at")
    p.add_argument("--force", action="store_true")
    a = p.parse_args()

    if a.out.exists() and not a.force:
        print(f"{a.out} exists; use --force to overwrite")
        return

    df = pd.read_csv(a.data)
    lib = Library(df["label"].values, df["cluster"].values)
    S, B, N = a.S, a.B, lib.n
    grid = make_grid(S, N, B)
    if a.max_queries:
        grid = grid[grid <= a.max_queries]
    watch = None
    if a.watchlist is not None:
        pos = pd.Series(np.arange(N), index=df["id"].values)
        watch = pos.loc[pd.read_csv(a.watchlist)["id"].values].values
    policy = None if a.method == "random" else make_policy(a.method, c=a.c, eps=a.eps, tau=a.tau)

    seeds = parse_seeds(a.seeds)
    partitions = {}
    curves, founds, secs = [], [], []
    for s in seeds:
        t0 = time.time()
        if a.method == "random":
            o = run_random(lib, s, S=S, B=B, max_queries=a.max_queries, watch=watch)
        else:
            L = lib
            if a.partition == "random":
                pid = s % a.n_partitions
                if pid not in partitions:
                    partitions[pid] = lib.random_partition(PARTITION_SEED0 + pid)
                L = partitions[pid]
            o = run(L, policy, s, S=S, B=B, k=a.k, alloc=a.alloc, alpha0=a.alpha0, beta0=a.beta0, max_queries=a.max_queries, watch=watch)
        curves.append(to_grid(o["queried"], o["cum"], grid).astype(np.int32))
        founds.append(o["found_at"])
        secs.append(time.time() - t0)

    cfg = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, grid=grid, curves=np.stack(curves), found_at=np.stack(founds), seeds=np.array(seeds),
                        seconds=np.array(secs), fit_seconds=np.zeros(len(seeds)), config=json.dumps(cfg))
    print(f"{a.out}: {len(seeds)} replicates, {np.sum(secs):.0f}s total")


if __name__ == "__main__":
    main()
