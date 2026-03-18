import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_CONFIG_KEYS = [
    "seed_size",
    "batch_size",
    "rounds",
    "alpha0",
    "beta0",
    "topk_clusters",
]


def load_config(config_path, defaults=None, allowed_binder_rules=None):
    with Path(config_path).open() as handle:
        config = json.load(handle)

    missing = [key for key in REQUIRED_CONFIG_KEYS if key not in config]
    if missing:
        raise ValueError(
            f"Config file is missing required keys: {', '.join(missing)}"
        )

    merged = {**(defaults or {}), **config}

    if merged.get("alloc", "proportional") not in {"proportional", "equal"}:
        raise ValueError("config.alloc must be 'proportional' or 'equal'")

    if allowed_binder_rules is not None:
        binder_rule = merged.get("binder_rule")
        if binder_rule not in allowed_binder_rules:
            allowed = ", ".join(f"'{rule}'" for rule in allowed_binder_rules)
            raise ValueError(f"config.binder_rule must be one of: {allowed}")

    return merged


def initialize_rng(random_state):
    seed_val = (
        int(np.random.randint(0, high=999999, size=1, dtype=int)[0])
        if random_state == 0
        else random_state
    )
    return seed_val, np.random.default_rng(seed_val)


def alloc_quota(chosen, sampled_thetas, total_batch, mode):
    thetas = np.array([sampled_thetas[c] for c in chosen], dtype=float)
    if mode == "equal":
        quota = np.full(len(chosen), total_batch // len(chosen), dtype=int)
    else:
        weights = thetas.clip(min=1e-9)
        weights = weights / weights.sum()
        quota = np.floor(weights * total_batch).astype(int)

    for i in range(len(quota)):
        if quota[i] == 0 and quota.sum() < total_batch:
            quota[i] += 1

    while quota.sum() < total_batch:
        quota[int(np.argmax(thetas))] += 1
    while quota.sum() > total_batch:
        quota[int(np.argmin(thetas))] -= 1

    return dict(zip(chosen, quota))


def label_from_passes(passes, rule):
    if not passes:
        return 0
    if rule == "top1":
        return int(passes[0])
    if rule == "majority":
        required = math.ceil(len(passes) / 2)
        return int(sum(passes) >= required)
    if rule == "four_of_five":
        return int(sum(passes) >= 4)
    return int(any(passes))


def ensure_parent(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def build_cluster_index(df, cluster_col="cluster"):
    clusters = df[cluster_col].astype(str).values
    cluster_to_idx = {}
    for i, cluster in enumerate(clusters):
        cluster_to_idx.setdefault(cluster, []).append(i)
    return clusters, {k: np.asarray(v, dtype=int) for k, v in cluster_to_idx.items()}


def select_batch(seen, cluster_to_idx, alpha, beta, args, rng):
    sampled = {cluster: rng.beta(alpha[cluster], beta[cluster]) for cluster in cluster_to_idx}
    remaining = {
        cluster: int((~seen[idxs]).sum()) for cluster, idxs in cluster_to_idx.items()
    }

    ordered = [
        cluster
        for cluster in sorted(sampled, key=sampled.get, reverse=True)
        if remaining[cluster] > 0
    ]
    if not ordered:
        return []

    chosen = ordered[: min(args.topk_clusters, len(ordered))]
    quota = alloc_quota(chosen, sampled, args.batch_size, args.alloc)

    picked = []
    leftover = 0
    for cluster in chosen:
        candidates = cluster_to_idx[cluster][~seen[cluster_to_idx[cluster]]]
        if len(candidates) == 0 or quota[cluster] == 0:
            leftover += quota.get(cluster, 0)
            continue
        rng.shuffle(candidates)
        take_n = min(quota[cluster], len(candidates))
        picked.extend(candidates[:take_n].tolist())
        if take_n < quota[cluster]:
            leftover += quota[cluster] - take_n

    idx_next = len(chosen)
    while leftover > 0 and idx_next < len(ordered):
        cluster = ordered[idx_next]
        idx_next += 1
        candidates = cluster_to_idx[cluster][~seen[cluster_to_idx[cluster]]]
        if len(candidates) == 0:
            continue
        rng.shuffle(candidates)
        take_n = min(leftover, len(candidates))
        picked.extend(candidates[:take_n].tolist())
        leftover -= take_n

    idx_next = 0
    while len(picked) < args.batch_size and idx_next < len(ordered):
        cluster = ordered[idx_next]
        idx_next += 1
        candidates = cluster_to_idx[cluster][~seen[cluster_to_idx[cluster]]]
        if len(candidates) == 0:
            continue
        picked.append(int(candidates[int(rng.integers(low=0, high=len(candidates)))]))

    return list(dict.fromkeys(picked))[: args.batch_size]


def save_outputs(
    result_rows,
    metric_rows,
    curve_rows,
    data_out,
    metrics_out,
    out_prefix,
    seed_val,
    data_id_column="id",
):
    data_df = pd.DataFrame(result_rows)
    metrics_df = pd.DataFrame(metric_rows)
    curve_df = pd.DataFrame(curve_rows, columns=["queried", "cum_yes"])
    out_prefix = Path(out_prefix)

    ensure_parent(data_out)
    ensure_parent(metrics_out)
    ensure_parent(Path(f"{out_prefix}_curve.csv"))

    export_df = data_df.loc[:, ["id", "cluster", "label"]].rename(
        columns={"id": data_id_column}
    )
    export_df.to_csv(data_out, index=False)
    metrics_df.to_csv(metrics_out, index=False)
    data_df.to_csv(f"{out_prefix}_selections.csv", index=False)
    data_df[data_df["round"] == -1].to_csv(f"{out_prefix}_seed.csv", index=False)
    curve_df.to_csv(f"{out_prefix}_curve.csv", index=False)

    summary = {
        "used_seed": seed_val,
        "queried_count": int(len(data_df)),
        "binder_count": int(data_df["label"].sum()) if not data_df.empty else 0,
        "observed_yes_rate": (
            float(data_df["label"].mean()) if not data_df.empty else 0.0
        ),
    }
    with Path(f"{out_prefix}_summary.json").open("w", newline="\n") as handle:
        json.dump(summary, handle, indent=2)

    try:
        import matplotlib.pyplot as plt

        plt.figure()
        plt.plot(curve_df["queried"], curve_df["cum_yes"], label="TS")
        plt.xlabel("Number of peptides queried")
        plt.ylabel("Cumulative binders discovered")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{out_prefix}_plot.png")
        plt.close()
    except ImportError:
        pass
