import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from TS import (
    build_cluster_index,
    initialize_rng,
    load_config,
    save_outputs,
    select_batch,
)


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Offline Thompson Sampling workflow for synthetic replay datasets with "
            "preassigned labels. The input CSV must contain name, cluster, and label."
        )
    )
    p.add_argument(
        "--config",
        type=Path,
        required=True,
        help=(
            "JSON file containing TS parameters such as seed_size, batch_size, "
            "rounds, alpha0, beta0, and topk_clusters."
        ),
    )
    p.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Synthetic CSV with columns: name, cluster, label.",
    )
    p.add_argument(
        "--data_out",
        type=Path,
        default=Path("data.csv"),
        help="Aggregated queried labels in the same format as tests/*.csv.",
    )
    p.add_argument(
        "--metrics_out",
        type=Path,
        default=Path("synthetic_metrics.csv"),
        help="Detailed per-query selection log for the queried peptides.",
    )
    p.add_argument(
        "--out_prefix",
        type=Path,
        default=Path("test_replay"),
        help="Prefix for TS logs, curve, summary, and plot outputs.",
    )
    return p.parse_args()


def validate_dataset(df):
    required = ["name", "cluster", "label"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            f"Dataset is missing required columns: {', '.join(missing)}"
        )
    if df["name"].duplicated().any():
        duplicated = df.loc[df["name"].duplicated(), "name"].tolist()[:5]
        raise ValueError(
            f"Dataset contains duplicate names in 'name': {duplicated}"
        )

    unique_labels = set(pd.to_numeric(df["label"], errors="coerce").dropna().astype(int))
    if not unique_labels.issubset({0, 1}) or len(unique_labels) == 0:
        raise ValueError("Dataset 'label' column must contain only binary values 0 and 1.")


def build_metric_record(row):
    return {
        "id": row["id"],
        "cluster": row["cluster"],
        "label": int(row["label"]),
    }


def main():
    args = parse_args()
    config = load_config(
        args.config,
        defaults={
            "alloc": "proportional",
            "random_state": 0,
        },
    )
    for key, value in config.items():
        setattr(args, key, value)

    dataset = pd.read_csv(args.dataset)
    validate_dataset(dataset)

    dataset = dataset.rename(columns={"name": "id"}).copy()
    dataset["cluster"] = dataset["cluster"].astype(str)
    dataset["label"] = pd.to_numeric(dataset["label"], errors="raise").astype(int)
    dataset = dataset.reset_index(drop=True)

    clusters, cluster_to_idx = build_cluster_index(dataset)
    rng_seed, rng = initialize_rng(args.random_state)

    n = len(dataset)
    seen = np.zeros(n, dtype=bool)
    alpha = {cluster: args.alpha0 for cluster in cluster_to_idx}
    beta = {cluster: args.beta0 for cluster in cluster_to_idx}

    result_rows = []
    metric_rows = []
    curve_rows = [(0, 0)]
    queried_count = 0
    binder_count = 0

    seed_indices = np.arange(n)
    rng.shuffle(seed_indices)
    seed_indices = seed_indices[: min(args.seed_size, n)]

    for idx in seed_indices:
        row = dataset.loc[int(idx)]
        seen[int(idx)] = True
        cluster = clusters[int(idx)]
        alpha[cluster] += int(row["label"])
        beta[cluster] += 1 - int(row["label"])
        queried_count += 1
        binder_count += int(row["label"])
        curve_rows.append((queried_count, binder_count))

        result_rows.append(
            {
                "round": -1,
                "id": row["id"],
                "cluster": row["cluster"],
                "label": int(row["label"]),
            }
        )
        metric_row = build_metric_record(row)
        metric_row["round"] = -1
        metric_rows.append(metric_row)

    save_outputs(
        result_rows,
        metric_rows,
        curve_rows,
        args.data_out,
        args.metrics_out,
        args.out_prefix,
        rng_seed,
        data_id_column="name",
    )

    for round_index in range(args.rounds):
        picked = select_batch(seen, cluster_to_idx, alpha, beta, args, rng)
        if not picked:
            print(f"[round {round_index}] All clusters exhausted. Ending.")
            break

        for idx in picked:
            row = dataset.loc[int(idx)]
            seen[int(idx)] = True
            cluster = clusters[int(idx)]
            alpha[cluster] += int(row["label"])
            beta[cluster] += 1 - int(row["label"])
            queried_count += 1
            binder_count += int(row["label"])
            curve_rows.append((queried_count, binder_count))

            result_rows.append(
                {
                    "round": round_index,
                    "id": row["id"],
                    "cluster": row["cluster"],
                    "label": int(row["label"]),
                }
            )
            metric_row = build_metric_record(row)
            metric_row["round"] = round_index
            metric_rows.append(metric_row)

        save_outputs(
            result_rows,
            metric_rows,
            curve_rows,
            args.data_out,
            args.metrics_out,
            args.out_prefix,
            rng_seed,
            data_id_column="name",
        )
        print(
            f"[round {round_index}] queried={queried_count} binders={binder_count} "
            f"observed_yes_rate={binder_count / max(queried_count, 1):.4f}"
        )

    print(f"Finished. Wrote {args.data_out} and {args.metrics_out}.")


if __name__ == "__main__":
    main()
