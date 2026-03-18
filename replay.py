import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from TS import (
    build_cluster_index,
    initialize_rng,
    label_from_passes,
    load_config,
    save_outputs,
    select_batch,
)


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Offline Thompson Sampling workflow that reads precomputed AF2 metrics "
            "from dict.csv and updates <cluster>.csv after each batch."
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
        "--library",
        type=Path,
        required=True,
        help="CSV with columns: name, sequence, cluster.",
    )
    p.add_argument(
        "--dict_path",
        type=Path,
        required=True,
        help=(
            "CSV with precomputed AF2 metrics in the format: "
            "name,sequence,plddt_0,dist_0,rog_0,...,plddt_4,dist_4,rog_4."
        ),
    )
    p.add_argument(
        "--data_out",
        type=Path,
        default=Path("data.csv"),
        help="Aggregated queried labels in the same format as example/data.csv.",
    )
    p.add_argument(
        "--metrics_out",
        type=Path,
        default=Path("af2_metrics.csv"),
        help="Detailed per-query AF2 metrics for queried peptides.",
    )
    p.add_argument(
        "--out_prefix",
        type=Path,
        default=Path("offline_ts"),
        help="Prefix for TS logs, curve, summary, and plot outputs.",
    )
    return p.parse_args()


def validate_library(df):
    required = ["name", "sequence", "cluster"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            f"Library is missing required columns: {', '.join(missing)}"
        )
    if df["name"].duplicated().any():
        duplicated = df.loc[df["name"].duplicated(), "name"].tolist()[:5]
        raise ValueError(
            f"Library contains duplicate names in 'name': {duplicated}"
        )


def validate_dict(df, args):
    required = ["name", "sequence"]
    metric_cols = []
    for i in range(args.models_per_query):
        metric_cols.extend([f"plddt_{i}", f"dist_{i}", f"rog_{i}"])
    missing = [col for col in required + metric_cols if col not in df.columns]
    if missing:
        raise ValueError(
            f"dict.csv is missing required columns: {', '.join(missing)}"
        )


def compute_model_passes(df, args):
    for i in range(args.models_per_query):
        plddt_col = f"plddt_{i}"
        dist_col = f"dist_{i}"
        pass_col = f"pass_{i}"
        status_col = f"status_{i}"
        rg_col = f"rog_{i}"

        df[pass_col] = (
            df[plddt_col].ge(args.plddt_threshold)
            & df[dist_col].lt(args.dist_threshold)
            & df[dist_col].notna()
        ).astype(int)
        df[status_col] = np.where(
            df[plddt_col].notna() & df[dist_col].notna(),
            "OK",
            "MISSING",
        )

        if rg_col not in df.columns:
            raise ValueError(
                f"dict.csv is missing required AF2 column for model {i}: {rg_col}"
            )

    pass_cols = [f"pass_{i}" for i in range(args.models_per_query)]
    df["label"] = df[pass_cols].apply(
        lambda row: label_from_passes(row.astype(int).tolist(), args.binder_rule),
        axis=1,
    )
    df["models_found"] = (
        df[[f"plddt_{i}" for i in range(args.models_per_query)]]
        .notna()
        .sum(axis=1)
    )
    return df


def merge_library_and_dict(library_df, dict_df):
    merged = pd.merge(
        library_df,
        dict_df,
        left_on="id",
        right_on="name",
        how="inner",
        suffixes=("", "_dict"),
    )

    if merged.empty:
        merged = pd.merge(
            library_df,
            dict_df,
            left_on="fasta_sequence",
            right_on="sequence",
            how="inner",
            suffixes=("", "_dict"),
        )
        if merged.empty:
            raise ValueError(
                "Could not match seqs.csv to dict.csv using id/name or sequence."
            )

    if merged["id"].duplicated().any():
        duplicated = merged.loc[merged["id"].duplicated(), "id"].tolist()[:5]
        raise ValueError(
            f"Multiple dict.csv rows matched the same library id: {duplicated}"
        )

    missing_ids = sorted(set(library_df["id"]) - set(merged["id"]))
    if missing_ids:
        preview = ", ".join(missing_ids[:5])
        raise ValueError(
            f"Missing AF2 metrics for {len(missing_ids)} library rows, including: {preview}"
        )

    return merged


def build_metric_record(row, args):
    record = {
        "id": row["id"],
        "cluster": row["cluster"],
        "name": row["name"],
        "sequence": row["sequence"],
        "models_found": int(row["models_found"]),
        "label": int(row["label"]),
    }
    for i in range(args.models_per_query):
        record[f"plddt_{i}"] = row[f"plddt_{i}"]
        record[f"dist_{i}"] = row[f"dist_{i}"]
        record[f"rog_{i}"] = row[f"rog_{i}"]
        record[f"pass_{i}"] = int(row[f"pass_{i}"])
        record[f"status_{i}"] = row[f"status_{i}"]
    return record


def main():
    args = parse_args()
    config = load_config(
        args.config,
        defaults={
            "alloc": "proportional",
            "random_state": 0,
            "models_per_query": 5,
            "plddt_threshold": 70.0,
            "dist_threshold": 20.0,
            "binder_rule": "four_of_five",
        },
        allowed_binder_rules={"any", "top1", "majority", "four_of_five"},
    )
    for key, value in config.items():
        setattr(args, key, value)

    library = pd.read_csv(args.library)
    dict_df = pd.read_csv(args.dict_path)

    validate_library(library)
    validate_dict(dict_df, args)

    library = library.rename(columns={"name": "id", "sequence": "fasta_sequence"})
    merged = merge_library_and_dict(library, dict_df)
    merged = compute_model_passes(merged, args).reset_index(drop=True)

    clusters, cluster_to_idx = build_cluster_index(merged)
    rng_seed, rng = initialize_rng(args.random_state)

    n = len(merged)
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
        row = merged.loc[int(idx)]
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
        metric_row = build_metric_record(row, args)
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
            row = merged.loc[int(idx)]
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
            metric_row = build_metric_record(row, args)
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
