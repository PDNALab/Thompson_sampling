# Thompson Sampling

This folder contains Thompson sampling workflows for peptide screening with cluster-aware selection.

- `replay.py`: replay workflow that reads precomputed AF2 metrics from `data/dict.csv`, combines them with cluster assignments, and writes `data.csv` plus TS outputs.

## Inputs

### Cluster assignments

Cluster assignments live in `./clusters` as CSV files with columns:

- `name`
- `sequence`
- `cluster`

Current clustering result files include:

- `cd-0.4.csv`, `cd-0.5.csv`, `cd-0.7.csv`, `cd-0.9.csv`
- `easy-cluster-0.4.csv`, `easy-cluster-0.5.csv`, `easy-cluster-0.7.csv`, `easy-cluster-0.9.csv`
- `easy-linclust-0.4.csv`, `easy-linclust-0.5.csv`, `easy-linclust-0.7.csv`, `easy-linclust-0.9.csv`

### Precomputed AF2 results

`data/dict.csv` stores AF2 metrics with columns:

- `name`
- `sequence`
- `plddt_0`, `dist_0`, `rog_0`
- `plddt_1`, `dist_1`, `rog_1`
- `plddt_2`, `dist_2`, `rog_2`
- `plddt_3`, `dist_3`, `rog_3`
- `plddt_4`, `dist_4`, `rog_4`

For the offline replay workflow, these metrics are converted into a binary binder label and then written to `data.csv` in the standard format:

- `name`
- `cluster`
- `label`

## Example replay run

The `./example` folder contains the outputs from:

```bash
python replay.py --config data/config.json --library clusters/cd-0.5.csv --dict_path data/dict.csv --data_out example/data.csv --metrics_out example/af2_metrics.csv --out_prefix example/replay
```

This produces:

- `example/data.csv`
- `example/af2_metrics.csv`
- `example/replay_seed.csv`
- `example/replay_selections.csv`
- `example/replay_curve.csv`
- `example/replay_summary.json`
- `example/replay_plot.png`

## Dependencies

- `numpy`
- `pandas`
- `matplotlib`
