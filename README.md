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

## Online workflow with localcolabfold on Frontera (`online_ts.py`)

`online_ts.py` is an iterative online Thompson sampling driver designed for
TACC Frontera. It selects peptides from an unlabeled library each round,
writes batch input files, generates and submits SLURM jobs for
localcolabfold, waits for PDB outputs, scores them, updates cluster
posteriors, and checkpoints state so interrupted runs can be resumed.

Two input modes are supported:

- **FASTA mode** (default): ColabFold generates MSAs from scratch for each
  peptide. Use for small runs or when no precomputed MSA is available.
- **A3M mode** (set `msa_template`): Uses a precomputed protein MSA so
  ColabFold skips the MSA search step.


### Config

Copy and edit `data/online_config.json`. The required TS fields are the
same as `data/config.json`, with four additions:

- `seqs_per_job` — sequences bundled into each SLURM job (default 10)
- `max_concurrent_jobs` — max jobs in the queue at once (default 4)
- `msa_template` — path to a precomputed protein MSA A3M file (default `""`,
  i.e., FASTA mode). See [Using precomputed MSAs](#using-precomputed-msas).
- `slurm` — nested dict of SLURM and localcolabfold options

Fill in `slurm.account` with your Frontera allocation before running.

### Using precomputed MSAs

Generating MSAs dominates AF2 runtime. By providing a precomputed
protein MSA you can skip that step and run ColabFold in structure-prediction-only
mode.

**Step 1 — generate `seq.a3m`.**

Run one ColabFold job on the target protein alone (or on a representative
peptide-protein complex). The `strip_peptide_msa.py` utility strips out
paired MSA rows and the peptide's own MSA, keeping only the protein's
unpaired MSA:

```bash
python strip_peptide_msa.py complex_query.a3m seq.a3m --peptide-chain A
```

**Step 2 — point the config at the template.**

In `data/online_config.json`:

```json
"msa_template": "/path/to/seq.a3m"
```

Or pass it at the command line:

```bash
python online_ts.py \
    --config data/online_config.json \
    --library my_library.csv \
    --work_root runs/ts_pilot \
    --msa_template /path/to/seq.a3m \
    --dry_run
```

When `msa_template` is set, `online_ts.py` writes one `.a3m` file per
peptide into a `batch_NNN_msas/` directory and invokes ColabFold on that
directory. The protein MSA rows are taken from `seq.a3m` and the
peptide slot is filled with only the peptide query sequence (no MSA for
the peptide). This approach works for peptides of any length.

### Assumptions about localcolabfold

- `colabfold_batch` is on `$PATH` inside the SLURM job (activate your
  conda environment via `slurm.module_setup` in the config).
- ColabFold writes PDB outputs into the directory passed as its second
  positional argument.
- Output filenames follow the ColabFold convention:
  `{query_id}_rank_001_alphafold2_*.pdb`
- For peptide-protein multimer predictions, set `peptide_chain`,
  `target_chain`, and `key_residues` in the config. Leave `key_residues`
  empty (`""`) to use pLDDT-only scoring (no distance check).

### Dry run (verify setup before submitting)

```bash
python online_ts.py \
    --config data/online_config.json \
    --library my_library.csv \
    --work_root runs/ts_pilot \
    --dry_run
```

This writes input files and SLURM scripts under `runs/ts_pilot/round_seed/`
and saves a checkpoint. No jobs are submitted. Inspect the generated files
and the SLURM scripts before proceeding. In A3M mode, each batch produces a
`batch_NNN_msas/` directory containing per-peptide `.a3m` files.

### Live run

```bash
python online_ts.py \
    --config data/online_config.json \
    --library my_library.csv \
    --work_root runs/ts_pilot \
    --data_out results/data.csv \
    --metrics_out results/af2_metrics.csv \
    --out_prefix results/online_ts
```

### Resume after interruption

```bash
python online_ts.py \
    --config data/online_config.json \
    --library my_library.csv \
    --work_root runs/ts_pilot \
    --data_out results/data.csv \
    --metrics_out results/af2_metrics.csv \
    --out_prefix results/online_ts \
    --resume
```

The workflow reads `runs/ts_pilot/checkpoint.json`, skips completed rounds,
re-checks any jobs that were submitted but not yet scored, and continues
from where it left off. The library file must be the same one used in the
original run (the MD5 hash is checked).

### Override options

```
--seqs_per_job N          sequences per SLURM job
--max_concurrent_jobs N   max jobs in queue at once
--batch_size N            sequences selected per round
--msa_template PATH       precomputed protein MSA A3M (enables A3M mode)
```

### Output files

Same as `replay.py`:

- `data_out` — `name,cluster,label` table
- `metrics_out` — per-query pLDDT, distance, RoG, and pass metrics
- `<out_prefix>_selections.csv`, `_seed.csv`, `_curve.csv`,
  `_summary.json`, `_plot.png`
- `<work_root>/checkpoint.json` — full TS state for restart

## Dependencies

- `numpy`
- `pandas`
- `matplotlib`
- `biopython` (required by `online_ts.py`)
