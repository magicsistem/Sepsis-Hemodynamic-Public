# Results Policy

The public repository should contain reproducible code, clean execution entrypoints, technical documentation, compact final outputs, and inventories/checksums.

## Public Outputs

The public result directories are:

- `results/statistics/`
- `results/internal_robustness/`
- `results/final_tables/`
- `results/final_figures/`

These directories should contain compact CSV, JSON, Markdown, and PNG files that can be regenerated from the documented workflow.

## Local-Only Material

The manuscript, manuscript-only assets, raw data, harmonized data, feature caches, trained models, OOF predictions, checkpoints, extensive logs, failed runs, and local/HPC scratch artifacts are not public workflow dependencies. If they are referenced for provenance, record them in an inventory with size, checksum, policy, and regeneration notes.

## Regeneration

The public flow starts from `data/raw/archive.zip`, a repository-level reproducibility snapshot derived from the public PhysioNet/Computing in Cardiology Challenge 2019 v1.0.0 training data. The authoritative source remains PhysioNet Challenge 2019 v1.0.0; `archive.zip` is not an official PhysioNet filename or direct download URL. The workflow creates `data/processed/kaggle_harmonized.csv`, runs baseline and enhanced training, builds statistics, builds internal robustness summaries, and writes final compact tables and figures.

LaTeX compilation is a local manuscript smoke test only. It is not a public repository acceptance criterion.
