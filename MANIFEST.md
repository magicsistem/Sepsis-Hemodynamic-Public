# Manifest

This manifest describes the intended public GitHub/main layout. The repository is organized around reproducible code, clean execution entrypoints, compact final outputs, and provenance inventories.

## Active Source Code

| path | role | notes |
|---|---|---|
| `src/data/data_harmonization.py` | Data harmonization | Reads `data/raw/archive.zip`, preserves `SourceSet`, and writes `data/processed/kaggle_harmonized.csv`. |
| `src/training/train_zabihi_cudf.py` | Training | Runs baseline or enhanced hemodynamic training with patient-grouped cross-validation. |
| `src/reporting/build_statistics.py` | Statistics | Builds compact statistics, validation checks, and inference tables from final baseline/enhanced artifacts. |
| `src/reporting/build_final_outputs.py` | Final outputs | Builds compact final tables and figures from statistics, robustness, and OOF artifacts. |
| `scripts/harmonize_data.py` | Wrapper | Runs harmonization with `data/raw/archive.zip` as the public default. |
| `scripts/train_baseline.py` | Wrapper | Runs baseline training with `results/baseline` as the public default. |
| `scripts/train_enhanced.py` | Wrapper | Runs enhanced training with `results/enhanced` as the public default. |
| `scripts/build_statistics.py` | Wrapper | Runs statistics generation with `results/statistics` as the public default. |
| `scripts/build_internal_robustness.py` | Robustness | Builds internal robustness outputs from final result artifacts. |
| `scripts/build_final_outputs.py` | Wrapper | Runs final table and figure generation. |
| `scripts/audit_methodology.py` | Audit | Checks no-leakage and methodology boundaries from code and compact artifacts. |
| `scripts/validate_public_manifests.py` | Validation | Verifies public manifest/checksum path, size, and SHA-256 records. |

## Active SLURM Jobs

| path | role |
|---|---|
| `jobs/run_baseline.slurm` | Full baseline training run. |
| `jobs/run_enhanced.slurm` | Full enhanced training run. |
| `jobs/run_baseline_features.slurm` | Baseline feature-cache stage. |
| `jobs/run_baseline_train.slurm` | Baseline train/post-processing stage from validated cache. |
| `jobs/run_enhanced_features.slurm` | Enhanced feature-cache stage. |
| `jobs/run_enhanced_train.slurm` | Enhanced train/post-processing stage from validated cache. |
| `jobs/run_statistics.slurm` | Statistics generation. |
| `jobs/run_methodology_audit.slurm` | No-leakage/methodology audit. |
| `jobs/run_internal_robustness.slurm` | Internal robustness generation. |
| `jobs/run_final_outputs.slurm` | Final public table/figure generation. |

## Public Compact Results

| path | role |
|---|---|
| `results/statistics/` | Compact metric tables, statistical tests, validation checks, and checksums. |
| `results/internal_robustness/` | Internal SourceSet, subgroup, threshold, and calibration robustness summaries. |
| `results/final_tables/` | Final compact CSV tables and manifests. |
| `results/final_figures/` | Final compact PNG figures and manifests. |

## Documentation And Inventories

| path | role |
|---|---|
| `README.md` | Public workflow overview and exact reproduction commands. |
| `docs/reproducibility.md` | Reproducibility workflow and command details. |
| `docs/results_policy.md` | Policy for compact public outputs and local-only artifacts. |
| `docs/final_audit_summary.md` | Consolidated public audit summary. |
| `docs/artifact_inventory.md` | Public inventory guide. |
| `docs/data_harmonization.md` | Data harmonization notes for the public workflow. |
| `external_artifacts/inventory/` | Public provenance, inventory, and checksum files. |

## Local-Only Material

The local manuscript tree, manuscript-only assets, historical audit notes, and detailed HPC/local inventories may remain on disk for traceability, but they are not public workflow dependencies.
