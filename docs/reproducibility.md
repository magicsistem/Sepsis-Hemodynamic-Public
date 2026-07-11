# Reproducibility

This document records the public command sequence for reproducing the compact outputs used by the project workflow.

## Input

The public workflow starts from the repository-level raw-data snapshot:

```text
data/raw/archive.zip
```

This snapshot is derived from the public PhysioNet/Computing in Cardiology Challenge 2019 v1.0.0 training data. It is not an official PhysioNet filename or a direct PhysioNet download link. The authoritative source remains PhysioNet Challenge 2019 v1.0.0, accessible through the official project file tree and documented terminal/AWS download commands:

```bash
wget -r -N -c -np https://physionet.org/files/challenge-2019/1.0.0/
```

```bash
aws s3 sync --no-sign-request s3://physionet-open/challenge-2019/1.0.0/ DESTINATION
```

## Commands

```bash
python src/data/data_harmonization.py \
  --kaggle_path data/raw/archive.zip \
  --output_dir data/processed
```

```bash
python src/training/train_zabihi_cudf.py \
  --data_dir data/processed \
  --output_dir results/baseline \
  --n_jobs 64 \
  --n_gpus 2 \
  --stage all
```

```bash
python src/training/train_zabihi_cudf.py \
  --data_dir data/processed \
  --output_dir results/enhanced \
  --n_jobs 64 \
  --n_gpus 2 \
  --use_hemo \
  --stage all
```

```bash
python scripts/build_statistics.py \
  --baseline-dir results/baseline \
  --enhanced-dir results/enhanced \
  --output-dir results/statistics \
  --n_bootstrap 2000 \
  --seed 42
```

```bash
python scripts/build_internal_robustness.py \
  --baseline-dir results/baseline \
  --enhanced-dir results/enhanced \
  --statistics-dir results/statistics \
  --harmonized data/processed/kaggle_harmonized.csv \
  --output-dir results/internal_robustness \
  --inventory-dir external_artifacts/inventory
```

```bash
python scripts/build_final_outputs.py \
  --statistics-dir results/statistics \
  --internal-robustness-dir results/internal_robustness \
  --baseline-dir results/baseline \
  --enhanced-dir results/enhanced \
  --tables-dir results/final_tables \
  --figures-dir results/final_figures
```

## Outputs

- `results/statistics/`: metrics, inference tables, validation checks, and statistics manifest.
- `results/internal_robustness/`: internal robustness summaries by source, subgroup, threshold, and calibration.
- `results/final_tables/`: final compact CSV tables.
- `results/final_figures/`: final compact PNG figures.

Utility Score is threshold-dependent and should not be described as a uniform operational improvement. All reported performance is from patient-grouped internal cross-validation on the public PhysioNet/CinC 2019 training split.
