# Data Harmonization

The public workflow expects the repository-level raw-data snapshot at:

```text
data/raw/archive.zip
```

This snapshot is derived from the public PhysioNet/Computing in Cardiology Challenge 2019 v1.0.0 training data. The authoritative source remains PhysioNet Challenge 2019 v1.0.0; do not treat `archive.zip` as an official PhysioNet filename or direct download URL.

Run:

```bash
python src/data/data_harmonization.py \
  --kaggle_path data/raw/archive.zip \
  --output_dir data/processed
```

The harmonizer reads patient-level PSV files when present, skips aggregate CSV/TSV files in that case, preserves `SourceSet` from training set paths when available, and writes `data/processed/kaggle_harmonized.csv`.
