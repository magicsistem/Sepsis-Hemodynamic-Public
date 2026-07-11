# Sepsis-Hemodynamic

Sepsis-Hemodynamic contains the code and reproducibility workflow used to train and evaluate an early sepsis prediction model with hemodynamic complexity features for a potential publication.

## Repository Contents

- `src/data/`: data harmonization from the repository raw-data snapshot derived from PhysioNet/CinC 2019.
- `src/training/`: baseline and enhanced XGBoost training with patient-grouped cross-validation.
- `src/reporting/`: statistics and final compact table/figure builders.
- `scripts/`: public command-line wrappers for the reproducibility workflow.
- `jobs/`: SLURM entrypoints for HPC execution.
- `docs/`: public technical documentation.
- `results/`: compact final outputs and reproducibility manifests.
- `external_artifacts/inventory/`: checksums and provenance inventories for artifacts that are referenced but not required as public repository inputs.

## Expected Input

The repository-level reproducibility input is:

```text
data/raw/archive.zip
```

This file is a reproducibility snapshot derived from the public PhysioNet/Computing in Cardiology Challenge 2019 v1.0.0 training data. It should not be described as an official PhysioNet filename or a direct PhysioNet download link. The authoritative source remains PhysioNet Challenge 2019 v1.0.0, which provides access through its official project file tree and documented terminal/AWS download commands:

```bash
wget -r -N -c -np https://physionet.org/files/challenge-2019/1.0.0/
```

```bash
aws s3 sync --no-sign-request s3://physionet-open/challenge-2019/1.0.0/ DESTINATION
```

The harmonization script reads this ZIP directly and writes `data/processed/kaggle_harmonized.csv`.

## Execution Environment

Training is intended for the project HPC/container environment, using an NVIDIA NGC PyTorch 24.01 container image with GPU-enabled XGBoost, cuDF, CuPy, and cuML. Reporting scripts require Python with pandas, NumPy, scikit-learn, and matplotlib for figure generation.

## Reproduce Final Outputs

From the repository root:

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

## Output Directories

- `results/statistics/`: compact metric tables, statistical tests, validation checks, and checksums.
- `results/internal_robustness/`: internal SourceSet, subgroup, threshold, and calibration robustness summaries.
- `results/final_tables/`: final compact CSV tables.
- `results/final_figures/`: final compact PNG figures.

## Local Manuscript Note

The IEEE Access manuscript and manuscript-only assets are local editing material. They are not required to reproduce the public repository workflow.
