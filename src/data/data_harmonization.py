#!/usr/bin/env python3
"""
Data harmonization for the PhysioNet/CinC 2019 public training set.

The script reads the repository-level raw-data snapshot derived from the public
Challenge 2019 v1.0.0 training data, preserves SourceSet when the archive
contains training_setA/training_setB paths, and writes one harmonized CSV for
the training workflow.
"""

import os
import sys
import argparse
import logging
import zipfile


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Data harmonization for the PhysioNet/CinC 2019 public training set"
    )
    parser.add_argument(
        "--kaggle_path",
        type=str,
        required=True,
        help="Path to the repository raw-data snapshot derived from PhysioNet/CinC 2019 v1.0.0 (default public example: data/raw/archive.zip)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory where harmonized CSV will be saved"
    )
    return parser


if any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
    build_arg_parser().parse_args()
    raise SystemExit(0)

import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Column mapping from source names to project-wide names.
KAGGLE_MAPPING = {
    # Vital signs
    'HR': 'HeartRate',
    'O2Sat': 'O2Sat',
    'Temp': 'Temperature',
    'SBP': 'SysBP',
    'MAP': 'MeanBP',
    'DBP': 'DiaBP',
    'Resp': 'RespRate',

    # Identifiers and label
    'patient_id': 'Patient_ID',
    'SepsisLabel': 'SepsisLabel',

    # Laboratory values
    'Glucose': 'Glucose',
    'BUN': 'BUN',
    'Creatinine': 'Creatinine',
    'Lactate': 'Lactate',
    'Calcium': 'Calcium',
    'Chloride': 'Chloride',
    'Potassium': 'Potassium',
    'Hgb': 'Hgb',
    'WBC': 'WBC',
    'Platelets': 'Platelets',
    'pH': 'pH',
    'PaCO2': 'PaCO2',
    'HCO3': 'HCO3',
    'BaseExcess': 'BaseExcess',
    'Bilirubin_total': 'Bilirubin',
    'AST': 'AST',
    'ALT': 'ALT',
    'Alkalinephos': 'Alkalinephos',
    'FiO2': 'FiO2',

    # Other variables
    'TroponinI': 'TroponinI',
    'Fibrinogen': 'Fibrinogen',
    'PT': 'PT',
    'PTT': 'PTT',
    'INR': 'INR',
    'HCT': 'HCT',
    'Age': 'Age',
    'Gender': 'Gender',
    'Unit1': 'Unit1',
    'Unit2': 'Unit2',
    'HospAdmTime': 'HospAdmTime',
    'ICULOS': 'ICULOS',
    'SaO2': 'SaO2',
    'EtCO2': 'EtCO2',
    'Bilirubin_direct': 'Bilirubin_direct',
    'Magnesium': 'Magnesium',
    'Phosphate': 'Phosphate',
}

META_COLS = ['Patient_ID', 'TimeStep', 'SourceSet', 'SepsisLabel']


def infer_source_set_from_zip_path(fname):
    normalized = fname.replace('\\', '/')
    if 'training_setA' in normalized.split('/'):
        return 'A'
    if 'training_setB' in normalized.split('/'):
        return 'B'
    return 'Unknown'


def extract_and_load_kaggle(filepath):
    """
    Loads PhysioNet 2019 patient-level files from the ZIP archive.
    If individual .psv files are present, aggregate .csv/.tsv files are skipped
    to avoid duplicating the cohort.
    Adds a 'Patient_ID' column if not present (using file name).
    Returns a single concatenated DataFrame.
    """
    df_list = []
    patient_id_col_candidates = {'Patient_ID', 'patient_id', 'id', 'subject_id'}

    if not filepath.endswith('.zip'):
        raise ValueError(f"Expected a .zip file, got {filepath}")

    with zipfile.ZipFile(filepath, 'r') as z:
        zip_files = z.namelist()
        psv_files = [f for f in zip_files if f.endswith('.psv')]
        csv_tsv_files = [f for f in zip_files if f.endswith(('.csv', '.tsv'))]

        logging.info(f"Found {len(psv_files)} PSV files in {filepath}")
        logging.info(f"Found {len(csv_tsv_files)} CSV/TSV files in {filepath}")

        if psv_files:
            tabular_files = psv_files
            logging.info(
                "Using PSV files only; aggregate CSV/TSV files skipped to avoid duplication"
            )
            if csv_tsv_files:
                logging.info("Skipped CSV/TSV files: %s", ", ".join(csv_tsv_files))
        else:
            tabular_files = csv_tsv_files
            logging.info("No PSV files found; using CSV/TSV files as fallback")

        for fname in tabular_files:
            if fname.endswith('.psv'):
                sep = '|'
            else:
                with z.open(fname) as f_sample:
                    first_line = f_sample.readline().decode('utf-8', errors='ignore')
                    sep = '\t' if '\t' in first_line else ','

            with z.open(fname) as f:
                df = pd.read_csv(f, sep=sep)

            df['SourceSet'] = infer_source_set_from_zip_path(fname)

            if not patient_id_col_candidates.intersection(df.columns):
                patient_id = os.path.splitext(os.path.basename(fname))[0]
                df['Patient_ID'] = patient_id

            df_list.append(df)

    if not df_list:
        raise ValueError("No data loaded from the archive.")

    return pd.concat(df_list, ignore_index=True)


def preprocess_kaggle(df):
    """
    Ensures the label is correctly typed and logs class prevalence.
    The PhysioNet 2019 dataset already has SepsisLabel shifted 6 hours ahead.
    """
    df = df.copy()
    df['Patient_ID'] = df['Patient_ID'].astype(str)

    if 'SourceSet' not in df.columns:
        df['SourceSet'] = 'Unknown'
    else:
        df['SourceSet'] = df['SourceSet'].fillna('Unknown').astype(str)

    if 'TimeStep' not in df.columns:
        df = df.sort_values('Patient_ID').reset_index(drop=True)
        df['TimeStep'] = df.groupby('Patient_ID').cumcount() + 1
    else:
        df = df.sort_values(['Patient_ID', 'TimeStep']).reset_index(drop=True)

    df['SepsisLabel'] = df['SepsisLabel'].astype('int8')

    n_pos = int(df['SepsisLabel'].sum())
    n_total = len(df)
    logging.info(
        f"Kaggle data: {n_total:,} rows, positive time steps: "
        f"{n_pos:,} ({100 * n_pos / n_total:.2f}%)"
    )

    source_summary = (
        df.groupby('SourceSet')
        .agg(
            patients=('Patient_ID', 'nunique'),
            rows=('Patient_ID', 'size'),
            sepsis_prevalence=('SepsisLabel', 'mean'),
        )
        .reset_index()
    )
    for row in source_summary.itertuples(index=False):
        logging.info(
            "SourceSet %s: patients=%s, rows=%s, SepsisLabel prevalence=%.2f%%",
            row.SourceSet,
            f"{row.patients:,}",
            f"{row.rows:,}",
            100 * row.sepsis_prevalence,
        )

    return df


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    logging.info("=== Loading raw Kaggle data ===")
    df_raw = extract_and_load_kaggle(args.kaggle_path)
    logging.info(f"Raw data shape: {df_raw.shape}")

    logging.info("=== Applying column mapping ===")
    df_raw = df_raw.rename(columns=KAGGLE_MAPPING)

    ordered_candidates = list(KAGGLE_MAPPING.values()) + META_COLS
    keep_cols = []
    for col in ordered_candidates:
        if col in df_raw.columns and col not in keep_cols:
            keep_cols.append(col)

    df = df_raw[keep_cols].copy()
    logging.info(f"Kept {len(keep_cols)} columns: {keep_cols}")

    logging.info("=== Skipping global z-score normalisation to avoid CV leakage ===")

    logging.info("=== Preprocessing (label verification, TimeStep) ===")
    df = preprocess_kaggle(df)

    logging.info("=== Saving harmonized dataset ===")
    out_path = os.path.join(args.output_dir, "kaggle_harmonized.csv")
    df.to_csv(out_path, index=False)

    nan_count = df.drop(columns=META_COLS, errors='ignore').isna().sum().sum()
    logging.info(f"Saved: {out_path} | shape {df.shape} | NaNs: {nan_count:,}")


if __name__ == "__main__":
    main()
