#!/usr/bin/env python3
"""Fixed-subset benchmark used by run.sh before choosing Slurm resources."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src import onset_koopman as onset
from src import scientific_pipeline as pipeline


def fixed_patients(features: pd.DataFrame, per_source_outcome: int = 250) -> list[str]:
    decisions = onset.primary_decisions(features)
    patients = decisions.groupby("Patient_ID", sort=True).agg(
        SourceSet=("SourceSet", "first"), outcome=(onset.TARGET_COLUMN, "max")
    ).reset_index()
    selected = []
    for _, group in patients.groupby(["SourceSet", "outcome"], sort=True):
        selected.extend(group.head(per_source_outcome)["Patient_ID"].tolist())
    if len(selected) < per_source_outcome * 4:
        raise pipeline.PipelineError("Fixed resource benchmark cannot fill all source/outcome strata")
    return sorted(selected)


def benchmark(run_dir: Path, output: Path) -> dict:
    io_start = time.perf_counter()
    features = pd.read_csv(run_dir / "features.csv").sort_values(["Patient_ID", "ICULOS"], kind="mergesort").reset_index(drop=True)
    patients = fixed_patients(features)
    subset = features.loc[features["Patient_ID"].isin(patients)].reset_index(drop=True)
    io_seconds = time.perf_counter() - io_start
    compute_start = time.perf_counter()
    supported = onset.select_dynamic_signals(subset, pipeline.DYNAMIC_COLUMNS, 0.05, 25, 20)
    if not supported:
        raise pipeline.PipelineError("Fixed benchmark has no supported dynamic signal")
    transformed = None
    transition_counts = {}
    fit_seconds = {}
    for lift in onset.KOOPMAN_POLICY["lifts"]:
        start = time.perf_counter()
        fitted = onset.fit_koopman(subset, pipeline.DYNAMIC_COLUMNS, lift, selected_signals=supported)
        transformed = pd.concat([subset, onset.transform_koopman(subset, fitted)], axis=1)
        fit_seconds[lift] = time.perf_counter() - start
        transition_counts[lift] = fitted.training_transition_counts
    assert transformed is not None
    decisions = onset.primary_decisions(transformed)
    columns = pipeline.primary_model_features(transformed, "C3")
    gpu = pipeline.gpu_runtime()
    if __import__("os").environ.get("REQUIRE_GPU", "false").lower() == "true" and not gpu["available"]:
        raise pipeline.PipelineError(f"GPU benchmark requires a usable device: {gpu['reason']}")
    model = pipeline.xgb_model(pipeline.MODEL_CANDIDATES[0], pipeline.SEED, gpu, n_estimators=50)
    pipeline.fit_xgb(model, decisions, columns, target_column=onset.TARGET_COLUMN)
    probability = model.predict_proba(pipeline.matrix(decisions, columns))[:, 1]
    if not np.isfinite(probability).all():
        raise pipeline.PipelineError("Resource benchmark produced invalid probabilities")
    compute_seconds = time.perf_counter() - compute_start
    state_width = min(len(supported), onset.KOOPMAN_POLICY["maximum_signals"]) * 2
    quadratic_width = state_width + state_width * (state_width + 1) // 2
    full_feature_gb = float(features.memory_usage(index=True, deep=True).sum() / 1024 ** 3)
    largest_outer_quadratic_gb = float(len(features) * 0.8 * quadratic_width * 4 / 1024 ** 3)
    payload = {
        "status": "PASS",
        "fixed_subset_patient_hash": pipeline.stable_hash(patients),
        "n_patients": len(patients),
        "n_rows": len(subset),
        "selected_signals": list(supported),
        "io_seconds": io_seconds,
        "compute_seconds": compute_seconds,
        "total_seconds": io_seconds + compute_seconds,
        "rows_per_compute_second": len(subset) / compute_seconds,
        "koopman_fit_seconds": fit_seconds,
        "koopman_training_transition_counts": transition_counts,
        "xgboost_fits": 1,
        "gpu": gpu,
        "full_feature_memory_gb": full_feature_gb,
        "largest_outer_quadratic_matrix_gb": largest_outer_quadratic_gb,
        "estimated_full_peak_gb": full_feature_gb + 2 * largest_outer_quadratic_gb,
        "estimation_note": "feature-frame memory plus two largest outer-fold quadratic matrices; no artificial allocation",
    }
    pipeline.atomic_json(output, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        payload = benchmark(args.run_dir, args.output)
    except (pipeline.PipelineError, onset.OnsetKoopmanError) as exc:
        print(f"FAIL: {exc}")
        return 1
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
