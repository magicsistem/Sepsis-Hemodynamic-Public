#!/usr/bin/env python3
"""Fixed-subset benchmark used by run.sh before choosing Slurm resources."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src import onset_koopman as onset
from src import scientific_pipeline as pipeline


BENCHMARK_XGBOOST_FITS = 10
BENCHMARK_XGBOOST_ESTIMATORS = 200
BENCHMARK_MAX_PATIENTS_PER_STRATUM = 1000
BENCHMARK_MIN_PATIENTS_PER_STRATUM = 500
FULL_FEATURE_FRAME_EQUIVALENTS_AT_MODEL_PEAK = 8


def estimated_full_peak_gb(
    full_feature_gb: float,
    largest_outer_quadratic_gb: float,
    primary_oof_pair_gb: float,
    robustness_wide_gb: float,
) -> float:
    return (
        FULL_FEATURE_FRAME_EQUIVALENTS_AT_MODEL_PEAK * full_feature_gb
        + 2 * largest_outer_quadratic_gb
        + primary_oof_pair_gb
        + robustness_wide_gb
    )


def fixed_patients(
    features: pd.DataFrame,
    maximum_per_stratum: int = BENCHMARK_MAX_PATIENTS_PER_STRATUM,
    minimum_per_stratum: int = BENCHMARK_MIN_PATIENTS_PER_STRATUM,
) -> list[str]:
    decisions = onset.primary_decisions(features)
    patients = decisions.groupby("Patient_ID", sort=True).agg(
        SourceSet=("SourceSet", "first"), outcome=(onset.TARGET_COLUMN, "max")
    ).reset_index()
    groups = list(patients.groupby(["SourceSet", "outcome"], sort=True))
    counts = {f"{source}:{int(outcome)}": int(len(group)) for (source, outcome), group in groups}
    if len(groups) != 4:
        raise pipeline.PipelineError(f"Fixed resource benchmark requires four source/outcome strata; got {counts}")
    per_stratum = min(maximum_per_stratum, min(counts.values()))
    if per_stratum < minimum_per_stratum:
        raise pipeline.PipelineError(
            f"Fixed resource benchmark has fewer than {minimum_per_stratum} eligible patients in a stratum: {counts}"
        )
    selected = []
    for _, group in groups:
        selected.extend(group.head(per_stratum)["Patient_ID"].tolist())
    return sorted(selected)


def benchmark(run_dir: Path, output: Path) -> dict:
    io_start = time.perf_counter()
    features = pd.read_csv(run_dir / "features.csv").sort_values(["Patient_ID", "ICULOS"], kind="mergesort").reset_index(drop=True)
    patients = fixed_patients(features)
    subset = features.loc[features["Patient_ID"].isin(patients)].reset_index(drop=True)
    io_seconds = time.perf_counter() - io_start
    compute_start = time.perf_counter()
    compute_cpu_start = time.process_time()
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
    if os.environ.get("REQUIRE_GPU", "false").lower() == "true" and not gpu["available"]:
        raise pipeline.PipelineError(f"GPU benchmark requires a usable device: {gpu['reason']}")
    fit_matrix = pipeline.matrix(decisions, columns)

    def fit_model(fit_index: int) -> np.ndarray:
        model = pipeline.xgb_model(
            pipeline.MODEL_CANDIDATES[0],
            pipeline.SEED + fit_index,
            gpu,
            n_estimators=BENCHMARK_XGBOOST_ESTIMATORS,
        )
        pipeline.fit_xgb(
            model,
            decisions,
            columns,
            target_column=onset.TARGET_COLUMN,
            train_matrix=fit_matrix,
        )
        return model.predict_proba(fit_matrix)[:, 1]

    xgboost_start = time.perf_counter()
    xgboost_cpu_start = time.process_time()
    concurrent_xgboost_fits = min(
        BENCHMARK_XGBOOST_FITS,
        pipeline.parallel_model_workers(gpu, BENCHMARK_XGBOOST_FITS)
        * pipeline.parallel_candidate_workers(gpu, len(pipeline.MODEL_CANDIDATES)),
    )
    probabilities = pipeline.ordered_parallel_map(
        fit_model, range(BENCHMARK_XGBOOST_FITS), concurrent_xgboost_fits
    )
    xgboost_seconds = time.perf_counter() - xgboost_start
    xgboost_cpu_seconds = time.process_time() - xgboost_cpu_start
    probability = probabilities[-1] if probabilities else None
    if probability is None or not np.isfinite(probability).all():
        raise pipeline.PipelineError("Resource benchmark produced invalid probabilities")
    compute_seconds = time.perf_counter() - compute_start
    compute_cpu_seconds = time.process_time() - compute_cpu_start
    cpus = pipeline.allocated_total_cpu_count()
    state_width = min(len(supported), onset.KOOPMAN_POLICY["maximum_signals"]) * 2
    quadratic_width = state_width + state_width * (state_width + 1) // 2
    full_feature_gb = float(features.memory_usage(index=True, deep=True).sum() / 1024 ** 3)
    largest_outer_quadratic_gb = float(len(features) * 0.8 * quadratic_width * 4 / 1024 ** 3)
    eligible_rows = int((features[onset.ELIGIBLE_COLUMN] == 1).sum())
    identity_sample = features.loc[
        features[onset.ELIGIBLE_COLUMN] == 1,
        ["Patient_ID", "SourceSet", "OnsetReconstructionStatus"],
    ].head(10000)
    identity_bytes_per_row = float(
        identity_sample.memory_usage(index=False, deep=True).sum()
        / len(identity_sample)
    )
    primary_oof_pair_gb = float(
        eligible_rows * (identity_bytes_per_row + 11 * 8) * 2 / 1024 ** 3
    )
    robustness_wide_gb = float(
        eligible_rows * (identity_bytes_per_row + 3 * 8 + 10 * 4) / 1024 ** 3
    )
    payload = {
        "status": "PASS",
        "fixed_subset_patient_hash": pipeline.stable_hash(patients),
        "n_patients": len(patients),
        "patients_per_source_outcome_stratum": len(patients) // 4,
        "maximum_patients_per_source_outcome_stratum": BENCHMARK_MAX_PATIENTS_PER_STRATUM,
        "minimum_patients_per_source_outcome_stratum": BENCHMARK_MIN_PATIENTS_PER_STRATUM,
        "n_rows": len(subset),
        "selected_signals": list(supported),
        "io_seconds": io_seconds,
        "compute_seconds": compute_seconds,
        "compute_cpu_seconds": compute_cpu_seconds,
        "active_cpu_efficiency": compute_cpu_seconds / (compute_seconds * cpus),
        "total_seconds": io_seconds + compute_seconds,
        "rows_per_compute_second": len(subset) / compute_seconds,
        "koopman_fit_seconds": fit_seconds,
        "koopman_training_transition_counts": transition_counts,
        "xgboost_fits": BENCHMARK_XGBOOST_FITS,
        "concurrent_xgboost_fits": concurrent_xgboost_fits,
        "fit_threads": pipeline.allocated_cpu_count(),
        "parallel_workers": pipeline.parallel_model_workers(
            gpu, BENCHMARK_XGBOOST_FITS
        ),
        "parallel_candidates": pipeline.parallel_candidate_workers(
            gpu, len(pipeline.MODEL_CANDIDATES)
        ),
        "xgboost_estimators_per_fit": BENCHMARK_XGBOOST_ESTIMATORS,
        "xgboost_fit_predict_seconds": xgboost_seconds,
        "xgboost_cpu_seconds": xgboost_cpu_seconds,
        "xgboost_active_cpu_efficiency": xgboost_cpu_seconds
        / (xgboost_seconds * cpus),
        "planned_full_xgboost_fits": pipeline.MODEL_POLICY[
            "planned_full_xgboost_fits"
        ],
        "planned_full_koopman_fits": pipeline.MODEL_POLICY[
            "planned_full_koopman_fits"
        ],
        "benchmark_xgboost_to_koopman_fit_ratio": (
            BENCHMARK_XGBOOST_FITS / len(onset.KOOPMAN_POLICY["lifts"])
        ),
        "planned_xgboost_to_koopman_fit_ratio": (
            pipeline.MODEL_POLICY["planned_full_xgboost_fits"]
            / pipeline.MODEL_POLICY["planned_full_koopman_fits"]
        ),
        "gpu": gpu,
        "full_feature_memory_gb": full_feature_gb,
        "full_feature_frame_equivalents_at_model_peak": (
            FULL_FEATURE_FRAME_EQUIVALENTS_AT_MODEL_PEAK
        ),
        "largest_outer_quadratic_matrix_gb": largest_outer_quadratic_gb,
        "primary_oof_pair_estimated_gb": primary_oof_pair_gb,
        "robustness_wide_oof_estimated_gb": robustness_wide_gb,
        "estimated_full_peak_gb": estimated_full_peak_gb(
            full_feature_gb,
            largest_outer_quadratic_gb,
            primary_oof_pair_gb,
            robustness_wide_gb,
        ),
        "estimation_note": (
            "eight concurrent full-feature-frame equivalents: seven measured at "
            "the nested C0 boundary plus one for transient model-matrix/runtime "
            "overhead; two largest outer-fold quadratic matrices, "
            "paired C0/C3 primary OOF, and one identity plus ten float32 robustness "
            "probability columns; no artificial allocation"
        ),
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
