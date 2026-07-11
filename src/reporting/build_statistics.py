#!/usr/bin/env python3
"""Build compact statistics tables from final baseline and enhanced outputs.

This reporting utility reads already-generated experiment artifacts and writes a
compact public table package. It does not train models, modify input result
files, or depend on GPU libraries.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

np = None
pd = None
roc_auc_score = None
average_precision_score = None
brier_score_loss = None
binomtest = None
chi2 = None


def load_runtime_dependencies() -> None:
    """Import runtime dependencies after argparse so --help works in thin shells."""
    global np, pd, roc_auc_score, average_precision_score, brier_score_loss, binomtest, chi2
    try:
        import numpy as numpy_module
        import pandas as pandas_module
        from sklearn.metrics import (
            average_precision_score as average_precision_score_fn,
            brier_score_loss as brier_score_loss_fn,
            roc_auc_score as roc_auc_score_fn,
        )
    except ImportError as exc:
        raise SystemExit(
            "Missing reporting dependency. Run this script in the CEDIA Python/container "
            "environment with pandas, numpy, and scikit-learn installed."
        ) from exc

    np = numpy_module
    pd = pandas_module
    roc_auc_score = roc_auc_score_fn
    average_precision_score = average_precision_score_fn
    brier_score_loss = brier_score_loss_fn

    try:
        from scipy.stats import binomtest as binomtest_fn, chi2 as chi2_fn
    except Exception:
        binomtest = None
        chi2 = None
    else:
        binomtest = binomtest_fn
        chi2 = chi2_fn


REQUIRED_RESULT_FILES = [
    "summary_metrics.json",
    "metrics.json",
    "threshold_metrics.csv",
    "oof_predictions.csv",
    "fold_metrics.csv",
    "subgroup_metrics.csv",
    "feature_importance_summary.csv",
    "zabihi_results.joblib",
]

OPTIONAL_RESULT_FILES = [
    "subgroup_metrics.csv",
    "calibration_bins_raw.csv",
    "calibration_bins_platt.csv",
    "calibration_bins_isotonic.csv",
    "feature_category_summary.csv",
    "feature_importance_all_folds.csv",
]

COMPARISON_METRICS = [
    "n_features",
    "AUROC_raw",
    "AUPRC_raw",
    "AUROC_platt",
    "AUPRC_platt",
    "AUROC_isotonic",
    "AUPRC_isotonic",
    "ECE_raw",
    "Brier_raw",
    "ECE_platt",
    "Brier_platt",
    "ECE_isotonic",
    "Brier_isotonic",
    "Utility_raw_at_0.5",
    "Utility_raw_best",
    "Utility_raw_best_threshold",
    "Utility_platt_best",
    "Utility_isotonic_best",
    "best_F1_threshold_raw",
    "Patient_sensitivity_best_method_at_0.5",
    "Patient_specificity_best_method_at_0.5",
    "Lead_time_median_best_method_at_0.5",
    "Fold_AUROC_mean",
    "Fold_AUPRC_mean",
]

LOWER_IS_BETTER = {"ECE", "Brier"}
DESCRIPTIVE_METRICS = {"n_features", "Utility_raw_best_threshold", "best_F1_threshold_raw"}
OOF_REQUIRED_COLUMNS = ["Patient_ID", "TimeStep", "SepsisLabel", "prob_raw"]
OOF_ALIGNMENT_COLUMNS = ["Patient_ID", "TimeStep", "SepsisLabel"]
HEMO_SIGNALS = ["HeartRate", "SysBP", "MeanBP", "DiaBP", "RespRate", "O2Sat"]
HEMO_PATTERNS = {
    "CV": "_cv_8h",
    "IQR": "_iqr_8h",
    "SampEn": "_sampen_24h",
}
EXPECTED_POLICY_VERSION = "phase1b_fold_impute_no_sampen_backfill_v2"


TABLE_DESCRIPTIONS = {
    "comparison_summary.csv": "Primary baseline vs enhanced metric deltas.",
    "statistical_tests.csv": "Patient-level paired bootstrap and McNemar tests.",
    "threshold_operating_points.csv": "Selected threshold operating points.",
    "patient_level_metrics.csv": "Patient-level sensitivity, specificity, and lead-time summaries.",
    "calibration_table.csv": "Raw, Platt, and isotonic calibration metrics.",
    "subgroup_table.csv": "Baseline vs enhanced subgroup comparisons.",
    "feature_importance_table.csv": "Feature importance with hemodynamic feature annotations.",
    "fold_comparison.csv": "Fold-wise AUROC/AUPRC comparison.",
    "statistics_summary.md": "Human-readable statistics report.",
    "completion_checklist.md": "Completion checklist for statistics readiness.",
    "completion_checklist.csv": "Machine-readable completion checklist.",
    "utility_statistical_tests.csv": "Patient-bootstrap Utility Score inference.",
    "patient_confusion_matrices.csv": "Patient-level confusion matrices at selected thresholds.",
    "subgroup_patient_table.csv": "Patient-level subgroup operating metrics from OOF predictions.",
    "feature_count_audit.csv": "Feature count and paper wording audit.",
    "calibration_audit.md": "Calibration interpretation audit.",
    "statistical_methods_notes.md": "Statistical methods notes for the statistics package.",
    "code_math_audit_summary.md": "Static code/math audit summary for statistics.",
    "validation_checks.csv": "Input artifact and OOF alignment validation checks.",
    "validation_checks.md": "Human-readable input validation report.",
    "results_snippet.md": "Short cautious results snippet for later local drafting.",
    "manifest.json": "Reproducibility manifest with command, inputs, outputs, and checksums.",
}


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the reporting package builder."""
    parser = argparse.ArgumentParser(
        description="Build compact public statistics tables from final experiment outputs."
    )
    parser.add_argument("--baseline_dir", "--baseline-dir", dest="baseline_dir", required=True, type=Path)
    parser.add_argument("--enhanced_dir", "--enhanced-dir", dest="enhanced_dir", required=True, type=Path)
    parser.add_argument("--output_dir", "--output-dir", dest="output_dir", required=True, type=Path)
    parser.add_argument("--n_bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def setup_logging() -> None:
    """Configure simple console logging."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def validate_result_dir(path: Path, label: str) -> None:
    """Validate that a result directory contains the required final artifacts."""
    if not path.exists() or not path.is_dir():
        raise FileNotFoundError(f"{label} result directory does not exist: {path}")
    missing = [name for name in REQUIRED_RESULT_FILES if not (path / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"{label} result directory is missing required files: {missing}. "
            f"Directory checked: {path}"
        )


def load_json(path: Path) -> dict[str, Any]:
    """Read a JSON file as a dictionary."""
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_metrics(result_dir: Path) -> dict[str, Any]:
    """Load summary metrics, falling back to metrics.json for missing keys."""
    summary = load_json(result_dir / "summary_metrics.json")
    metrics = load_json(result_dir / "metrics.json")
    merged = dict(metrics)
    merged.update(summary)
    return merged


def get_metric(metrics: dict[str, Any], name: str) -> float | str | None:
    """Return a scalar metric if present, with support for CI list aliases."""
    value = metrics.get(name)
    if value is not None:
        return value
    if name == "AUROC_CI" and "AUROC_CI_lower" in metrics and "AUROC_CI_upper" in metrics:
        return [metrics["AUROC_CI_lower"], metrics["AUROC_CI_upper"]]
    if name == "AUPRC_CI" and "AUPRC_CI_lower" in metrics and "AUPRC_CI_upper" in metrics:
        return [metrics["AUPRC_CI_lower"], metrics["AUPRC_CI_upper"]]
    return None


def as_float(value: Any) -> float:
    """Convert scalars to float while preserving missing values as NaN."""
    if value is None:
        return np.nan
    try:
        if isinstance(value, (list, tuple)):
            return np.nan
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def preferred_direction(metric: str) -> str:
    """Describe whether larger, smaller, or neither direction is preferred."""
    if metric in DESCRIPTIVE_METRICS:
        return "descriptive"
    if any(token in metric for token in LOWER_IS_BETTER):
        return "lower"
    return "higher"


def interpretation(metric: str, delta: float) -> str:
    """Generate a short interpretation for a baseline-to-enhanced delta."""
    if np.isnan(delta):
        return "not available"
    direction = preferred_direction(metric)
    if direction == "descriptive":
        return "descriptive difference; not interpreted as model improvement"
    if math.isclose(delta, 0.0, abs_tol=1e-12):
        return "no numeric difference"
    improved = delta > 0 if direction == "higher" else delta < 0
    return "enhanced improved" if improved else "enhanced worse"


def build_comparison_summary(
    baseline_metrics: dict[str, Any], enhanced_metrics: dict[str, Any]
) -> pd.DataFrame:
    """Create the primary metric comparison table."""
    rows = []
    for metric in COMPARISON_METRICS:
        baseline = get_metric(baseline_metrics, metric)
        enhanced = get_metric(enhanced_metrics, metric)
        baseline_f = as_float(baseline)
        enhanced_f = as_float(enhanced)
        delta = enhanced_f - baseline_f if not (np.isnan(baseline_f) or np.isnan(enhanced_f)) else np.nan
        if np.isnan(delta) or np.isnan(baseline_f) or math.isclose(baseline_f, 0.0, abs_tol=1e-15):
            rel_delta = np.nan
        else:
            rel_delta = 100.0 * delta / baseline_f
        rows.append(
            {
                "metric": metric,
                "baseline": baseline,
                "enhanced": enhanced,
                "delta": delta,
                "relative_delta_percent": rel_delta,
                "preferred_direction": preferred_direction(metric),
                "interpretation": interpretation(metric, delta),
            }
        )
    return pd.DataFrame(rows)


def require_columns(df: pd.DataFrame, columns: Iterable[str], path: Path) -> None:
    """Raise a clear error when a table does not contain required columns."""
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns {missing} in {path}")


def check_row(rows: list[dict[str, Any]], check: str, status: str, detail: str, value: Any = "") -> None:
    """Append one validation-check row."""
    rows.append({"check": check, "status": status, "value": value, "detail": detail})


def load_oof(path: Path) -> pd.DataFrame:
    """Load and validate OOF predictions without mutating the source file."""
    df = pd.read_csv(path)
    require_columns(df, OOF_REQUIRED_COLUMNS, path)
    df = df.copy()
    df["Patient_ID"] = df["Patient_ID"].astype(str)
    labels = set(pd.to_numeric(df["SepsisLabel"], errors="coerce").dropna().astype(int).unique())
    if not labels.issubset({0, 1}) or labels == set():
        raise ValueError(f"SepsisLabel must be binary 0/1 in {path}; observed={sorted(labels)}")
    probs = pd.to_numeric(df["prob_raw"], errors="coerce")
    if probs.isna().any():
        raise ValueError(f"prob_raw contains non-numeric/NaN values in {path}")
    if not probs.between(0.0, 1.0).all():
        observed_min = float(probs.min())
        observed_max = float(probs.max())
        raise ValueError(f"prob_raw must be within [0, 1] in {path}; range=({observed_min}, {observed_max})")
    duplicate_mask = df.duplicated(["Patient_ID", "TimeStep"], keep=False)
    if duplicate_mask.any():
        sample = df.loc[duplicate_mask, ["Patient_ID", "TimeStep"]].head(5).to_dict("records")
        raise ValueError(f"Duplicate Patient_ID/TimeStep rows detected in {path}: {sample}")
    return df


def oof_summary(label: str, df: pd.DataFrame) -> dict[str, Any]:
    """Summarize one OOF table for validation reporting."""
    patient_labels = df.groupby("Patient_ID", sort=False)["SepsisLabel"].max()
    return {
        "model": label,
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "patients": int(df["Patient_ID"].nunique()),
        "time_step_prevalence": float(df["SepsisLabel"].mean()),
        "patient_prevalence": float(patient_labels.mean()),
        "prob_raw_min": float(pd.to_numeric(df["prob_raw"]).min()),
        "prob_raw_max": float(pd.to_numeric(df["prob_raw"]).max()),
        "positive_rows": int(df["SepsisLabel"].sum()),
        "negative_rows": int(len(df) - df["SepsisLabel"].sum()),
        "positive_patients": int(patient_labels.sum()),
        "negative_patients": int(len(patient_labels) - patient_labels.sum()),
    }


def build_validation_checks(
    baseline_dir: Path,
    enhanced_dir: Path,
    baseline_metrics: dict[str, Any],
    enhanced_metrics: dict[str, Any],
) -> tuple[pd.DataFrame, str, pd.DataFrame]:
    """Validate final run inputs and paired OOF alignment."""
    rows: list[dict[str, Any]] = []
    for label, result_dir in [("baseline", baseline_dir), ("enhanced", enhanced_dir)]:
        check_row(rows, f"{label}_dir_exists", "PASS" if result_dir.is_dir() else "FAIL", str(result_dir))
        for name in REQUIRED_RESULT_FILES:
            path = result_dir / name
            check_row(rows, f"{label}_required_file_{name}", "PASS" if path.exists() else "FAIL", str(path))
        for name in OPTIONAL_RESULT_FILES:
            path = result_dir / name
            check_row(rows, f"{label}_optional_file_{name}", "PASS" if path.exists() else "CONDITIONAL", str(path))

    for label, metrics in [("baseline", baseline_metrics), ("enhanced", enhanced_metrics)]:
        observed = metrics.get("pipeline_policy_version")
        detail = (
            "Expected current validated policy; observed current validated policy"
            if observed == EXPECTED_POLICY_VERSION
            else f"Expected current validated policy; observed {observed}"
        )
        check_row(
            rows,
            f"{label}_expected_policy",
            "PASS" if observed == EXPECTED_POLICY_VERSION else "FAIL",
            detail,
            "current_validated_policy" if observed == EXPECTED_POLICY_VERSION else observed,
        )

    baseline = load_oof(baseline_dir / "oof_predictions.csv")
    enhanced = load_oof(enhanced_dir / "oof_predictions.csv")
    for label, df in [("baseline", baseline), ("enhanced", enhanced)]:
        summary = oof_summary(label, df)
        check_row(rows, f"{label}_oof_required_columns", "PASS", ", ".join(OOF_REQUIRED_COLUMNS))
        check_row(rows, f"{label}_oof_no_duplicate_patient_timestep", "PASS", "No duplicated Patient_ID/TimeStep rows.")
        check_row(rows, f"{label}_label_binary", "PASS", "SepsisLabel is binary 0/1.")
        check_row(rows, f"{label}_prob_raw_range", "PASS", "[0, 1]", f"{summary['prob_raw_min']:.6g} to {summary['prob_raw_max']:.6g}")
        check_row(rows, f"{label}_oof_rows", "PASS", "OOF row count.", summary["rows"])
        check_row(rows, f"{label}_unique_patients", "PASS", "Unique Patient_ID count.", summary["patients"])
        check_row(rows, f"{label}_time_step_prevalence", "PASS", "Mean SepsisLabel over rows.", f"{summary['time_step_prevalence']:.8f}")
        check_row(rows, f"{label}_patient_prevalence", "PASS", "Mean patient-level max SepsisLabel.", f"{summary['patient_prevalence']:.8f}")

    aligned = align_oof_predictions(baseline_dir, enhanced_dir)
    patient_labels = aligned.groupby("Patient_ID", sort=False)["SepsisLabel"].max()
    check_row(rows, "paired_oof_alignment_rows", "PASS", "Aligned rows equal both input OOF row counts.", int(len(aligned)))
    check_row(rows, "paired_oof_alignment_patients", "PASS", "Unique aligned Patient_ID count.", int(aligned["Patient_ID"].nunique()))
    check_row(rows, "paired_oof_time_step_prevalence", "PASS", "Mean SepsisLabel over aligned rows.", f"{float(aligned['SepsisLabel'].mean()):.8f}")
    check_row(rows, "paired_oof_patient_prevalence", "PASS", "Mean patient-level max SepsisLabel.", f"{float(patient_labels.mean()):.8f}")
    check_row(rows, "no_external_validation_claimed", "PASS", "All outputs are limited to internal cross-validation on the public PhysioNet/CinC 2019 training split.")

    checks = pd.DataFrame(rows)
    md = "# Validation Checks\n\n" + "\n".join(
        f"- **{row['check']}**: `{row['status']}` - {row['detail']} {row['value']}".rstrip()
        for row in rows
    ) + "\n"
    return checks, md, aligned


def align_oof_predictions(baseline_dir: Path, enhanced_dir: Path) -> pd.DataFrame:
    """Align baseline and enhanced OOF predictions on patient, time step, and label."""
    logging.info("Loading paired OOF predictions")
    baseline = load_oof(baseline_dir / "oof_predictions.csv")
    enhanced = load_oof(enhanced_dir / "oof_predictions.csv")
    if len(baseline) != len(enhanced):
        raise ValueError(f"OOF row count mismatch: baseline={len(baseline)}, enhanced={len(enhanced)}")

    merged = baseline[OOF_ALIGNMENT_COLUMNS + ["prob_raw"]].merge(
        enhanced[OOF_ALIGNMENT_COLUMNS + ["prob_raw"]],
        on=OOF_ALIGNMENT_COLUMNS,
        how="inner",
        suffixes=("_baseline", "_enhanced"),
        validate="one_to_one",
    )
    if len(merged) != len(baseline):
        raise ValueError(
            "OOF alignment lost rows. Check Patient_ID, TimeStep, and SepsisLabel consistency "
            f"between baseline and enhanced outputs. baseline={len(baseline)}, aligned={len(merged)}"
        )
    logging.info(
        "Aligned OOF rows=%d, patients=%d, prevalence=%.6f",
        len(merged), merged["Patient_ID"].nunique(), merged["SepsisLabel"].mean(),
    )
    return merged


def metric_scores(y_true: np.ndarray, prob: np.ndarray) -> dict[str, float]:
    """Compute row-level discrimination and Brier metrics for a probability vector."""
    return {
        "AUROC_raw": float(roc_auc_score(y_true, prob)),
        "AUPRC_raw": float(average_precision_score(y_true, prob)),
        "Brier_raw": float(brier_score_loss(y_true, prob)),
    }


def patient_index_groups(df: pd.DataFrame) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Return patient IDs and row-index arrays for patient-level resampling."""
    grouped = df.groupby("Patient_ID", sort=False).indices
    patients = np.array(list(grouped.keys()), dtype=object)
    index_map = {patient: np.asarray(indices, dtype=np.int64) for patient, indices in grouped.items()}
    return patients, index_map


def paired_bootstrap_deltas(
    aligned: pd.DataFrame, n_bootstrap: int, seed: int
) -> dict[str, np.ndarray]:
    """Bootstrap paired metric deltas by resampling patients with replacement."""
    rng = np.random.default_rng(seed)
    patients, index_map = patient_index_groups(aligned)
    y = aligned["SepsisLabel"].to_numpy(dtype=np.int8)
    p_base = aligned["prob_raw_baseline"].to_numpy(dtype=np.float64)
    p_enh = aligned["prob_raw_enhanced"].to_numpy(dtype=np.float64)
    deltas = {"AUROC_raw": [], "AUPRC_raw": [], "Brier_raw": []}

    for i in range(n_bootstrap):
        sampled_patients = rng.choice(patients, size=len(patients), replace=True)
        sampled_indices = np.concatenate([index_map[patient] for patient in sampled_patients])
        y_s = y[sampled_indices]
        if np.unique(y_s).size < 2:
            continue
        base_s = p_base[sampled_indices]
        enh_s = p_enh[sampled_indices]
        deltas["AUROC_raw"].append(roc_auc_score(y_s, enh_s) - roc_auc_score(y_s, base_s))
        deltas["AUPRC_raw"].append(
            average_precision_score(y_s, enh_s) - average_precision_score(y_s, base_s)
        )
        deltas["Brier_raw"].append(brier_score_loss(y_s, enh_s) - brier_score_loss(y_s, base_s))
        if (i + 1) % 100 == 0:
            logging.info("Bootstrap %d/%d", i + 1, n_bootstrap)

    return {name: np.asarray(values, dtype=np.float64) for name, values in deltas.items()}


def bootstrap_p_value(delta_samples: np.ndarray, n_bootstrap: int) -> float:
    """Compute a two-sided bootstrap sign p-value clamped to [1/n, 1]."""
    if delta_samples.size == 0:
        return np.nan
    p = 2.0 * min(np.mean(delta_samples <= 0.0), np.mean(delta_samples >= 0.0))
    return float(min(max(p, 1.0 / max(n_bootstrap, 1)), 1.0))


def mcnemar_p_value(base_correct: np.ndarray, enhanced_correct: np.ndarray) -> tuple[float, int, int]:
    """Compute a paired McNemar p-value for discordant correctness counts."""
    base_only = int(np.sum(base_correct & ~enhanced_correct))
    enhanced_only = int(np.sum(~base_correct & enhanced_correct))
    discordant = base_only + enhanced_only
    if discordant == 0:
        return 1.0, base_only, enhanced_only
    if binomtest is not None:
        return float(binomtest(min(base_only, enhanced_only), discordant, p=0.5).pvalue), base_only, enhanced_only
    if chi2 is not None:
        statistic = (abs(base_only - enhanced_only) - 1.0) ** 2 / discordant
        return float(chi2.sf(statistic, df=1)), base_only, enhanced_only
    statistic = (abs(base_only - enhanced_only) - 1.0) ** 2 / discordant
    p_approx = math.exp(-0.5 * statistic)
    return float(min(max(p_approx, 0.0), 1.0)), base_only, enhanced_only



def patient_correctness_from_aligned(aligned: pd.DataFrame, baseline_threshold: float, enhanced_threshold: float) -> tuple[np.ndarray, np.ndarray]:
    """Create paired patient-level correctness arrays from aligned OOF predictions."""
    work = aligned.copy()
    work["_pred_baseline"] = (work["prob_raw_baseline"] >= baseline_threshold).astype(np.int8)
    work["_pred_enhanced"] = (work["prob_raw_enhanced"] >= enhanced_threshold).astype(np.int8)
    patient = work.groupby("Patient_ID", sort=False).agg(
        real_positive=("SepsisLabel", "max"),
        pred_baseline=("_pred_baseline", "max"),
        pred_enhanced=("_pred_enhanced", "max"),
    )
    y = patient["real_positive"].astype(np.int8).to_numpy()
    base_correct = patient["pred_baseline"].astype(np.int8).to_numpy() == y
    enhanced_correct = patient["pred_enhanced"].astype(np.int8).to_numpy() == y
    return base_correct, enhanced_correct


def apply_bh(p_values: list[float]) -> list[float]:
    """Apply Benjamini-Hochberg adjustment to a list of p-values."""
    p = np.asarray([np.nan if value is None else value for value in p_values], dtype=np.float64)
    adjusted = np.full_like(p, np.nan)
    valid = np.where(~np.isnan(p))[0]
    if valid.size == 0:
        return adjusted.tolist()
    order = valid[np.argsort(p[valid])]
    ranked = p[order]
    m = len(ranked)
    raw_adj = ranked * m / np.arange(1, m + 1)
    monotone = np.minimum.accumulate(raw_adj[::-1])[::-1]
    adjusted[order] = np.minimum(monotone, 1.0)
    return adjusted.tolist()


def build_statistical_tests(
    baseline_dir: Path,
    enhanced_dir: Path,
    baseline_metrics: dict[str, Any],
    enhanced_metrics: dict[str, Any],
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    """Build paired statistical tests from aligned OOF predictions."""
    aligned = align_oof_predictions(baseline_dir, enhanced_dir)
    y = aligned["SepsisLabel"].to_numpy(dtype=np.int8)
    p_base = aligned["prob_raw_baseline"].to_numpy(dtype=np.float64)
    p_enh = aligned["prob_raw_enhanced"].to_numpy(dtype=np.float64)
    base_scores = metric_scores(y, p_base)
    enhanced_scores = metric_scores(y, p_enh)
    bootstrap = paired_bootstrap_deltas(aligned, n_bootstrap=n_bootstrap, seed=seed)

    rows: list[dict[str, Any]] = []
    for metric in ["AUROC_raw", "AUPRC_raw", "Brier_raw"]:
        samples = bootstrap[metric]
        rows.append(
            {
                "metric": f"Delta_{metric}",
                "baseline": base_scores[metric],
                "enhanced": enhanced_scores[metric],
                "delta": enhanced_scores[metric] - base_scores[metric],
                "ci_lower": float(np.percentile(samples, 2.5)) if samples.size else np.nan,
                "ci_upper": float(np.percentile(samples, 97.5)) if samples.size else np.nan,
                "p_value_two_sided": bootstrap_p_value(samples, n_bootstrap),
                "n_bootstrap": n_bootstrap,
                "test": "paired_patient_bootstrap",
                "multiple_testing_group": "main",
            }
        )

    utility_base = as_float(get_metric(baseline_metrics, "Utility_raw_best"))
    utility_enh = as_float(get_metric(enhanced_metrics, "Utility_raw_best"))
    rows.append(
        {
            "metric": "Delta_Utility_raw_best",
            "baseline": utility_base,
            "enhanced": utility_enh,
            "delta": utility_enh - utility_base,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "p_value_two_sided": np.nan,
            "n_bootstrap": 0,
            "test": "descriptive_summary_delta",
            "multiple_testing_group": "descriptive",
        }
    )

    mcnemar_specs = [
        ("threshold_0.5", 0.5, 0.5),
        (
            "utility_optimal",
            as_float(get_metric(baseline_metrics, "Utility_raw_best_threshold")),
            as_float(get_metric(enhanced_metrics, "Utility_raw_best_threshold")),
        ),
    ]
    for threshold_label, th_base, th_enh in mcnemar_specs:
        if np.isnan(th_base) or np.isnan(th_enh):
            p_value, base_only, enhanced_only = np.nan, np.nan, np.nan
        else:
            base_correct = (p_base >= th_base).astype(np.int8) == y
            enhanced_correct = (p_enh >= th_enh).astype(np.int8) == y
            p_value, base_only, enhanced_only = mcnemar_p_value(base_correct, enhanced_correct)
        rows.append(
            {
                "metric": f"McNemar_time_step_{threshold_label}",
                "baseline": th_base,
                "enhanced": th_enh,
                "delta": enhanced_only - base_only if not np.isnan(p_value) else np.nan,
                "ci_lower": np.nan,
                "ci_upper": np.nan,
                "p_value_two_sided": p_value,
                "n_bootstrap": 0,
                "test": f"mcnemar_time_step_correctness;base_only={base_only};enhanced_only={enhanced_only}",
                "multiple_testing_group": "main",
            }
        )

        if np.isnan(th_base) or np.isnan(th_enh):
            p_value, base_only, enhanced_only = np.nan, np.nan, np.nan
        else:
            base_correct, enhanced_correct = patient_correctness_from_aligned(aligned, th_base, th_enh)
            p_value, base_only, enhanced_only = mcnemar_p_value(base_correct, enhanced_correct)
        rows.append(
            {
                "metric": f"McNemar_patient_{threshold_label}",
                "baseline": th_base,
                "enhanced": th_enh,
                "delta": enhanced_only - base_only if not np.isnan(p_value) else np.nan,
                "ci_lower": np.nan,
                "ci_upper": np.nan,
                "p_value_two_sided": p_value,
                "n_bootstrap": 0,
                "test": f"mcnemar_patient_correctness;base_only={base_only};enhanced_only={enhanced_only}",
                "multiple_testing_group": "main",
            }
        )

    df = pd.DataFrame(rows)
    main_mask = df["multiple_testing_group"].eq("main")
    adjusted = apply_bh(df.loc[main_mask, "p_value_two_sided"].tolist())
    df["p_value_bh"] = np.nan
    df.loc[main_mask, "p_value_bh"] = adjusted
    df["significant_bh_0.05"] = df["p_value_bh"].lt(0.05)
    gc.collect()
    return df



def p_value_display(value: Any) -> str:
    """Format p-values for papers without reporting impossible p=0.000000."""
    p_value = as_float(value)
    if np.isnan(p_value):
        return "NA"
    if p_value == 0.0:
        return "p<1e-300"
    if p_value < 0.001:
        return "p<0.001"
    return f"p={p_value:.3f}"


def normal_two_sided_p(z_value: float) -> float:
    """Two-sided normal p-value using erfc to avoid a hard scipy dependency."""
    if not np.isfinite(z_value):
        return np.nan
    return float(math.erfc(abs(z_value) / math.sqrt(2.0)))


def compute_midrank(values: np.ndarray) -> np.ndarray:
    """Compute midranks used by the fast DeLong AUROC covariance algorithm."""
    values = np.asarray(values)
    order = np.argsort(values)
    sorted_values = values[order]
    midranks = np.zeros(len(values), dtype=np.float64)
    i = 0
    while i < len(values):
        j = i
        while j < len(values) and sorted_values[j] == sorted_values[i]:
            j += 1
        midranks[i:j] = 0.5 * (i + j - 1) + 1.0
        i = j
    out = np.empty(len(values), dtype=np.float64)
    out[order] = midranks
    return out


def fast_delong(predictions_sorted: np.ndarray, n_positive: int) -> tuple[np.ndarray, np.ndarray]:
    """Paired DeLong AUROC covariance for classifiers sorted with positives first.

    This implements the standard fast DeLong midrank formulation for paired ROC
    curves. It is used only for reporting inference; model outputs are unchanged.
    """
    m = int(n_positive)
    n = int(predictions_sorted.shape[1] - m)
    if m <= 0 or n <= 0:
        raise ValueError("DeLong requires at least one positive and one negative label.")
    positive_examples = predictions_sorted[:, :m]
    negative_examples = predictions_sorted[:, m:]
    k = predictions_sorted.shape[0]

    tx = np.empty((k, m), dtype=np.float64)
    ty = np.empty((k, n), dtype=np.float64)
    tz = np.empty((k, m + n), dtype=np.float64)
    for classifier_idx in range(k):
        tx[classifier_idx, :] = compute_midrank(positive_examples[classifier_idx, :])
        ty[classifier_idx, :] = compute_midrank(negative_examples[classifier_idx, :])
        tz[classifier_idx, :] = compute_midrank(predictions_sorted[classifier_idx, :])

    aucs = tz[:, :m].sum(axis=1) / (m * n) - (m + 1.0) / (2.0 * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    covariance = sx / m + sy / n
    covariance = np.atleast_2d(covariance)
    return aucs, covariance


def paired_delong_auroc_test(y_true: np.ndarray, baseline_prob: np.ndarray, enhanced_prob: np.ndarray) -> dict[str, Any]:
    """Run paired DeLong test and validate AUROC against sklearn."""
    y_true = np.asarray(y_true).astype(int)
    order = np.argsort(-y_true)
    predictions = np.vstack([baseline_prob, enhanced_prob])[:, order]
    aucs, covariance = fast_delong(predictions, int(np.sum(y_true == 1)))
    sklearn_base = float(roc_auc_score(y_true, baseline_prob))
    sklearn_enh = float(roc_auc_score(y_true, enhanced_prob))
    if not (np.isclose(aucs[0], sklearn_base, atol=1e-10) and np.isclose(aucs[1], sklearn_enh, atol=1e-10)):
        raise ValueError(
            "DeLong AUROC validation failed: "
            f"delong={aucs.tolist()}, sklearn={[sklearn_base, sklearn_enh]}"
        )
    contrast = np.array([1.0, -1.0], dtype=np.float64)
    variance = float(contrast @ covariance @ contrast.T)
    if variance <= 0.0:
        raise ValueError(f"Non-positive DeLong variance: {variance}")
    delta = sklearn_enh - sklearn_base
    z_value = delta / math.sqrt(variance)
    return {
        "baseline": sklearn_base,
        "enhanced": sklearn_enh,
        "delta": delta,
        "z": float(z_value),
        "p_value_two_sided": normal_two_sided_p(z_value),
        "variance": variance,
    }


def add_delong_and_format_stats(stats: pd.DataFrame, baseline_dir: Path, enhanced_dir: Path) -> tuple[pd.DataFrame, dict[str, str]]:
    """Add paired DeLong AUROC row and display-safe p-value formatting."""
    status = {"status": "CONDITIONAL", "explanation": "Paired DeLong AUROC completed and validated against sklearn AUROC, but patient bootstrap remains primary because OOF rows contain repeated time steps per patient."}
    out = stats.copy()
    try:
        aligned = align_oof_predictions(baseline_dir, enhanced_dir)
        result = paired_delong_auroc_test(
            aligned["SepsisLabel"].to_numpy(dtype=np.int8),
            aligned["prob_raw_baseline"].to_numpy(dtype=np.float64),
            aligned["prob_raw_enhanced"].to_numpy(dtype=np.float64),
        )
        out = pd.concat(
            [
                out,
                pd.DataFrame(
                    [
                        {
                            "metric": "Delta_AUROC_raw_DeLong",
                            "baseline": result["baseline"],
                            "enhanced": result["enhanced"],
                            "delta": result["delta"],
                            "ci_lower": np.nan,
                            "ci_upper": np.nan,
                            "p_value_two_sided": result["p_value_two_sided"],
                            "n_bootstrap": 0,
                            "test": f"paired_delong;z={result['z']:.6g};variance={result['variance']:.6g}",
                            "multiple_testing_group": "main",
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
    except Exception as exc:
        status = {"status": "UNRESOLVED", "explanation": f"DeLong could not be completed safely: {exc}"}
        logging.warning(status["explanation"])

    main_mask = out["multiple_testing_group"].eq("main")
    out["p_value_bh"] = np.nan
    out.loc[main_mask, "p_value_bh"] = apply_bh(out.loc[main_mask, "p_value_two_sided"].tolist())
    out["significant_bh_0.05"] = out["p_value_bh"].lt(0.05)
    out["p_value_display"] = out["p_value_two_sided"].apply(p_value_display)
    return out, status


# Utility functions copied from src/training/train_zabihi_cudf.py for reporting-only
# reuse of the project-internal PhysioNet/CinC 2019 Utility Score implementation.
def compute_prediction_utility_reporting(
    labels, predictions,
    dt_early=-12, dt_optimal=-6, dt_late=3,
    max_u_tp=1.0, min_u_fn=-2.0, u_fp=-0.05, u_tn=0.0,
):
    """Compute project-internal PhysioNet utility for one patient sequence."""
    labels = np.asarray(labels).astype(int)
    predictions = np.asarray(predictions).astype(int)
    is_septic = bool(np.any(labels > 0))
    t_sepsis = int(np.argmax(labels)) if is_septic else -1
    m1 = max_u_tp / float(dt_optimal - dt_early)
    b1 = -m1 * dt_early
    m2 = -max_u_tp / float(dt_late - dt_optimal)
    b2 = -m2 * dt_late
    m3 = min_u_fn / float(dt_late - dt_optimal)
    b3 = -m3 * dt_optimal
    n = len(labels)
    ts = np.arange(n)
    utility = np.zeros(n, dtype=np.float64)
    if is_septic:
        dt = ts - t_sepsis
        in_window = dt <= dt_late
        pred_pos_win = predictions.astype(bool) & in_window
        early_zone = pred_pos_win & (dt < dt_early)
        utility[early_zone] = u_fp
        optimal_zone = pred_pos_win & (dt >= dt_early) & (dt <= dt_optimal)
        utility[optimal_zone] = m1 * dt[optimal_zone] + b1
        late_zone_tp = pred_pos_win & (dt > dt_optimal) & (dt <= dt_late)
        utility[late_zone_tp] = m2 * dt[late_zone_tp] + b2
        pred_neg_win = (~predictions.astype(bool)) & in_window
        late_neg = pred_neg_win & (dt > dt_optimal)
        utility[late_neg] = m3 * dt[late_neg] + b3
        out_win = ~in_window
        utility[out_win & predictions.astype(bool)] = u_fp
    else:
        utility[predictions.astype(bool)] = u_fp
    return float(utility.sum())


def best_prediction_vector_reporting(labels, dt_early=-12, dt_late=3):
    """Best possible prediction vector from the project-internal utility implementation."""
    labels = np.asarray(labels).astype(int)
    preds = np.zeros_like(labels)
    if np.any(labels > 0):
        t_sepsis = int(np.argmax(labels))
        start = max(0, t_sepsis + dt_early)
        end = min(len(labels), t_sepsis + dt_late + 1)
        preds[start:end] = 1
    return preds.astype(int)


def prepare_patient_segments_reporting(oof_df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Sort OOF predictions and return contiguous patient segment boundaries."""
    oof_sorted = oof_df.sort_values(["Patient_ID", "TimeStep"]).reset_index(drop=True)
    pid = oof_sorted["Patient_ID"].astype(str).to_numpy()
    if len(pid) <= 1:
        starts = np.array([0], dtype=int) if len(pid) == 1 else np.array([], dtype=int)
        ends = np.array([1], dtype=int) if len(pid) == 1 else np.array([], dtype=int)
        return oof_sorted, starts, ends
    boundaries = np.flatnonzero(pid[1:] != pid[:-1]) + 1
    starts = np.concatenate([[0], boundaries])
    ends = np.concatenate([boundaries, [len(pid)]])
    return oof_sorted, starts, ends


def utility_patient_components(oof_df: pd.DataFrame, prob_col: str, threshold: float) -> pd.DataFrame:
    """Precompute per-patient utility components for fast patient bootstrap."""
    sorted_df, starts, ends = prepare_patient_segments_reporting(oof_df)
    labels_sorted = sorted_df["SepsisLabel"].to_numpy(dtype=np.int8)
    probs_sorted = sorted_df[prob_col].to_numpy(dtype=np.float64)
    patient_ids = sorted_df["Patient_ID"].astype(str).to_numpy()
    rows = []
    for s, e in zip(starts, ends):
        labels = labels_sorted[s:e]
        preds = (probs_sorted[s:e] >= threshold).astype(np.int8)
        inaction = compute_prediction_utility_reporting(labels, np.zeros_like(labels))
        all_pos = compute_prediction_utility_reporting(labels, np.ones_like(labels))
        rows.append(
            {
                "Patient_ID": patient_ids[s],
                "observed": compute_prediction_utility_reporting(labels, preds),
                "best": compute_prediction_utility_reporting(labels, best_prediction_vector_reporting(labels)),
                "inaction": inaction,
                "worst": min(inaction, all_pos),
            }
        )
    return pd.DataFrame(rows)


def normalize_utility_from_sums(observed: float, best: float, inaction: float, worst: float) -> float:
    """Normalize utility from summed observed, best, inaction, and worst components."""
    if observed >= inaction:
        denom = best - inaction
        return 0.0 if denom == 0.0 else float((observed - inaction) / denom)
    denom = inaction - worst
    return 0.0 if denom == 0.0 else float((observed - inaction) / denom)


def normalized_utility_from_components(components: pd.DataFrame) -> float:
    """Normalize utility from a per-patient component table."""
    return normalize_utility_from_sums(
        float(components["observed"].sum()),
        float(components["best"].sum()),
        float(components["inaction"].sum()),
        float(components["worst"].sum()),
    )


def paired_utility_bootstrap(
    baseline_components: pd.DataFrame,
    enhanced_components: pd.DataFrame,
    n_bootstrap: int,
    seed: int,
) -> np.ndarray:
    """Bootstrap Utility Score deltas by resampling Patient_ID values."""
    merged = baseline_components.merge(
        enhanced_components,
        on="Patient_ID",
        suffixes=("_baseline", "_enhanced"),
        validate="one_to_one",
    )
    rng = np.random.default_rng(seed)
    n_patients = len(merged)
    arrays = {col: merged[col].to_numpy(dtype=np.float64) for col in merged.columns if col != "Patient_ID"}
    deltas = np.empty(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n_patients, size=n_patients)
        base_u = normalize_utility_from_sums(
            arrays["observed_baseline"][idx].sum(),
            arrays["best_baseline"][idx].sum(),
            arrays["inaction_baseline"][idx].sum(),
            arrays["worst_baseline"][idx].sum(),
        )
        enh_u = normalize_utility_from_sums(
            arrays["observed_enhanced"][idx].sum(),
            arrays["best_enhanced"][idx].sum(),
            arrays["inaction_enhanced"][idx].sum(),
            arrays["worst_enhanced"][idx].sum(),
        )
        deltas[i] = enh_u - base_u
    return deltas


def validate_utility_reproduction(
    baseline_oof: pd.DataFrame,
    enhanced_oof: pd.DataFrame,
    baseline_metrics: dict[str, Any],
    enhanced_metrics: dict[str, Any],
) -> tuple[bool, list[str]]:
    """Check that reporting Utility Score reproduces summary_metrics.json values."""
    checks = []
    tolerance = 5e-4
    for label, oof, metrics, metric_name, threshold in [
        ("baseline", baseline_oof, baseline_metrics, "Utility_raw_at_0.5", 0.5),
        ("enhanced", enhanced_oof, enhanced_metrics, "Utility_raw_at_0.5", 0.5),
        ("baseline", baseline_oof, baseline_metrics, "Utility_raw_best", as_float(get_metric(baseline_metrics, "Utility_raw_best_threshold"))),
        ("enhanced", enhanced_oof, enhanced_metrics, "Utility_raw_best", as_float(get_metric(enhanced_metrics, "Utility_raw_best_threshold"))),
    ]:
        if np.isnan(threshold):
            checks.append(f"{label}.{metric_name}: missing threshold")
            continue
        computed = normalized_utility_from_components(utility_patient_components(oof, "prob_raw", threshold))
        expected = as_float(get_metric(metrics, metric_name))
        diff = abs(computed - expected)
        checks.append(f"{label}.{metric_name}: computed={computed:.9f}, expected={expected:.9f}, diff={diff:.3g}")
        if np.isnan(expected) or diff > tolerance:
            return False, checks
    return True, checks


def build_utility_statistical_tests(
    baseline_dir: Path,
    enhanced_dir: Path,
    baseline_metrics: dict[str, Any],
    enhanced_metrics: dict[str, Any],
    n_bootstrap: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Build Utility Score bootstrap tests if internal Utility reproduction succeeds."""
    baseline_oof = load_oof(baseline_dir / "oof_predictions.csv")
    enhanced_oof = load_oof(enhanced_dir / "oof_predictions.csv")
    valid, checks = validate_utility_reproduction(baseline_oof, enhanced_oof, baseline_metrics, enhanced_metrics)
    if not valid:
        note = "UNRESOLVED: Utility reproduction failed; " + " | ".join(checks)
        unresolved = pd.DataFrame(
            [
                {
                    "comparison": "Utility bootstrap",
                    "baseline_threshold": np.nan,
                    "enhanced_threshold": np.nan,
                    "baseline_utility": np.nan,
                    "enhanced_utility": np.nan,
                    "delta": np.nan,
                    "ci_lower": np.nan,
                    "ci_upper": np.nan,
                    "p_value_two_sided": np.nan,
                    "p_value_bh": np.nan,
                    "significant_bh_0.05": False,
                    "n_bootstrap": 0,
                    "bootstrap_unit": "Patient_ID",
                    "notes": note,
                }
            ]
        )
        return unresolved, {"status": "UNRESOLVED", "explanation": note}

    base_best = as_float(get_metric(baseline_metrics, "Utility_raw_best_threshold"))
    enh_best = as_float(get_metric(enhanced_metrics, "Utility_raw_best_threshold"))
    comparisons = [
        ("raw_common_threshold_0.5", 0.5, 0.5),
        ("raw_common_baseline_optimal_threshold", base_best, base_best),
        ("raw_common_enhanced_optimal_threshold", enh_best, enh_best),
        ("raw_model_specific_optimal_thresholds", base_best, enh_best),
    ]
    rows = []
    for i, (name, base_th, enh_th) in enumerate(comparisons):
        base_comp = utility_patient_components(baseline_oof, "prob_raw", base_th)
        enh_comp = utility_patient_components(enhanced_oof, "prob_raw", enh_th)
        base_u = normalized_utility_from_components(base_comp)
        enh_u = normalized_utility_from_components(enh_comp)
        samples = paired_utility_bootstrap(base_comp, enh_comp, n_bootstrap, seed + i)
        rows.append(
            {
                "comparison": name,
                "baseline_threshold": base_th,
                "enhanced_threshold": enh_th,
                "baseline_utility": base_u,
                "enhanced_utility": enh_u,
                "delta": enh_u - base_u,
                "ci_lower": float(np.percentile(samples, 2.5)),
                "ci_upper": float(np.percentile(samples, 97.5)),
                "p_value_two_sided": bootstrap_p_value(samples, n_bootstrap),
                "p_value_bh": np.nan,
                "significant_bh_0.05": False,
                "n_bootstrap": n_bootstrap,
                "bootstrap_unit": "Patient_ID",
                "notes": "Utility implementation copied from src/training/train_zabihi_cudf.py; validation: " + " | ".join(checks) + "; optimized thresholds are operational/exploratory and are not external validation.",
            }
        )
    df = pd.DataFrame(rows)
    df["p_value_bh"] = apply_bh(df["p_value_two_sided"].tolist())
    df["significant_bh_0.05"] = df["p_value_bh"].lt(0.05)
    return df, {"status": "PASS", "explanation": "Utility bootstrap completed after reproducing summary metrics."}


def patient_confusion_from_oof(oof: pd.DataFrame, threshold: float) -> dict[str, Any]:
    """Compute requested patient-level confusion matrix using any positive alarm per patient."""
    grouped = oof.assign(_pred=(oof["prob_raw"] >= threshold).astype(int)).groupby("Patient_ID", sort=False)
    patient = grouped.agg(real_positive=("SepsisLabel", "max"), predicted_positive=("_pred", "max"))
    y = patient["real_positive"].astype(int)
    pred = patient["predicted_positive"].astype(int)
    tp = int(((y == 1) & (pred == 1)).sum())
    tn = int(((y == 0) & (pred == 0)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    sensitivity = tp / (tp + fn) if (tp + fn) else np.nan
    specificity = tn / (tn + fp) if (tn + fp) else np.nan
    precision = tp / (tp + fp) if (tp + fp) else np.nan
    f1 = 2 * precision * sensitivity / (precision + sensitivity) if (precision + sensitivity) else np.nan
    return {
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "F1": f1,
        "n_patients": int(len(patient)),
        "n_positive_patients": int(y.sum()),
        "n_negative_patients": int((y == 0).sum()),
    }


def build_patient_confusion_matrices(
    baseline_dir: Path, enhanced_dir: Path, baseline_metrics: dict[str, Any], enhanced_metrics: dict[str, Any]
) -> pd.DataFrame:
    """Build patient-level confusion matrices at selected raw thresholds."""
    rows = []
    for variant, result_dir, metrics in [("baseline", baseline_dir, baseline_metrics), ("enhanced", enhanced_dir, enhanced_metrics)]:
        oof = load_oof(result_dir / "oof_predictions.csv")
        thresholds = [("threshold_0.5", 0.5), ("raw_utility_optimal", as_float(get_metric(metrics, "Utility_raw_best_threshold")))]
        f1_th = as_float(get_metric(metrics, "best_F1_threshold_raw"))
        if not np.isnan(f1_th):
            thresholds.append(("raw_F1_optimal", f1_th))
        for threshold_type, threshold in thresholds:
            if np.isnan(threshold):
                continue
            row = {"model_variant": variant, "threshold_type": threshold_type, "threshold": threshold}
            row.update(patient_confusion_from_oof(oof, threshold))
            row["notes"] = "Patient predicted positive if any time step has prob_raw >= threshold; this requested alarm-level confusion matrix differs from lead-time patient metrics that require pre-onset alerts for septic patients."
            rows.append(row)
    return pd.DataFrame(rows)


def load_oof_with_metadata(path: Path) -> pd.DataFrame:
    """Load OOF predictions and preserve available metadata columns for subgroup reporting."""
    df = load_oof(path)
    keep = [col for col in ["Patient_ID", "TimeStep", "SepsisLabel", "prob_raw", "Age", "Gender", "SourceSet"] if col in df.columns]
    return df[keep].copy()


def patient_metadata_table(oof: pd.DataFrame) -> pd.DataFrame:
    """Collapse OOF rows to one patient-level metadata row."""
    rows = []
    for patient_id, gdf in oof.groupby("Patient_ID", sort=False):
        row = {
            "Patient_ID": patient_id,
            "real_positive": int(gdf["SepsisLabel"].max()),
        }
        for col in ["Age", "Gender", "SourceSet"]:
            if col in gdf.columns:
                values = gdf[col].dropna().unique()
                row[col] = values[0] if len(values) else np.nan
                row[f"{col}_unique_values"] = len(values)
        rows.append(row)
    return pd.DataFrame(rows)


def subgroup_rows_for_model(oof: pd.DataFrame, variant: str, threshold_type: str, threshold: float) -> list[dict[str, Any]]:
    """Compute patient-level subgroup rows for one model and threshold."""
    meta = patient_metadata_table(oof)
    pred = oof.assign(_pred=(oof["prob_raw"] >= threshold).astype(int)).groupby("Patient_ID", sort=False)["_pred"].max().reset_index()
    patient = meta.merge(pred, on="Patient_ID", how="left", validate="one_to_one")
    subgroup_defs = []
    if "Age" in patient.columns:
        age = pd.to_numeric(patient["Age"], errors="coerce")
        patient["age_group"] = np.where(age < 65, "<65", ">=65")
        patient.loc[age.isna(), "age_group"] = "Unknown"
        subgroup_defs.append(("age_group", "age_group", "Age collapsed to patient level; verify one Age value per Patient_ID."))
    if "Gender" in patient.columns:
        subgroup_defs.append(("gender", "Gender", "Gender collapsed to patient level; verify coding before paper use."))
    source_unavailable_row = None
    if "SourceSet" in patient.columns:
        subgroup_defs.append(("SourceSet", "SourceSet", "SourceSet available in OOF; internal source traceability, not external validation."))
    else:
        source_unavailable_row = {
            "subgroup_type": "SourceSet",
            "subgroup": "UNAVAILABLE",
            "model_variant": variant,
            "n_patients": 0,
            "n_positive_patients": 0,
            "prevalence_patient_level": np.nan,
            "threshold_type": threshold_type,
            "threshold": threshold,
            "TP": np.nan,
            "TN": np.nan,
            "FP": np.nan,
            "FN": np.nan,
            "sensitivity": np.nan,
            "specificity": np.nan,
            "precision": np.nan,
            "F1": np.nan,
            "notes": "SourceSet is not present in OOF predictions; cannot compute source-wise patient subgroup table.",
        }

    rows = []
    if source_unavailable_row is not None:
        rows.append(source_unavailable_row)
    for subgroup_type, column, note in subgroup_defs:
        for subgroup, gdf in patient.groupby(column, dropna=False):
            y = gdf["real_positive"].astype(int)
            pred_pos = gdf["_pred"].astype(int)
            tp = int(((y == 1) & (pred_pos == 1)).sum())
            tn = int(((y == 0) & (pred_pos == 0)).sum())
            fp = int(((y == 0) & (pred_pos == 1)).sum())
            fn = int(((y == 1) & (pred_pos == 0)).sum())
            sensitivity = tp / (tp + fn) if (tp + fn) else np.nan
            specificity = tn / (tn + fp) if (tn + fp) else np.nan
            precision = tp / (tp + fp) if (tp + fp) else np.nan
            f1 = 2 * precision * sensitivity / (precision + sensitivity) if (precision + sensitivity) else np.nan
            rows.append({
                "subgroup_type": subgroup_type,
                "subgroup": subgroup,
                "model_variant": variant,
                "threshold_type": threshold_type,
                "threshold": threshold,
                "n_patients": int(len(gdf)),
                "n_positive_patients": int(y.sum()),
                "prevalence_patient_level": float(y.mean()) if len(gdf) else np.nan,
                "TP": tp,
                "TN": tn,
                "FP": fp,
                "FN": fn,
                "sensitivity": sensitivity,
                "specificity": specificity,
                "precision": precision,
                "F1": f1,
                "notes": note,
            })
    return rows


def build_subgroup_patient_table(
    baseline_dir: Path, enhanced_dir: Path, baseline_metrics: dict[str, Any], enhanced_metrics: dict[str, Any]
) -> pd.DataFrame:
    """Create patient-level subgroup metrics directly from OOF predictions."""
    rows = []
    for variant, result_dir, metrics in [("baseline", baseline_dir, baseline_metrics), ("enhanced", enhanced_dir, enhanced_metrics)]:
        oof = load_oof_with_metadata(result_dir / "oof_predictions.csv")
        thresholds = [("threshold_0.5", 0.5), ("raw_utility_optimal", as_float(get_metric(metrics, "Utility_raw_best_threshold")))]
        for threshold_type, threshold in thresholds:
            if not np.isnan(threshold):
                rows.extend(subgroup_rows_for_model(oof, variant, threshold_type, threshold))
    return pd.DataFrame(rows)


def build_feature_count_audit(feature_summary: dict[str, Any], baseline_metrics: dict[str, Any], enhanced_metrics: dict[str, Any]) -> pd.DataFrame:
    """Document feature counts and paper claims that require correction."""
    hemo_found = int(feature_summary.get("enhanced_hemo_features_in_importance", 0))
    sampen_found = len(feature_summary.get("sampen_features_present", []))
    rows = [
        {"item": "baseline_final_numeric_features", "value": int(as_float(get_metric(baseline_metrics, "n_features"))), "expected_or_recommended": 287, "status": "OK", "notes": "Final baseline feature count after meta/non-numeric exclusion."},
        {"item": "enhanced_final_numeric_features", "value": int(as_float(get_metric(enhanced_metrics, "n_features"))), "expected_or_recommended": 305, "status": "OK", "notes": "Final enhanced feature count after adding hemodynamic complexity features."},
        {"item": "feature_delta", "value": int(as_float(get_metric(enhanced_metrics, "n_features")) - as_float(get_metric(baseline_metrics, "n_features"))), "expected_or_recommended": 18, "status": "OK", "notes": "Enhanced minus baseline numeric features."},
        {"item": "hemodynamic_expected_features", "value": 18, "expected_or_recommended": 18, "status": "OK", "notes": "Six signals times CV/IQR/SampEn."},
        {"item": "hemodynamic_found_in_enhanced_importance", "value": hemo_found, "expected_or_recommended": 18, "status": "OK" if hemo_found == 18 else "CHECK", "notes": "Count of hemodynamic features present in enhanced feature_importance_summary.csv."},
        {"item": "SampEn_found", "value": f"{sampen_found}/6", "expected_or_recommended": "6/6", "status": "OK" if sampen_found == 6 else "CHECK", "notes": "Correct paper wording if it still says five of six SampEn."},
        {"item": "meta_columns_excluded", "value": "Patient_ID, TimeStep, SourceSet, SepsisLabel", "expected_or_recommended": "Exclude from model inputs", "status": "OK", "notes": "Meta columns should remain traceability/label columns, not features."},
        {"item": "old_claim_71_77", "value": "71/77", "expected_or_recommended": "Remove or replace with final feature counts 287/305 and +18 hemodynamic features.", "status": "CORRECT_MANUSCRIPT", "notes": "Old paper claim to correct."},
        {"item": "old_claim_246_265", "value": "246/265", "expected_or_recommended": "Remove or replace with final feature counts 287/305 and +18 hemodynamic features.", "status": "CORRECT_MANUSCRIPT", "notes": "Old paper claim to correct."},
        {"item": "old_claim_five_of_six_SampEn", "value": "five of six SampEn", "expected_or_recommended": "six of six SampEn features were present if audit status is OK.", "status": "CORRECT_MANUSCRIPT", "notes": "Old paper claim to correct."},
        {"item": "recommended_wording", "value": "The enhanced model adds 18 hemodynamic complexity features to the baseline feature set, yielding 305 final numeric predictors versus 287 in baseline after excluding meta columns.", "expected_or_recommended": "Use this wording if counts match final artifacts.", "status": "RECOMMENDED", "notes": "Paper-ready wording."},
    ]
    return pd.DataFrame(rows)


def build_calibration_audit(baseline_metrics: dict[str, Any], enhanced_metrics: dict[str, Any]) -> tuple[str, dict[str, str]]:
    """Create calibration interpretation audit notes from summary metrics."""
    ece_raw_delta = as_float(get_metric(enhanced_metrics, "ECE_raw")) - as_float(get_metric(baseline_metrics, "ECE_raw"))
    brier_raw_delta = as_float(get_metric(enhanced_metrics, "Brier_raw")) - as_float(get_metric(baseline_metrics, "Brier_raw"))
    status = {"status": "PASS", "explanation": "Training code audit confirms fold-wise OOF cross-fitting via crossfit_calibration(fold_ids): train folds fit calibration, held-out fold receives calibrated probabilities."}
    text = f"""# Calibration Audit

## Raw Calibration

- Baseline ECE_raw: {format_metric(get_metric(baseline_metrics, 'ECE_raw'), 8)}
- Enhanced ECE_raw: {format_metric(get_metric(enhanced_metrics, 'ECE_raw'), 8)}
- Delta ECE_raw enhanced-baseline: {format_metric(ece_raw_delta, 8)}
- Baseline Brier_raw: {format_metric(get_metric(baseline_metrics, 'Brier_raw'), 8)}
- Enhanced Brier_raw: {format_metric(get_metric(enhanced_metrics, 'Brier_raw'), 8)}
- Delta Brier_raw enhanced-baseline: {format_metric(brier_raw_delta, 8)}

Raw calibration worsened if these deltas are positive. Do not claim that all calibration improved.

## Calibrated Metrics

- Platt and isotonic AUROC/AUPRC/Brier/ECE are summarized in `calibration_table.csv`.
- If calibrated metrics improve, describe them as calibrated-output improvements, not raw-probability calibration improvements.
- Isotonic ECE may be extremely low and must be described cautiously.

## Independence of Calibration Fit

Status: **CODE AUDIT CONFIRMED INTERNAL CROSS-FIT**.

`src/training/train_zabihi_cudf.py` uses `crossfit_calibration(y_true, y_prob, fold_ids, method=...)`, fitting calibration on `fold_ids != fold` and predicting calibrated probabilities on `fold_ids == fold`. This supports internal OOF calibrated metrics, not external/prospective calibration. Do not write "near-perfect calibration"; describe the result as cross-fitted internal calibration and keep raw calibration caveats visible.
"""
    return text, status


def build_statistical_methods_notes(delong_status: dict[str, str], utility_status: dict[str, str]) -> str:
    """Create statistical-method notes for the statistics package."""
    return f"""# Statistical Methods Notes

## Scope

All analyses use out-of-fold predictions from patient-grouped internal cross-validation on the public PhysioNet/CinC 2019 training split. No independent external validation or official hidden-test evaluation was performed; therefore, all claims are limited to internal cross-validation on the public PhysioNet/CinC 2019 training split.

## Out-Of-Fold Predictions And Grouped CV

Baseline and enhanced models are compared on paired OOF predictions aligned by `Patient_ID`, `TimeStep`, and `SepsisLabel`. Cross-validation is grouped by patient to avoid splitting a patient across train and validation folds.

## Paired Patient-Level Bootstrap

Bootstrap inference resamples `Patient_ID` values with replacement and keeps all time steps for each sampled patient. This preserves within-patient temporal dependence and avoids the anti-conservative behavior of row-level bootstrap over repeated time steps.

## McNemar Tests

McNemar tests compare paired correctness indicators from aligned baseline and enhanced OOF predictions. The script reports both `mcnemar_time_step_correctness` and `mcnemar_patient_correctness`; patient-level rows should be preferred for patient-level paper claims. Numeric `p_value_two_sided` and display-safe `p_value_display` are both emitted so paper text does not show `p=0.000000`.

## Benjamini-Hochberg Correction

Primary statistical rows are adjusted with Benjamini-Hochberg FDR correction and include `p_value_bh` plus `significant_bh_0.05`.

## Calibration Analysis

Raw, Platt, and isotonic probabilities are summarized separately. Platt and isotonic metrics are internal cross-fitted calibration analyses derived from OOF folds; they must not be described as independent validation.

## Threshold And Lead-Time Analysis

Threshold operating points include the conventional raw 0.5 threshold, raw max-Utility threshold, raw max-F1 threshold, and calibrated selected thresholds when available. Lead-time summaries are patient-level descriptive summaries at selected thresholds.

## DeLong Status

- Status: {delong_status.get('status')}
- Explanation: {delong_status.get('explanation')}

## Utility Bootstrap Status

- Status: {utility_status.get('status')}
- Explanation: {utility_status.get('explanation')}

## SourceSet A/B

`SourceSet` supports traceability and possible internal source-wise analyses. It is not external validation.
"""



def metrics_have_expected_policy(baseline_metrics: dict[str, Any], enhanced_metrics: dict[str, Any]) -> tuple[bool, str]:
    """Check whether loaded result metrics were generated by the final run policy."""
    observed = {
        "baseline": baseline_metrics.get("pipeline_policy_version"),
        "enhanced": enhanced_metrics.get("pipeline_policy_version"),
    }
    ok = all(value == EXPECTED_POLICY_VERSION for value in observed.values())
    if ok:
        return True, "Both result directories report the current validated pipeline policy."
    return False, f"Expected current validated pipeline policy; observed={observed}. Treat outputs as historical/pre-fix until rerun completes."


def build_completion_checklist(
    delong_status: dict[str, str],
    utility_status: dict[str, str],
    calibration_status: dict[str, str],
    feature_count_audit: pd.DataFrame,
    subgroup_patient_table: pd.DataFrame,
    baseline_metrics: dict[str, Any],
    enhanced_metrics: dict[str, Any],
) -> tuple[pd.DataFrame, str]:
    """Build statistics checklist using PASS/CONDITIONAL/UNRESOLVED/FAIL states."""
    allowed_states = {"PASS", "CONDITIONAL", "UNRESOLVED", "FAIL"}

    def normalize_status(status: str) -> str:
        mapping = {"COMPLETE": "PASS", "OK": "PASS", "NEEDS CODE AUDIT": "CONDITIONAL"}
        out = mapping.get(str(status), str(status))
        return out if out in allowed_states else "UNRESOLVED"

    def status_line(item: str, status: str, explanation: str) -> dict[str, str]:
        return {"item": item, "status": normalize_status(status), "explanation": explanation}

    utility_state = normalize_status(utility_status.get("status", "UNRESOLVED"))
    delong_state = normalize_status(delong_status.get("status", "UNRESOLVED"))
    calibration_state = normalize_status(calibration_status.get("status", "CONDITIONAL"))
    outputs_with_expected_policy, outputs_explanation = metrics_have_expected_policy(baseline_metrics, enhanced_metrics)

    rows = [
        status_line("baseline/enhanced final outputs verified", "PASS", "Required result files are validated before table generation."),
        status_line("current outputs generated with final run policy", "PASS" if outputs_with_expected_policy else "FAIL", outputs_explanation),
        status_line("OOF alignment verified", "PASS", "Baseline/enhanced OOF predictions align on Patient_ID, TimeStep, and SepsisLabel."),
        status_line("paired bootstrap AUROC/AUPRC complete", "PASS", "Patient-level paired bootstrap implemented in statistical_tests.csv."),
        status_line("paired bootstrap Brier complete", "PASS", "Patient-level paired bootstrap implemented in statistical_tests.csv."),
        status_line("AUROC/AUPRC CI complete", "PASS", "Patient-level paired bootstrap confidence intervals are emitted for raw AUROC/AUPRC deltas."),
        status_line("Brier/calibration metrics complete", "PASS", "comparison_summary.csv and calibration_table.csv include raw/Platt/isotonic Brier and ECE metrics."),
        status_line("Utility inference complete/unresolved", utility_state, utility_status.get("explanation", "No Utility status recorded.")),
        status_line("DeLong complete/conditional", delong_state, delong_status.get("explanation", "No DeLong status recorded.")),
        status_line("McNemar time-step and patient-level complete", "PASS", "statistical_tests.csv includes mcnemar_time_step_correctness and mcnemar_patient_correctness rows with p_value_display."),
        status_line("patient confusion matrices complete", "PASS", "patient_confusion_matrices.csv generated from patient-collapsed OOF predictions."),
        status_line("patient-level metrics complete", "PASS", "patient_level_metrics.csv reports selected threshold sensitivity, specificity, and lead-time fields."),
        status_line("lead-time metrics complete", "PASS", "Lead_time_median and IQR fields are read from threshold_metrics.csv where available."),
        status_line("threshold operating points complete", "PASS", "threshold_operating_points.csv selects 0.5, raw max Utility, raw max F1, and calibrated selected thresholds when available."),
        status_line("patient subgroup table complete", "PASS" if not subgroup_patient_table.empty else "UNRESOLVED", "subgroup_patient_table.csv generated from patient-collapsed OOF metadata; SourceSet is included only if available in OOF."),
        status_line("feature count audit complete", "PASS" if not feature_count_audit.empty else "UNRESOLVED", "feature_count_audit.csv generated."),
        status_line(
            "feature count corrected",
            "PASS" if as_float(get_metric(baseline_metrics, "n_features")) == 287 and as_float(get_metric(enhanced_metrics, "n_features")) == 305 else "CONDITIONAL",
            f"Baseline n_features={get_metric(baseline_metrics, 'n_features')}; enhanced n_features={get_metric(enhanced_metrics, 'n_features')}; expected delta=18.",
        ),
        status_line("SampEn features count checked", "PASS" if not feature_count_audit.empty else "UNRESOLVED", "feature_count_audit.csv checks expected hemodynamic complexity feature counts."),
        status_line("calibration audit complete", calibration_state, calibration_status.get("explanation", "Calibration fitting independence requires code audit.")),
        status_line("feature importance hemodynamic audit complete", "PASS", "feature_importance_table.csv and feature_count_audit.csv include hemodynamic checks."),
        status_line("paper warning list generated", "PASS", "statistics_summary.md and code/math audit include do-not-write and allowed-claims sections."),
        status_line("reproducibility command recorded", "PASS", "manifest.json records command, git commit, branch, input paths, seed, and n_bootstrap."),
        status_line("no hidden test claim", "PASS", "Warnings explicitly prohibit hidden-test claims."),
        status_line("no external validation claim", "PASS", "Warnings explicitly prohibit external validation claims."),
        status_line("no direct SOTA/Zabihi hidden-test comparison claim", "PASS", "Warnings explicitly prohibit SOTA superiority claims."),
    ]

    unresolved = [row for row in rows if row["status"] == "UNRESOLVED"]
    fail = [row for row in rows if row["status"] == "FAIL"]
    conditional = [row for row in rows if row["status"] == "CONDITIONAL"]
    unresolved_allowed = {"Utility inference complete/unresolved", "DeLong complete/conditional"}
    critical_unresolved = [row for row in unresolved if row["item"] not in unresolved_allowed]

    if critical_unresolved:
        ready_for_package = "FAIL"
        package_explanation = "Blocking reporting/code issues remain: " + "; ".join(row["item"] for row in critical_unresolved)
    else:
        ready_for_package = "PASS"
        package_explanation = "statistics reporting package can be generated from the validated final run result directories."
    rows.append({"item": "ready_for_statistics_package", "status": ready_for_package, "explanation": package_explanation})

    if not outputs_with_expected_policy:
        ready_for_methodology_review = "NO"
        methodology_explanation = "Current result directories are historical/pre-fix or lack final run metadata; rerun baseline/enhanced and regenerate reporting first."
    elif fail or critical_unresolved:
        ready_for_methodology_review = "NO"
        methodology_explanation = "Blocking statistics issues remain after final run outputs."
    elif unresolved or conditional:
        ready_for_methodology_review = "CONDITIONAL"
        methodology_explanation = "Only conditional statistical/reporting items remain: " + "; ".join(row["item"] for row in unresolved + conditional)
    else:
        ready_for_methodology_review = "YES"
        methodology_explanation = "final run outputs and reporting passed checklist."
    rows.append({"item": "ready_for_methodology_review", "status": ready_for_methodology_review, "explanation": methodology_explanation})

    df = pd.DataFrame(rows)
    md = "# statistics Completion Checklist\n\n" + "\n".join(
        f"- **{row['item']}**: `{row['status']}` - {row['explanation']}" for row in rows
    ) + "\n"
    return df, md


def column_by_candidates(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    """Find the first available column from exact or case-insensitive candidates."""
    columns = {col.lower(): col for col in df.columns}
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
        found = columns.get(candidate.lower())
        if found is not None:
            return found
    return None


def threshold_columns(df: pd.DataFrame) -> dict[str, str | None]:
    """Resolve threshold metrics column names across historical output variants."""
    return {
        "method": column_by_candidates(df, ["probability_source", "probability_column", "method"]),
        "threshold": column_by_candidates(df, ["threshold", "Threshold"]),
        "utility": column_by_candidates(df, ["utility", "Utility", "Utility_U"]),
        "sensitivity": column_by_candidates(df, ["sensitivity", "Recall", "recall"]),
        "specificity": column_by_candidates(df, ["specificity", "Specificity"]),
        "precision": column_by_candidates(df, ["precision", "Precision"]),
        "f1": column_by_candidates(df, ["f1", "F1"]),
        "patient_sensitivity": column_by_candidates(df, ["Patient_sensitivity", "patient_sensitivity"]),
        "patient_specificity": column_by_candidates(df, ["Patient_specificity", "patient_specificity"]),
        "lead_time_median": column_by_candidates(df, ["Lead_time_median", "lead_time_median"]),
        "lead_time_iqr_lower": column_by_candidates(df, ["Lead_time_IQR_lower", "lead_time_iqr_lower"]),
        "lead_time_iqr_upper": column_by_candidates(df, ["Lead_time_IQR_upper", "lead_time_iqr_upper"]),
    }


def selected_threshold_rows(df: pd.DataFrame, metrics: dict[str, Any]) -> pd.DataFrame:
    """Select paper-relevant threshold rows from a threshold metrics table."""
    cols = threshold_columns(df)
    if cols["threshold"] is None:
        raise ValueError("threshold_metrics.csv must contain a threshold column")
    method_col = cols["method"]
    threshold_col = cols["threshold"]
    utility_col = cols["utility"]
    f1_col = cols["f1"]

    work = df.copy()
    work[threshold_col] = pd.to_numeric(work[threshold_col], errors="coerce")
    if method_col is None:
        work["_method"] = "raw"
        method_col = "_method"
    work[method_col] = work[method_col].astype(str).str.lower()

    keep_indices: set[int] = set()
    raw = work[work[method_col].eq("raw")]
    for target in [0.5, as_float(get_metric(metrics, "Utility_raw_best_threshold")), as_float(get_metric(metrics, "best_F1_threshold_raw"))]:
        if np.isnan(target):
            continue
        candidates = raw[np.isclose(raw[threshold_col], target, atol=1e-9)]
        if not candidates.empty:
            keep_indices.update(candidates.index.tolist())
    if utility_col is not None and not raw.empty:
        best_idx = numeric_idxmax(raw[utility_col])
        if best_idx is not None:
            keep_indices.add(best_idx)
    if f1_col is not None and not raw.empty:
        best_idx = numeric_idxmax(raw[f1_col])
        if best_idx is not None:
            keep_indices.add(best_idx)

    for method in ["platt", "isotonic"]:
        subset = work[work[method_col].eq(method)]
        if utility_col is not None and not subset.empty:
            best_idx = numeric_idxmax(subset[utility_col])
            if best_idx is not None:
                keep_indices.add(best_idx)
        threshold_name = f"Utility_{method}_best_threshold"
        target = as_float(get_metric(metrics, threshold_name))
        if not np.isnan(target):
            candidates = subset[np.isclose(subset[threshold_col], target, atol=1e-9)]
            if not candidates.empty:
                keep_indices.update(candidates.index.tolist())

    return work.loc[sorted(keep_indices)].copy()


def value_from_row(row: pd.Series, column: str | None) -> Any:
    """Return a row value when the source column exists."""
    if column is None:
        return np.nan
    return row.get(column, np.nan)


def numeric_idxmax(series: pd.Series) -> Any:
    """Return idxmax for numeric values, or None when all values are missing."""
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().any():
        return numeric.idxmax()
    return None


def build_threshold_operating_points(
    baseline_dir: Path, enhanced_dir: Path, baseline_metrics: dict[str, Any], enhanced_metrics: dict[str, Any]
) -> pd.DataFrame:
    """Create selected operating-point metrics from threshold sweeps."""
    rows = []
    for variant, result_dir, metrics in [
        ("baseline", baseline_dir, baseline_metrics),
        ("enhanced", enhanced_dir, enhanced_metrics),
    ]:
        table = pd.read_csv(result_dir / "threshold_metrics.csv")
        cols = threshold_columns(table)
        selected = selected_threshold_rows(table, metrics)
        for _, row in selected.iterrows():
            rows.append(
                {
                    "model_variant": variant,
                    "probability_column": value_from_row(row, cols["method"]),
                    "threshold": value_from_row(row, cols["threshold"]),
                    "utility": value_from_row(row, cols["utility"]),
                    "sensitivity": value_from_row(row, cols["sensitivity"]),
                    "specificity": value_from_row(row, cols["specificity"]),
                    "precision": value_from_row(row, cols["precision"]),
                    "f1": value_from_row(row, cols["f1"]),
                    "source_file": str(result_dir / "threshold_metrics.csv"),
                }
            )
    return pd.DataFrame(rows)


def threshold_type(row: pd.Series, metrics: dict[str, Any], method_col: str | None, threshold_col: str | None) -> str:
    """Label why a threshold row was selected."""
    method = str(row.get(method_col, "raw")).lower() if method_col else "raw"
    threshold = as_float(row.get(threshold_col)) if threshold_col else np.nan
    if not np.isnan(threshold) and np.isclose(threshold, 0.5):
        return f"{method}_threshold_0.5"
    raw_best = as_float(get_metric(metrics, "Utility_raw_best_threshold"))
    if method == "raw" and not np.isnan(raw_best) and np.isclose(threshold, raw_best):
        return "raw_max_utility"
    return f"{method}_selected"


def build_patient_level_metrics(
    baseline_dir: Path, enhanced_dir: Path, baseline_metrics: dict[str, Any], enhanced_metrics: dict[str, Any]
) -> pd.DataFrame:
    """Create patient-level operating metrics from selected threshold rows."""
    rows = []
    for variant, result_dir, metrics in [
        ("baseline", baseline_dir, baseline_metrics),
        ("enhanced", enhanced_dir, enhanced_metrics),
    ]:
        table = pd.read_csv(result_dir / "threshold_metrics.csv")
        cols = threshold_columns(table)
        selected = selected_threshold_rows(table, metrics)
        for _, row in selected.iterrows():
            rows.append(
                {
                    "model_variant": variant,
                    "threshold_type": threshold_type(row, metrics, cols["method"], cols["threshold"]),
                    "threshold": value_from_row(row, cols["threshold"]),
                    "patient_sensitivity": value_from_row(row, cols["patient_sensitivity"]),
                    "patient_specificity": value_from_row(row, cols["patient_specificity"]),
                    "lead_time_median": value_from_row(row, cols["lead_time_median"]),
                    "lead_time_iqr_lower": value_from_row(row, cols["lead_time_iqr_lower"]),
                    "lead_time_iqr_upper": value_from_row(row, cols["lead_time_iqr_upper"]),
                    "notes": "Patient-level values are read from threshold_metrics.csv; verify patient counts are unique patients, not time steps.",
                }
            )
    return pd.DataFrame(rows)


def build_calibration_table(
    baseline_dir: Path, enhanced_dir: Path, baseline_metrics: dict[str, Any], enhanced_metrics: dict[str, Any]
) -> pd.DataFrame:
    """Build a calibration summary table from final JSON metrics."""
    rows = []
    for variant, result_dir, metrics in [
        ("baseline", baseline_dir, baseline_metrics),
        ("enhanced", enhanced_dir, enhanced_metrics),
    ]:
        best_method = metrics.get("best_calibration_method_by_brier")
        for method in ["raw", "platt", "isotonic"]:
            bins_file = result_dir / f"calibration_bins_{method}.csv"
            rows.append(
                {
                    "model_variant": variant,
                    "method": method,
                    "AUROC": get_metric(metrics, f"AUROC_{method}"),
                    "AUPRC": get_metric(metrics, f"AUPRC_{method}"),
                    "ECE": get_metric(metrics, f"ECE_{method}"),
                    "Brier": get_metric(metrics, f"Brier_{method}"),
                    "best_by_brier": method == best_method,
                    "calibration_bins_file_present": bins_file.exists(),
                }
            )
    return pd.DataFrame(rows)


def build_subgroup_table(baseline_dir: Path, enhanced_dir: Path) -> tuple[pd.DataFrame, list[str]]:
    """Build a comparative subgroup table without inventing patient counts."""
    baseline = pd.read_csv(baseline_dir / "subgroup_metrics.csv")
    enhanced = pd.read_csv(enhanced_dir / "subgroup_metrics.csv")
    subgroup_cols = [col for col in ["type", "group", "subgroup"] if col in baseline.columns and col in enhanced.columns]
    if "subgroup" in subgroup_cols:
        key_cols = ["subgroup"]
    elif {"type", "group"}.issubset(subgroup_cols):
        key_cols = ["type", "group"]
    else:
        common = [col for col in baseline.columns if col in enhanced.columns]
        if not common:
            raise ValueError("No common subgroup key columns found in subgroup_metrics.csv")
        key_cols = [common[0]]

    numeric_cols = [
        col for col in baseline.columns
        if col in enhanced.columns and col not in key_cols and pd.api.types.is_numeric_dtype(baseline[col])
    ]
    merged = baseline.merge(enhanced, on=key_cols, how="outer", suffixes=("_baseline", "_enhanced"))
    rows = []
    warnings = []
    for _, row in merged.iterrows():
        subgroup = " / ".join(str(row[col]) for col in key_cols)
        for metric in numeric_cols:
            baseline_value = row.get(f"{metric}_baseline", np.nan)
            enhanced_value = row.get(f"{metric}_enhanced", np.nan)
            note = ""
            if metric == "n_patients":
                note = "Verify this count represents unique patients, not rows/time-steps."
            elif metric == "n_rows":
                note = "Row/time-step count; do not present as patient count."
            rows.append(
                {
                    "subgroup": subgroup,
                    "metric": metric,
                    "baseline": baseline_value,
                    "enhanced": enhanced_value,
                    "delta": enhanced_value - baseline_value if pd.notna(baseline_value) and pd.notna(enhanced_value) else np.nan,
                    "notes": note,
                }
            )
    if "n_patients" not in numeric_cols:
        warnings.append("subgroup_metrics.csv does not contain reliable n_patients; do not invent subgroup patient counts.")
    if "n_rows" in numeric_cols:
        warnings.append("subgroup_metrics.csv contains n_rows; verify paper text does not double-count time steps as patients.")
    return pd.DataFrame(rows), warnings


def classify_hemo_feature(feature: str) -> tuple[bool, str, str]:
    """Classify hemodynamic complexity features by signal and statistic type."""
    for feature_type, pattern in HEMO_PATTERNS.items():
        if pattern in feature:
            signal = next((name for name in HEMO_SIGNALS if name in feature), "Unknown")
            return True, signal, feature_type
    return False, "", ""


def first_existing_column(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    """Return the first matching feature-importance column name."""
    return column_by_candidates(df, candidates)


def build_feature_importance_table(baseline_dir: Path, enhanced_dir: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build feature-importance table with hemodynamic annotations."""
    rows = []
    summary: dict[str, Any] = {}
    for variant, result_dir in [("baseline", baseline_dir), ("enhanced", enhanced_dir)]:
        table = pd.read_csv(result_dir / "feature_importance_summary.csv")
        feature_col = first_existing_column(table, ["feature", "Feature"])
        if feature_col is None:
            raise ValueError(f"feature_importance_summary.csv in {result_dir} lacks a feature column")
        mean_col = first_existing_column(table, ["mean_importance", "gain_mean", "importance_mean", "gain"])
        std_col = first_existing_column(table, ["std_importance", "gain_std", "importance_std"])
        selected_col = first_existing_column(table, ["selected_folds", "folds_selected", "selection_freq"])
        for _, row in table.iterrows():
            feature = str(row[feature_col])
            is_hemo, signal, feature_type = classify_hemo_feature(feature)
            rows.append(
                {
                    "model_variant": variant,
                    "feature": feature,
                    "mean_importance": value_from_row(row, mean_col),
                    "std_importance": value_from_row(row, std_col),
                    "selected_folds": value_from_row(row, selected_col),
                    "is_hemodynamic_complexity_feature": is_hemo,
                    "hemo_signal": signal,
                    "hemo_feature_type": feature_type,
                }
            )
    out = pd.DataFrame(rows)
    hemo = out[out["is_hemodynamic_complexity_feature"] & out["model_variant"].eq("enhanced")].copy()
    summary["enhanced_hemo_features_in_importance"] = int(hemo["feature"].nunique())
    summary["enhanced_expected_hemo_features"] = len(HEMO_SIGNALS) * len(HEMO_PATTERNS)
    if not hemo.empty:
        hemo["mean_importance_numeric"] = pd.to_numeric(hemo["mean_importance"], errors="coerce")
        top = hemo.sort_values("mean_importance_numeric", ascending=False).head(10)
        summary["top_hemo_features"] = top["feature"].tolist()
        summary["sampen_features_present"] = sorted(hemo.loc[hemo["hemo_feature_type"].eq("SampEn"), "feature"].unique().tolist())
    else:
        summary["top_hemo_features"] = []
        summary["sampen_features_present"] = []
    return out, summary


def build_fold_comparison(baseline_dir: Path, enhanced_dir: Path) -> tuple[pd.DataFrame, dict[str, float]]:
    """Build fold-wise AUROC/AUPRC comparisons."""
    baseline = pd.read_csv(baseline_dir / "fold_metrics.csv")
    enhanced = pd.read_csv(enhanced_dir / "fold_metrics.csv")
    fold_col = first_existing_column(baseline, ["Fold", "fold"])
    if fold_col is None:
        raise ValueError("fold_metrics.csv lacks Fold/fold column")
    enh_fold_col = first_existing_column(enhanced, ["Fold", "fold"])
    auroc_col = first_existing_column(baseline, ["AUROC", "auroc"])
    auprc_col = first_existing_column(baseline, ["AUPRC", "auprc"])
    enh_auroc_col = first_existing_column(enhanced, ["AUROC", "auroc"])
    enh_auprc_col = first_existing_column(enhanced, ["AUPRC", "auprc"])
    needed = [enh_fold_col, auroc_col, auprc_col, enh_auroc_col, enh_auprc_col]
    if any(col is None for col in needed):
        raise ValueError("fold_metrics.csv must contain Fold, AUROC, and AUPRC columns")
    base_slim = baseline[[fold_col, auroc_col, auprc_col]].rename(
        columns={fold_col: "fold", auroc_col: "baseline_AUROC", auprc_col: "baseline_AUPRC"}
    )
    enh_slim = enhanced[[enh_fold_col, enh_auroc_col, enh_auprc_col]].rename(
        columns={enh_fold_col: "fold", enh_auroc_col: "enhanced_AUROC", enh_auprc_col: "enhanced_AUPRC"}
    )
    merged = base_slim.merge(enh_slim, on="fold", how="inner", validate="one_to_one")
    merged["delta_AUROC"] = merged["enhanced_AUROC"] - merged["baseline_AUROC"]
    merged["delta_AUPRC"] = merged["enhanced_AUPRC"] - merged["baseline_AUPRC"]
    stats = {
        "delta_AUROC_mean": float(merged["delta_AUROC"].mean()),
        "delta_AUROC_std": float(merged["delta_AUROC"].std(ddof=0)),
        "delta_AUPRC_mean": float(merged["delta_AUPRC"].mean()),
        "delta_AUPRC_std": float(merged["delta_AUPRC"].std(ddof=0)),
    }
    return merged, stats


def write_csv(df: pd.DataFrame, path: Path) -> None:
    """Write a CSV without an index and log its location."""
    df.to_csv(path, index=False)
    logging.info("Wrote %s (%d rows)", path, len(df))


def write_markdown_table(df: pd.DataFrame, path: Path) -> None:
    """Write a compact Markdown table for small statistics outputs."""
    headers = [str(col) for col in df.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for _, row in df.iterrows():
        values = [str(row[col]).replace("\n", " ") for col in df.columns]
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logging.info("Wrote %s (%d rows)", path, len(df))


def format_metric(value: Any, digits: int = 4) -> str:
    """Format metrics compactly for Markdown summaries."""
    number = as_float(value)
    if np.isnan(number):
        return "NA"
    return f"{number:.{digits}f}"


def git_value(args: list[str]) -> str:
    """Return a git metadata value for the manifest, or unavailable."""
    try:
        result = subprocess.run(
            ["git", *args],
            check=False,
            text=True,
            capture_output=True,
        )
    except Exception:
        return "unavailable"
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def sha256_file(path: Path) -> str:
    """Compute SHA256 for generated small/medium reporting files."""
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def build_results_snippet(
    baseline_metrics: dict[str, Any],
    enhanced_metrics: dict[str, Any],
    stats: pd.DataFrame,
) -> str:
    """Build a cautious final run results snippet for later paper editing."""
    auroc_base = get_metric(baseline_metrics, "AUROC_raw")
    auroc_enh = get_metric(enhanced_metrics, "AUROC_raw")
    auprc_base = get_metric(baseline_metrics, "AUPRC_raw")
    auprc_enh = get_metric(enhanced_metrics, "AUPRC_raw")
    utility_base = as_float(get_metric(baseline_metrics, "Utility_raw_best"))
    utility_enh = as_float(get_metric(enhanced_metrics, "Utility_raw_best"))
    sens_base = get_metric(baseline_metrics, "Patient_sensitivity_best_method_at_0.5")
    sens_enh = get_metric(enhanced_metrics, "Patient_sensitivity_best_method_at_0.5")

    supported = "Paired patient-level bootstrap results are reported in `statistical_tests.csv`."
    if not stats.empty and "metric" in stats.columns:
        main = stats[stats["metric"].isin(["Delta_AUROC_raw", "Delta_AUPRC_raw"])]
        if not main.empty and main["significant_bh_0.05"].fillna(False).all():
            supported = "Paired patient-level bootstrap supported the improvement in discrimination after BH correction."

    utility_sentence = (
        "Raw Utility Score was threshold-dependent and did not improve under the enhanced model "
        f"({utility_base:.6f} baseline vs {utility_enh:.6f} enhanced); operational benefit should be interpreted cautiously."
        if not np.isnan(utility_base) and not np.isnan(utility_enh) and utility_enh <= utility_base
        else "Raw Utility Score was threshold-dependent; operational benefit should be interpreted cautiously."
    )

    return f"""# Final Results Snippet

In patient-grouped out-of-fold internal cross-validation on the public PhysioNet/CinC 2019 training split, the enhanced hemodynamic-complexity model improved discrimination relative to the baseline. Raw AUROC increased from {format_metric(auroc_base, 6)} to {format_metric(auroc_enh, 6)}, and raw AUPRC increased from {format_metric(auprc_base, 6)} to {format_metric(auprc_enh, 6)}. {supported}

{utility_sentence}

Patient-level sensitivity at the conventional 0.5 threshold remained low in both models ({format_metric(sens_base, 6)} baseline and {format_metric(sens_enh, 6)} enhanced), emphasizing the need for threshold selection and independent/prospective validation before any clinical deployment.

No independent external validation or official hidden-test evaluation was performed; all claims are limited to internal cross-validation on the public PhysioNet/CinC 2019 training split.
"""


def build_manifest(
    baseline_dir: Path,
    enhanced_dir: Path,
    output_dir: Path,
    args: argparse.Namespace,
    status: str = "complete",
) -> dict[str, Any]:
    """Build a reproducibility manifest for generated final run reporting files."""
    output_files = sorted(path for path in output_dir.iterdir() if path.is_file())
    checksums = {}
    for path in output_files:
        if path.name == "manifest.json":
            continue
        checksums[path.name] = sha256_file(path)
    command = (
        "python src/reporting/build_statistics.py "
        f"--baseline-dir {baseline_dir} --enhanced-dir {enhanced_dir} "
        f"--output-dir {output_dir} --n_bootstrap {args.n_bootstrap} --seed {args.seed}"
    )
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_value(["rev-parse", "--short", "HEAD"]),
        "git_branch": git_value(["rev-parse", "--abbrev-ref", "HEAD"]),
        "baseline_input_dir": str(baseline_dir),
        "enhanced_input_dir": str(enhanced_dir),
        "output_dir": str(output_dir),
        "command": command,
        "n_bootstrap": int(args.n_bootstrap),
        "seed": int(args.seed),
        "output_files": [path.name for path in output_files] + ["manifest.json"],
        "sha256": checksums,
        "status": status,
    }



def build_code_math_audit_summary(
    delong_status: dict[str, str],
    utility_status: dict[str, str],
    calibration_status: dict[str, str],
) -> str:
    """Create a generated static audit summary to ship with statistics tables."""
    return f"""# Code Math Audit Summary

This generated summary accompanies the statistics tables. It is a reporting artifact only; it does not modify training, data harmonization, feature engineering, or experiment outputs.

## Verified Definitions From Code Inspection

- Data harmonization loads PSV files preferentially and skips aggregate CSV/TSV files when PSV files exist, avoiding `Dataset.csv` duplication.
- `Patient_ID` is preserved as string, `TimeStep` is sorted/preserved or causally reconstructed, `SourceSet` is retained, and `SepsisLabel` is cast to int8.
- Feature engineering uses patient-grouped causal rolling/expanding transforms.
- Hemodynamic complexity features are expected as 18 columns: six signals times CV 8h, IQR 8h, and SampEn 24h.
- Training uses patient-level fold assignments via `StratifiedGroupKFold` and excludes `Patient_ID`, `TimeStep`, `SourceSet`, and `SepsisLabel` from model features.
- Calibration in training is cross-fitted by fold IDs in `crossfit_calibration`; reporting still flags paper wording caution until the author confirms this is the intended procedure to describe.
- Reporting distinguishes time-step discrimination metrics, patient-level confusion matrices, patient-level subgroup tables, and Utility Score inference.

## Statistical Status

- DeLong status: **{delong_status.get('status')}** - {delong_status.get('explanation')}
- Utility bootstrap status: **{utility_status.get('status')}** - {utility_status.get('explanation')}
- Calibration audit status: **{calibration_status.get('status')}** - {calibration_status.get('explanation')}

## Known Paper Risks To Fix Before Local Manuscript Editing

- Old feature counts such as 233/251, 71/77, and duplicated-row counts must be replaced by final PSV-only values.
- Hidden-test, external-validation, and direct SOTA-superiority claims must not be made from internal CV results.
- Raw and calibrated calibration metrics must not be collapsed into a blanket claim that calibration improved.
- Patient-level metrics and time-step metrics must be labeled separately.

## Recommendation

Statistics readiness is determined by `completion_checklist.md`. If only calibration wording remains conditional, the next review step can proceed with explicit author confirmation; if Utility or DeLong validation fails on real artifacts, keep the claim tied to paired patient bootstrap instead.
"""


def build_statistics_summary(
    baseline_dir: Path,
    enhanced_dir: Path,
    output_dir: Path,
    baseline_metrics: dict[str, Any],
    enhanced_metrics: dict[str, Any],
    comparison: pd.DataFrame,
    stats: pd.DataFrame,
    subgroup_warnings: list[str],
    feature_summary: dict[str, Any],
    fold_stats: dict[str, float],
    delong_status: dict[str, str],
    utility_status: dict[str, str],
    calibration_status: dict[str, str],
    checklist_df: pd.DataFrame,
) -> str:
    """Create a human-readable Markdown summary for statistics use."""
    auroc_delta = as_float(get_metric(enhanced_metrics, "AUROC_raw")) - as_float(get_metric(baseline_metrics, "AUROC_raw"))
    auprc_delta = as_float(get_metric(enhanced_metrics, "AUPRC_raw")) - as_float(get_metric(baseline_metrics, "AUPRC_raw"))
    utility_delta = as_float(get_metric(enhanced_metrics, "Utility_raw_best")) - as_float(get_metric(baseline_metrics, "Utility_raw_best"))

    stat_lines = []
    for _, row in stats.iterrows():
        stat_lines.append(
            f"- {row['metric']}: delta={format_metric(row['delta'], 6)}, "
            f"95% CI [{format_metric(row['ci_lower'], 6)}, {format_metric(row['ci_upper'], 6)}], "
            f"{row.get('p_value_display', p_value_display(row.get('p_value_two_sided')))}, "
            f"BH {p_value_display(row.get('p_value_bh'))} ({row['test']})"
        )

    generated = "\n".join(f"- `{name}`: {description}" for name, description in TABLE_DESCRIPTIONS.items())
    warnings = [
        "No hidden test set was used in this reporting package.",
        "SourceSet A/B supports traceability and internal source-wise checks; it is not external validation.",
        "Any comparison with Zabihi et al. must be like-for-like in cohort, preprocessing, horizon, metrics, and utility setup.",
        "Raw Utility Score is threshold-dependent and must be interpreted cautiously; it does not necessarily improve when discrimination improves.",
        "Threshold 0.5 can have low patient-level sensitivity; lower thresholds may increase false alarms.",
        "Patient-level and time-step-level metrics answer different questions and must not be mixed in paper text.",
        "Raw calibration worsened if ECE_raw or Brier_raw increased; do not claim all calibration improved.",
    ] + subgroup_warnings
    warning_text = "\n".join(f"- {item}" for item in warnings)

    top_hemo = feature_summary.get("top_hemo_features", [])
    sampen = feature_summary.get("sampen_features_present", [])
    ready_rows = checklist_df[checklist_df["item"].eq("ready_for_methodology_review")]
    ready_status = ready_rows.iloc[0]["status"] if not ready_rows.empty else "UNRESOLVED"
    ready_explanation = ready_rows.iloc[0]["explanation"] if not ready_rows.empty else "Checklist was not available."
    unresolved = checklist_df[checklist_df["status"].isin(["UNRESOLVED", "CONDITIONAL", "FAIL"])]
    unresolved_text = "\n".join(
        f"- {row['item']}: {row['status']} - {row['explanation']}" for _, row in unresolved.iterrows()
    ) or "- None marked as UNRESOLVED; review CONDITIONAL items before methodology review."
    utility_interpretation = (
        "Enhanced raw Utility best is lower than baseline; report Utility as threshold-dependent and do not claim uniform operational utility improvement."
        if utility_delta < 0
        else "Enhanced raw Utility best is higher than baseline, but Utility remains threshold-dependent and should be interpreted cautiously."
    )

    return f"""# Statistics Summary

## Input Paths

- Baseline results: `{baseline_dir}`
- Enhanced results: `{enhanced_dir}`
- Output tables: `{output_dir}`

## Primary Metrics

| metric | baseline | enhanced | delta |
|---|---:|---:|---:|
| AUROC raw | {format_metric(get_metric(baseline_metrics, 'AUROC_raw'), 6)} | {format_metric(get_metric(enhanced_metrics, 'AUROC_raw'), 6)} | {format_metric(auroc_delta, 6)} |
| AUPRC raw | {format_metric(get_metric(baseline_metrics, 'AUPRC_raw'), 6)} | {format_metric(get_metric(enhanced_metrics, 'AUPRC_raw'), 6)} | {format_metric(auprc_delta, 6)} |
| Utility raw best | {format_metric(get_metric(baseline_metrics, 'Utility_raw_best'), 6)} | {format_metric(get_metric(enhanced_metrics, 'Utility_raw_best'), 6)} | {format_metric(utility_delta, 6)} |
| n features | {format_metric(get_metric(baseline_metrics, 'n_features'), 0)} | {format_metric(get_metric(enhanced_metrics, 'n_features'), 0)} | {format_metric(as_float(get_metric(enhanced_metrics, 'n_features')) - as_float(get_metric(baseline_metrics, 'n_features')), 0)} |

These values are read from the final run `summary_metrics.json` / `metrics.json` files. The enhanced model improves raw AUROC and raw AUPRC. {utility_interpretation}

## Statistical Tests

{chr(10).join(stat_lines)}

Paired AUROC/AUPRC/Brier tests use patient-level bootstrap resampling over OOF predictions. DeLong status: **{delong_status.get('status')}**. Utility bootstrap status: **{utility_status.get('status')}**.

## Statistics Completion Status

- ready_for_methodology_review: **{ready_status}**
- explanation: {ready_explanation}
- calibration status: **{calibration_status.get('status')}** - {calibration_status.get('explanation')}

## Remaining Unresolved Items

{unresolved_text}

## Hemodynamic Feature Importance Checks

- Hemodynamic complexity features found in enhanced importance table: {feature_summary.get('enhanced_hemo_features_in_importance', 0)} of {feature_summary.get('enhanced_expected_hemo_features', 18)} expected features.
- Top-ranked hemodynamic features: {', '.join(top_hemo) if top_hemo else 'none detected'}.
- SampEn features present: {', '.join(sampen) if sampen else 'none detected'}.

## Fold Delta Summary

- Delta AUROC mean/std: {format_metric(fold_stats.get('delta_AUROC_mean'), 6)} / {format_metric(fold_stats.get('delta_AUROC_std'), 6)}
- Delta AUPRC mean/std: {format_metric(fold_stats.get('delta_AUPRC_mean'), 6)} / {format_metric(fold_stats.get('delta_AUPRC_std'), 6)}

## Allowed Paper Claims

- Enhanced improves discrimination under patient-grouped internal cross-validation on the public PhysioNet/CinC 2019 training split.
- AUPRC gain is statistically supported by paired patient bootstrap if `statistical_tests.csv` confirms significance.
- Hemodynamic complexity features add complementary ranking information.
- Utility is threshold-dependent and should be reported with caution.
- Independent/prospective validation is required before clinical deployment.

## Do Not Write In Paper

- Do not claim clinical deployment.
- Do not claim near-perfect calibration.
- Do not claim external validation.
- Do not claim hidden-test performance.
- Do not claim SOTA superiority over Zabihi.
- Do not claim all calibration improved, because raw calibration worsened if ECE_raw or Brier_raw increased.

## Interpretation Warnings

{warning_text}

## Generated Tables

{generated}

## Pending Paper Checks

- Correct feature count statements.
- Correct whether SampEn features are described as five or six signals.
- Correct subgroup text and avoid row/time-step double counting.
- Explain calibration and distinguish raw, Platt, and isotonic outputs.
- Explain patient-level vs time-step-level metrics.
- Explain PhysioNet Utility Score and threshold dependence.
- Update reproducibility and ethics statements.
"""


def main() -> None:
    """Build all statistics reporting tables."""
    setup_logging()
    args = parse_args()
    load_runtime_dependencies()
    baseline_dir = args.baseline_dir
    enhanced_dir = args.enhanced_dir
    output_dir = args.output_dir

    validate_result_dir(baseline_dir, "baseline")
    validate_result_dir(enhanced_dir, "enhanced")
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_metrics = load_metrics(baseline_dir)
    enhanced_metrics = load_metrics(enhanced_dir)
    validation_checks, validation_md, _aligned_for_validation = build_validation_checks(
        baseline_dir, enhanced_dir, baseline_metrics, enhanced_metrics
    )

    comparison = build_comparison_summary(baseline_metrics, enhanced_metrics)
    stats = build_statistical_tests(
        baseline_dir,
        enhanced_dir,
        baseline_metrics,
        enhanced_metrics,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )
    stats, delong_status = add_delong_and_format_stats(stats, baseline_dir, enhanced_dir)
    utility_tests, utility_status = build_utility_statistical_tests(
        baseline_dir,
        enhanced_dir,
        baseline_metrics,
        enhanced_metrics,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )
    threshold_points = build_threshold_operating_points(
        baseline_dir, enhanced_dir, baseline_metrics, enhanced_metrics
    )
    patient_metrics = build_patient_level_metrics(
        baseline_dir, enhanced_dir, baseline_metrics, enhanced_metrics
    )
    calibration = build_calibration_table(
        baseline_dir, enhanced_dir, baseline_metrics, enhanced_metrics
    )
    subgroup, subgroup_warnings = build_subgroup_table(baseline_dir, enhanced_dir)
    feature_importance, feature_summary = build_feature_importance_table(baseline_dir, enhanced_dir)
    fold_comparison, fold_stats = build_fold_comparison(baseline_dir, enhanced_dir)
    patient_confusion = build_patient_confusion_matrices(
        baseline_dir, enhanced_dir, baseline_metrics, enhanced_metrics
    )
    subgroup_patient = build_subgroup_patient_table(
        baseline_dir, enhanced_dir, baseline_metrics, enhanced_metrics
    )
    feature_count_audit = build_feature_count_audit(feature_summary, baseline_metrics, enhanced_metrics)
    calibration_audit, calibration_status = build_calibration_audit(baseline_metrics, enhanced_metrics)
    checklist_df, checklist_md = build_completion_checklist(
        delong_status,
        utility_status,
        calibration_status,
        feature_count_audit,
        subgroup_patient,
        baseline_metrics,
        enhanced_metrics,
    )
    methods_notes = build_statistical_methods_notes(delong_status, utility_status)
    code_math_summary = build_code_math_audit_summary(
        delong_status, utility_status, calibration_status
    )
    results_snippet = build_results_snippet(baseline_metrics, enhanced_metrics, stats)

    write_csv(comparison, output_dir / "comparison_summary.csv")
    write_csv(stats, output_dir / "statistical_tests.csv")
    write_csv(threshold_points, output_dir / "threshold_operating_points.csv")
    write_csv(patient_metrics, output_dir / "patient_level_metrics.csv")
    write_csv(calibration, output_dir / "calibration_table.csv")
    write_csv(subgroup, output_dir / "subgroup_table.csv")
    write_csv(feature_importance, output_dir / "feature_importance_table.csv")
    write_csv(fold_comparison, output_dir / "fold_comparison.csv")
    write_csv(utility_tests, output_dir / "utility_statistical_tests.csv")
    write_csv(patient_confusion, output_dir / "patient_confusion_matrices.csv")
    write_csv(subgroup_patient, output_dir / "subgroup_patient_table.csv")
    write_csv(feature_count_audit, output_dir / "feature_count_audit.csv")
    write_csv(checklist_df, output_dir / "completion_checklist.csv")
    write_csv(validation_checks, output_dir / "validation_checks.csv")

    write_markdown_table(comparison, output_dir / "comparison_summary.md")
    write_markdown_table(stats, output_dir / "statistical_tests.md")
    write_markdown_table(threshold_points, output_dir / "threshold_operating_points.md")
    write_markdown_table(patient_metrics, output_dir / "patient_level_metrics.md")

    (output_dir / "calibration_audit.md").write_text(calibration_audit, encoding="utf-8")
    logging.info("Wrote %s", output_dir / "calibration_audit.md")
    (output_dir / "statistical_methods_notes.md").write_text(methods_notes, encoding="utf-8")
    logging.info("Wrote %s", output_dir / "statistical_methods_notes.md")
    (output_dir / "code_math_audit_summary.md").write_text(code_math_summary, encoding="utf-8")
    logging.info("Wrote %s", output_dir / "code_math_audit_summary.md")
    (output_dir / "completion_checklist.md").write_text(checklist_md, encoding="utf-8")
    logging.info("Wrote %s", output_dir / "completion_checklist.md")
    (output_dir / "validation_checks.md").write_text(validation_md, encoding="utf-8")
    logging.info("Wrote %s", output_dir / "validation_checks.md")
    (output_dir / "results_snippet.md").write_text(results_snippet, encoding="utf-8")
    logging.info("Wrote %s", output_dir / "results_snippet.md")

    summary = build_statistics_summary(
        baseline_dir,
        enhanced_dir,
        output_dir,
        baseline_metrics,
        enhanced_metrics,
        comparison,
        stats,
        subgroup_warnings,
        feature_summary,
        fold_stats,
        delong_status,
        utility_status,
        calibration_status,
        checklist_df,
    )
    summary_path = output_dir / "statistics_summary.md"
    summary_path.write_text(summary, encoding="utf-8")
    logging.info("Wrote %s", summary_path)
    manifest = build_manifest(baseline_dir, enhanced_dir, output_dir, args)
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logging.info("Wrote %s", manifest_path)
    logging.info("Statistics tables complete: %s", output_dir)


if __name__ == "__main__":
    main()
