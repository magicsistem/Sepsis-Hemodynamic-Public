#!/usr/bin/env python
"""Build internal robustness outputs from existing final artifacts.

This script reads out-of-fold predictions and statistics reporting artifacts only.
It does not train models, download data, or read independent cohorts.
"""

import argparse
import csv
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

EXPECTED_FINAL_SUMMARY = {
    "baseline": {
        "n_features": 287,
        "AUROC_raw": 0.9314974500214285,
        "AUPRC_raw": 0.3060453644628811,
        "Utility_raw_at_0.5": 0.5014922711148125,
        "Utility_raw_best": 0.5056648262039968,
        "Utility_raw_best_threshold": 0.44999998807907104,
    },
    "enhanced": {
        "n_features": 305,
        "AUROC_raw": 0.934294714682833,
        "AUPRC_raw": 0.3264125134862838,
        "Utility_raw_at_0.5": 0.49546665779939514,
        "Utility_raw_best": 0.5010110201040191,
        "Utility_raw_best_threshold": 0.4099999964237213,
    },
}

EXPECTED_FINAL_PATIENT = {
    "baseline": {
        "Patient_sensitivity_best_method_at_0.5": 0.041950886766712145,
        "Patient_specificity_best_method_at_0.5": 0.993476633515132,
        "Lead_time_median_best_method_at_0.5": 24.0,
    },
    "enhanced": {
        "Patient_sensitivity_best_method_at_0.5": 0.05184174624829468,
        "Patient_specificity_best_method_at_0.5": 0.9912308843973906,
        "Lead_time_median_best_method_at_0.5": 23.5,
    },
}

PROBABILITY_COLUMNS = {
    "raw": "prob_raw",
    "platt": "prob_platt",
    "isotonic": "prob_isotonic",
}


def import_analysis_stack():
    import numpy as np
    import pandas as pd
    from sklearn.metrics import average_precision_score, roc_auc_score

    return np, pd, average_precision_score, roc_auc_score


def load_utility_helpers():
    """Load final Utility helpers with their numpy/pandas globals initialized.

    The statistics reporting module can be imported in contexts where optional stack
    globals are not initialized. Rebinding them here prevents Utility subgroup
    reports from silently degrading to NOT_AVAILABLE.
    """
    try:
        np, pd, _, _ = import_analysis_stack()
        import src.reporting.build_statistics as statistics_tables

        statistics_tables.np = np
        statistics_tables.pd = pd

        return (
            statistics_tables.utility_patient_components,
            statistics_tables.normalized_utility_from_components,
            None,
        )
    except Exception as exc:  # pragma: no cover - depends on execution environment
        return None, None, f"utility function not importable: {repr(exc)}"


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def write_markdown_table(path, title, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(f"# {title}\n\n")
        f.write("| " + " | ".join(fieldnames) + " |\n")
        f.write("| " + " | ".join(["---"] * len(fieldnames)) + " |\n")
        for row in rows:
            f.write("| " + " | ".join(str(row.get(name, "")) for name in fieldnames) + " |\n")


def safe_float(value):
    if value is None or value == "":
        return math.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def metric_from_dict(metrics, key):
    if key in metrics:
        return metrics[key]
    aliases = {
        "n_features": ["features", "n_final_features", "final_numeric_features"],
        "Patient_sensitivity_best_method_at_0.5": [
            "patient_sensitivity_best_method_at_0.5",
        ],
        "Patient_specificity_best_method_at_0.5": [
            "patient_specificity_best_method_at_0.5",
        ],
        "Lead_time_median_best_method_at_0.5": [
            "lead_time_median_best_method_at_0.5",
        ],
    }
    for alias in aliases.get(key, []):
        if alias in metrics:
            return metrics[alias]
    return None


def column_by_candidates(df, candidates):
    columns = {str(col).lower(): col for col in df.columns}
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
        found = columns.get(candidate.lower())
        if found is not None:
            return found
    return None


def find_statistics_patient_metrics(label, statistics_dir, result_dir):
    _, pd, _, _ = import_analysis_stack()
    attempts = []

    for path in [result_dir / "summary_metrics.json", result_dir / "metrics.json"]:
        metrics = load_json(path)
        if metrics is None:
            attempts.append(f"{path}: missing")
            continue
        observed = {key: safe_float(metric_from_dict(metrics, key)) for key in EXPECTED_FINAL_PATIENT[label]}
        if all(not math.isnan(value) for value in observed.values()):
            columns = {key: key for key in EXPECTED_FINAL_PATIENT[label]}
            return observed, path, columns, attempts
        missing = [key for key, value in observed.items() if math.isnan(value)]
        attempts.append(f"{path}: missing {missing}")

    comparison_path = statistics_dir / "comparison_summary.csv"
    if not comparison_path.exists():
        attempts.append(f"{comparison_path}: missing")
        return None, None, {}, attempts

    table = pd.read_csv(comparison_path)
    metric_col = column_by_candidates(table, ["metric", "Metric", "name"])
    value_col = column_by_candidates(table, [label, label.capitalize()])
    if metric_col is None or value_col is None:
        attempts.append(f"{comparison_path}: missing metric or {label} value column")
        return None, None, {}, attempts

    observed = {}
    for key in EXPECTED_FINAL_PATIENT[label]:
        rows = table[table[metric_col].astype(str) == key]
        if rows.empty:
            observed[key] = math.nan
        else:
            observed[key] = safe_float(rows.iloc[0][value_col])
    if all(not math.isnan(value) for value in observed.values()):
        columns = {key: f"{metric_col}/{value_col}" for key in EXPECTED_FINAL_PATIENT[label]}
        return observed, comparison_path, columns, attempts

    missing = [key for key, value in observed.items() if math.isnan(value)]
    attempts.append(f"{comparison_path}: missing {missing}")
    return None, None, {}, attempts


def load_json(path):
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def status_check(rows, check_id, status, details, evidence=""):
    rows.append(
        {
            "check_id": check_id,
            "status": status,
            "details": details,
            "evidence": evidence,
        }
    )


def resolve_oof(result_dir, pd):
    candidates = [
        result_dir / "oof_predictions.csv",
        result_dir / "oof.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return pd.read_csv(candidate), candidate
    fold_files = sorted(result_dir.glob("fold*_oof.csv"))
    if fold_files:
        frames = [pd.read_csv(path) for path in fold_files]
        return pd.concat(frames, ignore_index=True), fold_files[0].parent / "fold*_oof.csv"
    raise FileNotFoundError(f"No OOF prediction file found in {result_dir}")


def normalize_oof_columns(df):
    rename = {}
    for col in df.columns:
        lower = col.lower()
        if lower == "fold":
            rename[col] = "fold"
        elif lower == "patient_id":
            rename[col] = "Patient_ID"
        elif lower == "timestep":
            rename[col] = "TimeStep"
        elif lower == "sepsislabel":
            rename[col] = "SepsisLabel"
    df = df.rename(columns=rename).copy()
    if "Patient_ID" in df.columns:
        df["Patient_ID"] = df["Patient_ID"].astype(str)
    return df


def probability_range(df, prob_col):
    vals = df[prob_col].dropna()
    if vals.empty:
        return math.nan, math.nan, False
    min_v = float(vals.min())
    max_v = float(vals.max())
    return min_v, max_v, min_v >= 0.0 and max_v <= 1.0


def label_summary(df):
    y = df["SepsisLabel"].astype(int)
    patient = df.groupby("Patient_ID")["SepsisLabel"].max().astype(int)
    return {
        "row_positives": int(y.sum()),
        "row_negatives": int((y == 0).sum()),
        "patient_positives": int(patient.sum()),
        "patient_negatives": int((patient == 0).sum()),
    }


def has_two_classes(y):
    vals = set(int(v) for v in y.dropna().unique())
    return vals == {0, 1}


def ece_score(np, y_true, prob, n_bins=10):
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(prob, dtype=float)
    mask = np.isfinite(y) & np.isfinite(p)
    y = y[mask]
    p = p[mask]
    if len(y) == 0:
        return math.nan
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    total = len(y)
    ece = 0.0
    for i in range(n_bins):
        if i == n_bins - 1:
            idx = (p >= bins[i]) & (p <= bins[i + 1])
        else:
            idx = (p >= bins[i]) & (p < bins[i + 1])
        if idx.any():
            ece += float(idx.sum()) / total * abs(float(y[idx].mean()) - float(p[idx].mean()))
    return ece


def binary_metrics(np, y_true, prob, threshold):
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(prob, dtype=float)
    pred = (p >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    sensitivity = tp / (tp + fn) if tp + fn else math.nan
    specificity = tn / (tn + fp) if tn + fp else math.nan
    precision = tp / (tp + fp) if tp + fp else math.nan
    recall = sensitivity
    f1 = 2 * precision * recall / (precision + recall) if precision + recall and not math.isnan(precision) and not math.isnan(recall) else math.nan
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def patient_level_metrics(df, prob_col, threshold):
    np, _, _, _ = import_analysis_stack()
    patient = (
        df.groupby("Patient_ID")
        .agg(SepsisLabel=("SepsisLabel", "max"), probability=(prob_col, "max"))
        .reset_index()
    )
    metrics = binary_metrics(np, patient["SepsisLabel"], patient["probability"], threshold)
    metrics["n_alerted_patients"] = int((patient["probability"] >= threshold).sum())
    return metrics


def utility_at_threshold(df, prob_col, threshold, utility_helpers):
    utility_patient_components, normalized_utility_from_components, error = utility_helpers
    if utility_patient_components is None:
        return math.nan, f"NOT_AVAILABLE: {error}"
    needed = ["Patient_ID", "TimeStep", "SepsisLabel", prob_col]
    missing = [col for col in needed if col not in df.columns]
    if missing:
        return math.nan, f"NOT_AVAILABLE: missing {missing}"
    tmp = df[needed].copy()
    try:
        components = utility_patient_components(tmp, prob_col, threshold)
        return float(normalized_utility_from_components(components)), "computed_from_repo_reporting_utility"
    except Exception as exc:
        return math.nan, f"NOT_AVAILABLE: {repr(exc)}"


def model_metrics(df, prob_col, utility_helpers, threshold=0.5):
    np, _, average_precision_score, roc_auc_score = import_analysis_stack()
    y = df["SepsisLabel"].astype(int)
    p = df[prob_col].astype(float)
    out = {
        "brier": float(np.mean((p - y) ** 2)),
        "ece": ece_score(np, y, p),
    }
    if has_two_classes(y):
        out["auroc"] = float(roc_auc_score(y, p))
        out["auprc"] = float(average_precision_score(y, p))
        out["status"] = "OK"
        out["reason"] = ""
    else:
        out["auroc"] = math.nan
        out["auprc"] = math.nan
        out["status"] = "NA"
        out["reason"] = "insufficient class diversity"
    out.update(binary_metrics(np, y, p, threshold))
    patient = patient_level_metrics(df, prob_col, threshold)
    out["patient_sensitivity"] = patient["sensitivity"]
    out["patient_specificity"] = patient["specificity"]
    out["n_alerted_patients"] = patient["n_alerted_patients"]
    utility, utility_source = utility_at_threshold(df, prob_col, threshold, utility_helpers)
    out["utility"] = utility
    out["utility_source"] = utility_source
    return out


def format_float(value):
    if value is None:
        return ""
    try:
        val = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(val):
        return "NA"
    return f"{val:.9g}"


def add_sources_from_harmonized(base, enhanced, harmonized, checks):
    if "SourceSet" in base.columns and "SourceSet" in enhanced.columns:
        status_check(checks, "sourceset_available", "PASS", "SourceSet present in OOF predictions.", "OOF")
        return base, enhanced
    if not harmonized.exists():
        status_check(checks, "sourceset_available", "NOT_AVAILABLE", "SourceSet unavailable because harmonized CSV was not found.", str(harmonized))
        return base, enhanced
    np, pd, _, _ = import_analysis_stack()
    try:
        source = pd.read_csv(harmonized, usecols=["Patient_ID", "TimeStep", "SourceSet"])
    except Exception as exc:
        status_check(checks, "sourceset_available", "NOT_AVAILABLE", f"Could not read SourceSet from harmonized CSV: {exc}", str(harmonized))
        return base, enhanced
    source["Patient_ID"] = source["Patient_ID"].astype(str)
    before_base = len(base)
    before_enh = len(enhanced)
    base = base.merge(source, on=["Patient_ID", "TimeStep"], how="left")
    enhanced = enhanced.merge(source, on=["Patient_ID", "TimeStep"], how="left")
    if len(base) != before_base or len(enhanced) != before_enh:
        status_check(checks, "sourceset_merge", "FAIL", "SourceSet merge changed OOF row counts.", str(harmonized))
    elif base["SourceSet"].notna().any() and enhanced["SourceSet"].notna().any():
        status_check(checks, "sourceset_available", "PASS", "SourceSet merged from harmonized PSV-only artifact.", str(harmonized))
    else:
        status_check(checks, "sourceset_available", "NOT_AVAILABLE", "SourceSet column was read but did not match OOF rows.", str(harmonized))
    return base, enhanced


def validate_inputs(args, checks, allow_missing):
    for label, path in [
        ("baseline_dir", args.baseline_dir),
        ("enhanced_dir", args.enhanced_dir),
        ("statistics_dir", args.statistics_dir),
    ]:
        if path.exists():
            status_check(checks, f"{label}_exists", "PASS", f"{label} exists.", str(path))
        else:
            status = "NOT_AVAILABLE" if allow_missing else "FAIL"
            status_check(checks, f"{label}_exists", status, f"{label} is missing.", str(path))
            if not allow_missing:
                raise FileNotFoundError(path)
    status_check(checks, "no_independent_cohort_read", "PASS", "Script uses final OOF predictions and harmonized traceability only.", "CLI inputs")


def validate_methodology_audit(args, checks):
    audit_csv = args.inventory_dir / "no_leakage_audit.csv"
    if not audit_csv.exists():
        status_check(checks, "methodology_audit_pass_count", "NOT_AVAILABLE", "methodology audit CSV was not found.", str(audit_csv))
        return
    _, pd, _, _ = import_analysis_stack()
    audit = pd.read_csv(audit_csv)
    status_col = "status" if "status" in audit.columns else None
    if status_col is None:
        status_check(checks, "methodology_audit_pass_count", "CONDITIONAL", "methodology audit CSV has no status column.", str(audit_csv))
        return
    pass_count = int((audit[status_col] == "PASS").sum())
    status = "PASS" if pass_count == 31 else "CONDITIONAL"
    status_check(checks, "methodology_audit_pass_count", status, f"methodology audit PASS count = {pass_count}; expected 31.", str(audit_csv))


def validate_oof(label, df, source_path, checks):
    required = ["Patient_ID", "TimeStep", "SepsisLabel", "fold", "prob_raw"]
    missing = [col for col in required if col not in df.columns]
    status_check(
        checks,
        f"{label}_oof_columns",
        "PASS" if not missing else "FAIL",
        f"{label} OOF required columns missing: {missing}" if missing else f"{label} OOF required columns present.",
        str(source_path),
    )
    if {"Patient_ID", "TimeStep"}.issubset(df.columns):
        duplicates = int(df.duplicated(["Patient_ID", "TimeStep"]).sum())
        status_check(
            checks,
            f"{label}_no_duplicate_patient_timestep",
            "PASS" if duplicates == 0 else "FAIL",
            f"Duplicate Patient_ID+TimeStep rows: {duplicates}.",
            str(source_path),
        )
    if "SepsisLabel" in df.columns:
        labels = sorted(str(v) for v in df["SepsisLabel"].dropna().unique())
        ok = set(labels).issubset({"0", "1", "0.0", "1.0"})
        status_check(
            checks,
            f"{label}_binary_labels",
            "PASS" if ok else "FAIL",
            f"Observed labels: {labels}.",
            str(source_path),
        )
    for method, prob_col in PROBABILITY_COLUMNS.items():
        if prob_col in df.columns:
            lo, hi, ok = probability_range(df, prob_col)
            status_check(
                checks,
                f"{label}_{method}_probability_range",
                "PASS" if ok else "FAIL",
                f"{prob_col} range: [{lo}, {hi}].",
                str(source_path),
            )


def validate_alignment(base, enhanced, checks):
    key = ["Patient_ID", "TimeStep", "SepsisLabel", "fold"]
    missing = [col for col in key if col not in base.columns or col not in enhanced.columns]
    if missing:
        status_check(checks, "baseline_enhanced_oof_alignment", "FAIL", f"Cannot align OOF; missing key columns: {missing}.", "")
        return None
    base_key = base[key].copy()
    enhanced_key = enhanced[key].copy()
    base_key["Patient_ID"] = base_key["Patient_ID"].astype(str)
    enhanced_key["Patient_ID"] = enhanced_key["Patient_ID"].astype(str)
    same = len(base_key) == len(enhanced_key) and base_key.equals(enhanced_key)
    if same:
        status_check(checks, "baseline_enhanced_oof_alignment", "PASS", "OOF rows are aligned by Patient_ID, TimeStep, SepsisLabel, and fold.", "")
    else:
        merged = base_key.merge(enhanced_key, on=key, how="inner")
        status_check(
            checks,
            "baseline_enhanced_oof_alignment",
            "FAIL",
            f"OOF alignment mismatch; baseline rows={len(base_key)}, enhanced rows={len(enhanced_key)}, inner rows={len(merged)}.",
            "",
        )
    return same


def validate_expected_metrics(label, result_dir, checks):
    summary = load_json(result_dir / "summary_metrics.json") or load_json(result_dir / "metrics.json")
    if summary is None:
        status_check(checks, f"{label}_final_metric_match", "NOT_AVAILABLE", "No summary_metrics.json or metrics.json found.", str(result_dir))
        return
    tolerance = 5e-4
    all_ok = True
    messages = []
    for key, expected in EXPECTED_FINAL_SUMMARY[label].items():
        observed = safe_float(metric_from_dict(summary, key))
        if math.isnan(observed):
            all_ok = False
            messages.append(f"{key}: missing")
            continue
        diff = abs(observed - expected)
        if diff > tolerance:
            all_ok = False
        messages.append(f"{key}: observed={observed:.9g}, expected={expected:.9g}, diff={diff:.3g}")
    status_check(
        checks,
        f"{label}_final_metric_match",
        "PASS" if all_ok else "FAIL",
        "; ".join(messages),
        str(result_dir / "summary_metrics.json"),
    )


def validate_statistics_patient_metrics(label, statistics_dir, result_dir, checks):
    observed, source_path, columns, attempts = find_statistics_patient_metrics(label, statistics_dir, result_dir)
    if observed is None:
        status_check(
            checks,
            f"{label}_patient_level_and_lead_time_metrics_match",
            "NOT_AVAILABLE",
            "Could not locate patient-level/lead-time metrics in statistics tables. Attempts: " + "; ".join(attempts),
            str(statistics_dir),
        )
        return
    tolerance = 5e-4
    all_found = True
    all_match = True
    messages = []
    for key, expected in EXPECTED_FINAL_PATIENT[label].items():
        value = observed.get(key, math.nan)
        col = columns.get(key)
        if math.isnan(value):
            all_found = False
            messages.append(f"{key}: missing in column {col}")
            continue
        diff = abs(value - expected)
        if diff > tolerance:
            all_match = False
        messages.append(f"{key}: observed={value:.9g}, expected={expected:.9g}, diff={diff:.3g}, column={col}")
    if all_found and all_match:
        status = "PASS"
    elif all_found:
        status = "FAIL"
    else:
        status = "CONDITIONAL"
    status_check(
        checks,
        f"{label}_patient_level_and_lead_time_metrics_match",
        status,
        "; ".join(messages),
        str(source_path),
    )


def build_sourcewise_metrics(base, enhanced, utility_helpers):
    rows = []
    if "SourceSet" not in base.columns or "SourceSet" not in enhanced.columns:
        return [
            {
                "group": "SourceSet",
                "level": "NOT_AVAILABLE",
                "status": "NOT_AVAILABLE",
                "reason": "SourceSet unavailable in harmonized artifact or OOF predictions.",
            }
        ]
    for source in sorted(set(base["SourceSet"].dropna().astype(str)) | set(enhanced["SourceSet"].dropna().astype(str))):
        base_sub = base[base["SourceSet"].astype(str) == source]
        enh_sub = enhanced[enhanced["SourceSet"].astype(str) == source]
        rows.extend(compare_group_metrics("SourceSet", source, base_sub, enh_sub, utility_helpers))
    return rows


def compare_group_metrics(group, level, base_sub, enh_sub, utility_helpers):
    rows = []
    summary = label_summary(base_sub) if len(base_sub) else {}
    for model, sub in [("baseline", base_sub), ("enhanced", enh_sub)]:
        if len(sub) == 0 or "prob_raw" not in sub.columns:
            rows.append({"group": group, "level": level, "model": model, "status": "NOT_AVAILABLE", "reason": "No rows or missing prob_raw."})
            continue
        metrics = model_metrics(sub, "prob_raw", utility_helpers, threshold=0.5)
        rows.append(
            {
                "group": group,
                "level": level,
                "model": model,
                "status": metrics["status"],
                "reason": metrics["reason"],
                "n_time_steps": len(sub),
                "n_patients": sub["Patient_ID"].nunique(),
                "row_positives": summary.get("row_positives", ""),
                "row_negatives": summary.get("row_negatives", ""),
                "patient_positives": summary.get("patient_positives", ""),
                "patient_negatives": summary.get("patient_negatives", ""),
                "time_step_prevalence": format_float(summary.get("row_positives", 0) / len(base_sub) if len(base_sub) else math.nan),
                "patient_prevalence": format_float(summary.get("patient_positives", 0) / base_sub["Patient_ID"].nunique() if len(base_sub) else math.nan),
                "auroc_raw": format_float(metrics["auroc"]),
                "auprc_raw": format_float(metrics["auprc"]),
                "brier_raw": format_float(metrics["brier"]),
                "ece_raw": format_float(metrics["ece"]),
                "utility_raw_at_0.5": format_float(metrics["utility"]),
                "utility_source": metrics["utility_source"],
                "sensitivity_at_0.5": format_float(metrics["sensitivity"]),
                "specificity_at_0.5": format_float(metrics["specificity"]),
                "patient_sensitivity_at_0.5": format_float(metrics["patient_sensitivity"]),
                "patient_specificity_at_0.5": format_float(metrics["patient_specificity"]),
            }
        )
    if len(base_sub) and len(enh_sub) and has_two_classes(base_sub["SepsisLabel"]):
        b = model_metrics(base_sub, "prob_raw", utility_helpers, threshold=0.5)
        e = model_metrics(enh_sub, "prob_raw", utility_helpers, threshold=0.5)
        rows.append(
            {
                "group": group,
                "level": level,
                "model": "enhanced_minus_baseline",
                "status": "OK",
                "delta_auroc_raw": format_float(e["auroc"] - b["auroc"]),
                "delta_auprc_raw": format_float(e["auprc"] - b["auprc"]),
            }
        )
    return rows


def build_subgroup_metrics(base, enhanced, utility_helpers):
    rows = []
    if "Age" in base.columns and "Age" in enhanced.columns:
        for df in (base, enhanced):
            df["AgeBin"] = df["Age"].apply(age_bin)
        for level in ["<40", "40-59", "60-79", ">=80", "unknown"]:
            rows.extend(compare_group_metrics("AgeBin", level, base[base["AgeBin"] == level], enhanced[enhanced["AgeBin"] == level], utility_helpers))
    else:
        rows.append({"group": "AgeBin", "level": "NOT_AVAILABLE", "status": "NOT_AVAILABLE", "reason": "Age column unavailable."})

    gender_col = None
    for candidate in ["Gender", "Sex", "sex", "gender"]:
        if candidate in base.columns and candidate in enhanced.columns:
            gender_col = candidate
            break
    if gender_col:
        levels = sorted(set(base[gender_col].dropna().astype(str)) | set(enhanced[gender_col].dropna().astype(str)))
        for level in levels:
            rows.extend(compare_group_metrics(gender_col, level, base[base[gender_col].astype(str) == level], enhanced[enhanced[gender_col].astype(str) == level], utility_helpers))
    else:
        rows.append({"group": "Gender", "level": "NOT_AVAILABLE", "status": "NOT_AVAILABLE", "reason": "Gender/Sex column unavailable."})

    if "SourceSet" in base.columns and "SourceSet" in enhanced.columns:
        for source in sorted(set(base["SourceSet"].dropna().astype(str)) | set(enhanced["SourceSet"].dropna().astype(str))):
            rows.extend(compare_group_metrics("SourceSet", source, base[base["SourceSet"].astype(str) == source], enhanced[enhanced["SourceSet"].astype(str) == source], utility_helpers))
    else:
        rows.append({"group": "SourceSet", "level": "NOT_AVAILABLE", "status": "NOT_AVAILABLE", "reason": "SourceSet unavailable."})
    return rows


def age_bin(value):
    try:
        age = float(value)
    except (TypeError, ValueError):
        return "unknown"
    if math.isnan(age):
        return "unknown"
    if age < 40:
        return "<40"
    if age < 60:
        return "40-59"
    if age < 80:
        return "60-79"
    return ">=80"


def build_threshold_robustness(base, enhanced, utility_helpers):
    np, _, _, _ = import_analysis_stack()
    rows = []
    thresholds = [round(v, 2) for v in np.arange(0.05, 1.0, 0.05)]
    for model, df in [("baseline", base), ("enhanced", enhanced)]:
        for method, prob_col in PROBABILITY_COLUMNS.items():
            if prob_col not in df.columns:
                continue
            for threshold in thresholds:
                metrics = binary_metrics(np, df["SepsisLabel"], df[prob_col], threshold)
                patient = patient_level_metrics(df, prob_col, threshold)
                utility, utility_source = utility_at_threshold(df, prob_col, threshold, utility_helpers)
                rows.append(
                    {
                        "model": model,
                        "method": method,
                        "probability_column": prob_col,
                        "threshold": threshold,
                        "sensitivity": format_float(metrics["sensitivity"]),
                        "specificity": format_float(metrics["specificity"]),
                        "precision": format_float(metrics["precision"]),
                        "recall": format_float(metrics["recall"]),
                        "f1": format_float(metrics["f1"]),
                        "utility": format_float(utility),
                        "utility_source": utility_source,
                        "patient_sensitivity": format_float(patient["sensitivity"]),
                        "patient_specificity": format_float(patient["specificity"]),
                        "n_alerted_patients": patient["n_alerted_patients"],
                    }
                )
    return rows


def build_calibration_robustness(base, enhanced):
    rows = []
    np, _, _, _ = import_analysis_stack()
    for model, df in [("baseline", base), ("enhanced", enhanced)]:
        raw_brier = None
        raw_ece = None
        if "prob_raw" in df.columns:
            raw_brier = float(np.mean((df["prob_raw"] - df["SepsisLabel"]) ** 2))
            raw_ece = ece_score(np, df["SepsisLabel"], df["prob_raw"])
        for method, prob_col in PROBABILITY_COLUMNS.items():
            if prob_col not in df.columns:
                continue
            brier = float(np.mean((df[prob_col] - df["SepsisLabel"]) ** 2))
            ece = ece_score(np, df["SepsisLabel"], df[prob_col])
            rows.append(
                {
                    "model": model,
                    "method": method,
                    "probability_column": prob_col,
                    "brier": format_float(brier),
                    "ece": format_float(ece),
                    "brier_change_vs_raw": format_float(brier - raw_brier) if raw_brier is not None else "NA",
                    "ece_change_vs_raw": format_float(ece - raw_ece) if raw_ece is not None else "NA",
                    "calibration_scope": "internal_oof_only",
                }
            )
    return rows


def build_ablation_inventory(args):
    rows = []
    search_paths = [
        args.project_root / "configs" / "ablations",
        args.project_root / "configs" / "robustness",
    ]
    for directory in search_paths:
        if directory.exists():
            for path in sorted(directory.rglob("*")):
                if path.is_file():
                    rows.append({"path": str(path), "kind": "config", "status": "CONFIG_ONLY", "notes": "Configuration found; no new internal robustness training was run."})
    for pattern in ["ablation_*.yaml", "robustness_*.yaml"]:
        for path in sorted((args.project_root / "configs").glob(pattern)):
            rows.append({"path": str(path), "kind": "config", "status": "CONFIG_ONLY", "notes": "Configuration found; no new internal robustness training was run."})
    results_dir = args.project_root / "results"
    if results_dir.exists():
        for pattern in ["*ablation*", "*robustness*"]:
            for path in sorted(results_dir.glob(pattern)):
                if path.resolve() == args.output_dir.resolve():
                    continue
                rows.append({"path": str(path), "kind": "result_or_directory", "status": "FOUND_EXISTING", "notes": "Existing artifact only; not generated by this script."})
    if not rows:
        rows.append({"path": "", "kind": "", "status": "NOT_AVAILABLE", "notes": "No ablation or robustness artifacts were found."})
    return rows


def write_summary(path, checks, source_rows, subgroup_rows, threshold_rows, calibration_rows, ablation_rows):
    pass_count = sum(1 for row in checks if row["status"] == "PASS")
    fail_count = sum(1 for row in checks if row["status"] == "FAIL")
    conditional_count = sum(1 for row in checks if row["status"] == "CONDITIONAL")
    not_available_count = sum(1 for row in checks if row["status"] == "NOT_AVAILABLE")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("# Internal Robustness Summary\n\n")
        f.write("This report uses existing final out-of-fold predictions and statistics reporting artifacts only.\n")
        f.write("It is an internal robustness analysis under patient-grouped cross-validation, not an independent cohort study.\n\n")
        f.write("## Validation Checks\n\n")
        f.write(f"- PASS: {pass_count}\n")
        f.write(f"- FAIL: {fail_count}\n")
        f.write(f"- CONDITIONAL: {conditional_count}\n")
        f.write(f"- NOT_AVAILABLE: {not_available_count}\n\n")
        f.write("## Analyses Generated\n\n")
        f.write(f"- Source-wise rows: {len(source_rows)}\n")
        f.write(f"- Subgroup rows: {len(subgroup_rows)}\n")
        f.write(f"- Threshold robustness rows: {len(threshold_rows)}\n")
        f.write(f"- Calibration robustness rows: {len(calibration_rows)}\n")
        f.write(f"- Ablation/window inventory rows: {len(ablation_rows)}\n\n")
        f.write("## Interpretation Guardrails\n\n")
        f.write("- Patient-level and lead-time best-method summary values are checked from final JSON summaries or comparison_summary.csv.\n")
        f.write("- Operating-point patient tables remain descriptive and are not used for the best-method summary check.\n")
        f.write("- Utility is threshold-dependent.\n")
        f.write("- Enhanced did not uniformly improve Utility in the final internal run.\n")
        f.write("- Baseline had slightly higher best raw Utility in the final internal run.\n")
        f.write("- Threshold 0.5 is not an optimized clinical operating point.\n")
        f.write("- No official hidden-test evaluation, MIMIC-IV, eICU, or prospective validation was performed.\n")
        f.write("- Claims should remain restricted to patient-grouped internal cross-validation.\n")


def run(args):
    checks = []
    validate_inputs(args, checks, args.allow_missing_inputs)
    validate_methodology_audit(args, checks)

    output_dir = args.output_dir
    inventory_dir = args.inventory_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    inventory_dir.mkdir(parents=True, exist_ok=True)

    try:
        np, pd, _, _ = import_analysis_stack()
        base, base_oof_path = resolve_oof(args.baseline_dir, pd)
        enhanced, enhanced_oof_path = resolve_oof(args.enhanced_dir, pd)
        base = normalize_oof_columns(base)
        enhanced = normalize_oof_columns(enhanced)
        validate_oof("baseline", base, base_oof_path, checks)
        validate_oof("enhanced", enhanced, enhanced_oof_path, checks)
        validate_alignment(base, enhanced, checks)
        validate_expected_metrics("baseline", args.baseline_dir, checks)
        validate_expected_metrics("enhanced", args.enhanced_dir, checks)
        validate_statistics_patient_metrics("baseline", args.statistics_dir, args.baseline_dir, checks)
        validate_statistics_patient_metrics("enhanced", args.statistics_dir, args.enhanced_dir, checks)
        base, enhanced = add_sources_from_harmonized(base, enhanced, args.harmonized, checks)
        utility_helpers = load_utility_helpers()

        source_rows = build_sourcewise_metrics(base, enhanced, utility_helpers)
        subgroup_rows = build_subgroup_metrics(base, enhanced, utility_helpers)
        threshold_rows = build_threshold_robustness(base, enhanced, utility_helpers)
        calibration_rows = build_calibration_robustness(base, enhanced)
    except Exception as exc:
        if not args.allow_missing_inputs:
            raise
        status_check(checks, "robustness_data_load", "NOT_AVAILABLE", f"internal robustness data-dependent outputs skipped: {exc}", "")
        source_rows = [{"group": "SourceSet", "level": "NOT_AVAILABLE", "status": "NOT_AVAILABLE", "reason": str(exc)}]
        subgroup_rows = [{"group": "all", "level": "NOT_AVAILABLE", "status": "NOT_AVAILABLE", "reason": str(exc)}]
        threshold_rows = [{"status": "NOT_AVAILABLE", "reason": str(exc)}]
        calibration_rows = [{"status": "NOT_AVAILABLE", "reason": str(exc)}]

    ablation_rows = build_ablation_inventory(args)

    check_fields = ["check_id", "status", "details", "evidence"]
    source_fields = [
        "group", "level", "model", "status", "reason", "n_time_steps", "n_patients",
        "row_positives", "row_negatives", "patient_positives", "patient_negatives",
        "time_step_prevalence", "patient_prevalence", "auroc_raw", "auprc_raw",
        "delta_auroc_raw", "delta_auprc_raw", "brier_raw", "ece_raw",
        "utility_raw_at_0.5", "utility_source", "sensitivity_at_0.5",
        "specificity_at_0.5", "patient_sensitivity_at_0.5", "patient_specificity_at_0.5",
    ]
    subgroup_fields = source_fields
    threshold_fields = [
        "model", "method", "probability_column", "threshold", "sensitivity", "specificity",
        "precision", "recall", "f1", "utility", "utility_source",
        "patient_sensitivity", "patient_specificity", "n_alerted_patients", "status", "reason",
    ]
    calibration_fields = ["model", "method", "probability_column", "brier", "ece", "brier_change_vs_raw", "ece_change_vs_raw", "calibration_scope", "status", "reason"]
    ablation_fields = ["path", "kind", "status", "notes"]

    write_csv(output_dir / "validation_checks.csv", checks, check_fields)
    write_markdown_table(output_dir / "validation_checks.md", "Internal Robustness Validation Checks", checks, check_fields)
    write_csv(output_dir / "sourcewise_metrics.csv", source_rows, source_fields)
    write_csv(output_dir / "subgroup_metrics.csv", subgroup_rows, subgroup_fields)
    write_csv(output_dir / "threshold_robustness.csv", threshold_rows, threshold_fields)
    write_csv(output_dir / "calibration_robustness.csv", calibration_rows, calibration_fields)
    write_csv(output_dir / "ablation_inventory.csv", ablation_rows, ablation_fields)
    write_summary(output_dir / "internal_robustness_summary.md", checks, source_rows, subgroup_rows, threshold_rows, calibration_rows, ablation_rows)

    write_csv(inventory_dir / "internal_robustness_checklist.csv", checks, check_fields)
    write_markdown_table(inventory_dir / "internal_robustness_checklist.md", "Internal Robustness Checklist", checks, check_fields)
    (inventory_dir / "internal_robustness_summary.md").write_text(
        (output_dir / "internal_robustness_summary.md").read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_scope": "internal_oof_robustness",
        "inputs": {
            "baseline_dir": str(args.baseline_dir),
            "enhanced_dir": str(args.enhanced_dir),
            "statistics_dir": str(args.statistics_dir),
            "harmonized": str(args.harmonized),
        },
        "outputs": sorted(path.name for path in output_dir.iterdir() if path.is_file()),
        "inventory_outputs": [
            "internal_robustness_summary.md",
            "internal_robustness_checklist.csv",
            "internal_robustness_checklist.md",
        ],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

def parse_args():
    parser = argparse.ArgumentParser(description="Build internal robustness outputs from final OOF artifacts.")
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--enhanced-dir", type=Path, required=True)
    parser.add_argument("--statistics-dir", type=Path, required=True)
    parser.add_argument("--harmonized", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--inventory-dir", type=Path, required=True)
    parser.add_argument("--allow-missing-inputs", action="store_true", help="Write NOT_AVAILABLE outputs instead of failing when final artifacts are absent.")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    return parser.parse_args()


def main():
    args = parse_args()
    args.project_root = args.project_root.resolve()
    run(args)


if __name__ == "__main__":
    main()
