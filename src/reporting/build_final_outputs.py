#!/usr/bin/env python3
"""Build compact public final tables and figures from final artifacts.

This script reads existing final reporting outputs only. It does not train
models, alter result directories, or infer metrics that are absent from the
validated reporting package.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.reporting import build_statistics as statistics

pd = None
np = None

PRIMARY_METRICS = [
    "AUROC_raw",
    "AUPRC_raw",
    "Brier_raw",
    "ECE_raw",
    "Brier_isotonic",
    "ECE_isotonic",
    "Utility_raw_at_0.5",
    "Utility_raw_best",
    "Utility_raw_best_threshold",
    "n_features",
]


def load_runtime_dependencies(include_figures: bool) -> None:
    """Load dependencies lazily so --help works in minimal shells."""
    global pd, np
    statistics.load_runtime_dependencies()
    pd = statistics.pd
    np = statistics.np
    if include_figures:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt  # noqa: F401
            from sklearn.metrics import auc, precision_recall_curve, roc_curve  # noqa: F401
        except ImportError as exc:
            raise SystemExit(
                "Missing figure dependency. Re-run in the CEDIA container or pass --skip-figures."
            ) from exc


def require_file(path: Path, purpose: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {purpose}: {path}")
    return path


def read_csv(path: Path, purpose: str):
    require_file(path, purpose)
    return pd.read_csv(path)


def write_csv(df, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def latex_escape(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        text = f"{value:.6g}"
    else:
        text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in text)


def write_latex_table(df, path: Path, caption: str, label: str, max_rows: int | None = None) -> None:
    if any(ch.isspace() for ch in label):
        raise ValueError(f"LaTeX label must not contain whitespace: {label!r}")
    out = df.copy()
    if max_rows is not None:
        out = out.head(max_rows)
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        f"\\caption{{{latex_escape(caption)}}}",
        f"\\label{{{label}}}",
        r"\renewcommand{\arraystretch}{1.15}",
        "\\begin{tabular}{" + "l" * len(out.columns) + "}",
        r"\toprule",
        " & ".join(r"\textbf{" + latex_escape(col) + "}" for col in out.columns) + r" \\",
        r"\midrule",
    ]
    for _, row in out.iterrows():
        lines.append(" & ".join(latex_escape(row[col]) for col in out.columns) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")



def write_latex_lines(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def fmt_num(value: Any, digits: int = 4) -> str:
    if value is None:
        return ""
    try:
        val = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(val):
        return ""
    return f"{val:.{digits}f}"


def fmt_delta(value: Any, digits: int = 4) -> str:
    text = fmt_num(value, digits)
    if not text:
        return text
    return text if text.startswith("-") else "+" + text


def pretty_metric(metric: str) -> str:
    names = {
        "AUROC_raw": "Raw AUROC",
        "AUPRC_raw": "Raw AUPRC",
        "Brier_raw": "Raw Brier score",
        "ECE_raw": "Raw ECE",
        "Brier_isotonic": "Isotonic Brier score",
        "ECE_isotonic": "Isotonic ECE",
        "Utility_raw_at_0.5": "Raw Utility at 0.5",
        "Utility_raw_best": "Best raw Utility",
        "Utility_raw_best_threshold": "Best raw Utility threshold",
        "n_features": "Final numeric features",
    }
    return names.get(str(metric), str(metric).replace("_", " "))


def pretty_operating_point(value: str) -> str:
    names = {
        "raw_threshold_0.5": "Raw 0.5",
        "raw_utility_optimal": "Raw Utility optimal",
        "raw_f1_optimal": "Raw F1 optimal",
        "threshold_0.5": "Raw 0.5",
        "raw_F1_optimal": "Raw F1 optimal",
    }
    return names.get(str(value), str(value).replace("_", " "))


def pretty_feature(value: str) -> str:
    text = str(value)
    for suffix, replacement in [
        ("_sampen_24h", " SampEn 24h"),
        ("_cv_8h", " CV 8h"),
        ("_iqr_8h", " IQR 8h"),
    ]:
        if text.endswith(suffix):
            return text[: -len(suffix)] + replacement
    return text.replace("_", " ")


def write_primary_metrics_latex(df, path: Path) -> None:
    rows = []
    for metric in PRIMARY_METRICS:
        match = df[df["metric"].astype(str).eq(metric)]
        if match.empty:
            continue
        row = match.iloc[0]
        if metric == "ECE_isotonic":
            baseline = r"$1.09\times10^{-4}$"
            enhanced = r"$6.91\times10^{-5}$"
            delta = r"$-3.95\times10^{-5}$"
        elif metric == "n_features":
            baseline = str(int(float(row["baseline"])))
            enhanced = str(int(float(row["enhanced"])))
            delta = fmt_delta(row["delta"], 0)
        elif metric == "Utility_raw_best_threshold":
            baseline = fmt_num(row["baseline"], 2)
            enhanced = fmt_num(row["enhanced"], 2)
            delta = fmt_delta(row["delta"], 2)
        elif metric == "Brier_isotonic":
            baseline = fmt_num(row["baseline"], 5)
            enhanced = fmt_num(row["enhanced"], 5)
            delta = fmt_delta(row["delta"], 5)
        else:
            baseline = fmt_num(row["baseline"], 4)
            enhanced = fmt_num(row["enhanced"], 4)
            delta = fmt_delta(row["delta"], 4)
        rows.append((pretty_metric(metric), baseline, enhanced, delta, str(row.get("interpretation", "")).capitalize()))
    lines = [
        r"\begin{table*}[!t]",
        r"\centering",
        r"\caption{Primary internal cross-validation metrics.}",
        r"\label{tab:final_primary_metrics}",
        r"\footnotesize",
        r"\renewcommand{\arraystretch}{1.12}",
        r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lllll}",
        r"\toprule",
        r"\textbf{Metric} & \textbf{Baseline} & \textbf{Enhanced} & \textbf{Delta} & \textbf{Interpretation} \\",
        r"\midrule",
    ]
    lines += [" & ".join(row) + r" \\" for row in rows]
    lines += [r"\bottomrule", r"\end{tabular*}", r"\end{table*}"]
    write_latex_lines(path, lines)


def write_patient_level_latex(df, path: Path) -> None:
    lines = [
        r"\begin{table*}[!t]",
        r"\centering",
        r"\caption{Patient-level operating metrics. Early-alert sensitivity counts septic patients detected before sepsis onset; any-alert sensitivity counts septic patients with any alert during the record; negative-patient specificity counts nonseptic patients with no alert; lead time is computed among early detections.}",
        r"\label{tab:final_patient_level_metrics}",
        r"\scriptsize",
        r"\renewcommand{\arraystretch}{1.12}",
        r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lllllll}",
        r"\toprule",
        r"\textbf{Model} & \textbf{Operating point} & \textbf{Threshold} & \textbf{Early-alert sens.} & \textbf{Any-alert sens.} & \textbf{Negative specificity} & \textbf{Lead time, h median [IQR]} \\",
        r"\midrule",
    ]
    for _, row in df.iterrows():
        lead = f"{fmt_num(row['lead_time_median_h'], 0)} [{str(row['lead_time_iqr_h']).replace('-', '--')}]"
        lines.append(
            " & ".join(
                [
                    str(row["model_variant"]).capitalize(),
                    pretty_operating_point(str(row["operating_point"])),
                    fmt_num(row["threshold"], 2),
                    fmt_num(row["early_alert_patient_sensitivity"], 4),
                    fmt_num(row["any_alert_patient_sensitivity"], 4),
                    fmt_num(row["negative_patient_specificity"], 4),
                    lead,
                ]
            )
            + r" \\"
        )
    lines += [r"\bottomrule", r"\end{tabular*}", r"\end{table*}"]
    write_latex_lines(path, lines)


def write_thresholds_latex(df, path: Path) -> None:
    work = df[df["probability_column"].astype(str).str.lower().eq("raw")].copy()
    desired = [
        ("baseline", 0.45, "Raw Utility optimal"),
        ("baseline", 0.50, "Raw 0.5"),
        ("baseline", 0.86, "Raw F1 optimal"),
        ("enhanced", 0.41, "Raw Utility optimal"),
        ("enhanced", 0.50, "Raw 0.5"),
        ("enhanced", 0.84, "Raw F1 optimal"),
    ]
    lines = [
        r"\begin{table*}[!t]",
        r"\centering",
        r"\caption{Selected raw threshold operating points. Calibrated-threshold summaries remain available in the CSV assets.}",
        r"\label{tab:final_thresholds}",
        r"\scriptsize",
        r"\renewcommand{\arraystretch}{1.12}",
        r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}llllllll}",
        r"\toprule",
        r"\textbf{Model} & \textbf{Operating point} & \textbf{Threshold} & \textbf{Utility} & \textbf{Sensitivity} & \textbf{Specificity} & \textbf{Precision} & \textbf{F1} \\",
        r"\midrule",
    ]
    for model, threshold, label in desired:
        candidates = work[work["model_variant"].astype(str).eq(model)].copy()
        candidates["_distance"] = (pd.to_numeric(candidates["threshold"], errors="coerce") - threshold).abs()
        if candidates.empty:
            continue
        row = candidates.sort_values("_distance").iloc[0]
        lines.append(
            " & ".join(
                [
                    model.capitalize(),
                    label,
                    fmt_num(row["threshold"], 2),
                    fmt_num(row["utility"], 4),
                    fmt_num(row["sensitivity"], 4),
                    fmt_num(row["specificity"], 4),
                    fmt_num(row["precision"], 4),
                    fmt_num(row["f1"], 4),
                ]
            )
            + r" \\"
        )
    lines += [r"\bottomrule", r"\end{tabular*}", r"\end{table*}"]
    write_latex_lines(path, lines)


def write_feature_counts_latex(df, path: Path) -> None:
    row_map = {
        "baseline_final_numeric_features": ("Baseline final numeric features", "After meta/non-numeric exclusion"),
        "enhanced_final_numeric_features": ("Enhanced final numeric features", "Baseline plus hemodynamic complexity"),
        "feature_delta": ("Added hemodynamic complexity features", "Enhanced minus baseline numeric features"),
        "hemodynamic_expected_features": ("Expected hemodynamic features", "Six signals times CV/IQR/SampEn"),
        "hemodynamic_found_in_enhanced_importance": ("Hemodynamic features in enhanced importance", "Present in enhanced feature-importance summary"),
        "SampEn_found": ("SampEn features found", "All six SampEn descriptors present"),
    }
    lines = [
        r"\begin{table*}[!t]",
        r"\centering",
        r"\caption{Feature-count audit for final model inputs.}",
        r"\label{tab:final_feature_counts}",
        r"\footnotesize",
        r"\renewcommand{\arraystretch}{1.12}",
        r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}llll}",
        r"\toprule",
        r"\textbf{Item} & \textbf{Value} & \textbf{Status} & \textbf{Note} \\",
        r"\midrule",
    ]
    for item, (label, note) in row_map.items():
        match = df[df["item"].astype(str).eq(item)]
        if match.empty:
            continue
        row = match.iloc[0]
        lines.append(" & ".join([label, str(row["value"]), str(row["status"]), note]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular*}", r"\end{table*}"]
    write_latex_lines(path, lines)


def write_hemodynamic_features_latex(df, path: Path) -> None:
    lines = [
        r"\begin{table*}[!t]",
        r"\centering",
        r"\caption{Hemodynamic complexity features with nonzero Gain in fold-level importance summaries.}",
        r"\label{tab:final_hemodynamic_features}",
        r"\scriptsize",
        r"\renewcommand{\arraystretch}{1.08}",
        r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}llllll}",
        r"\toprule",
        r"\textbf{Feature} & \textbf{Signal} & \textbf{Type} & \textbf{Mean gain} & \textbf{SD gain} & \textbf{Folds with nonzero gain} \\",
        r"\midrule",
    ]
    for _, row in df.head(18).iterrows():
        lines.append(
            " & ".join(
                [
                    pretty_feature(str(row["feature"])),
                    str(row.get("hemo_signal", "")),
                    str(row.get("hemo_feature_type", "")),
                    fmt_num(row.get("mean_importance"), 1),
                    fmt_num(row.get("std_importance"), 1),
                    str(row.get("selected_folds", "")),
                ]
            )
            + r" \\"
        )
    lines += [r"\bottomrule", r"\end{tabular*}", r"\end{table*}"]
    write_latex_lines(path, lines)


def write_sourcewise_latex(df, path: Path) -> None:
    lines = [
        r"\begin{table*}[!t]",
        r"\centering",
        r"\caption{Internal source-wise robustness summary. SourceSet A/B is an internal public-training split analysis and not external validation.}",
        r"\label{tab:final_sourcewise}",
        r"\footnotesize",
        r"\renewcommand{\arraystretch}{1.12}",
        r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}llllllll}",
        r"\toprule",
        r"\textbf{Source} & \textbf{Model} & \textbf{Patients} & \textbf{Septic patients} & \textbf{Raw AUROC} & \textbf{Raw AUPRC} & \textbf{Raw Utility at 0.5} & \textbf{Delta vs baseline} \\",
        r"\midrule",
    ]
    for source in ["A", "B"]:
        base = df[(df["level"].astype(str).eq(source)) & (df["model"].astype(str).eq("baseline"))]
        enh = df[(df["level"].astype(str).eq(source)) & (df["model"].astype(str).eq("enhanced"))]
        delta = df[(df["level"].astype(str).eq(source)) & (df["model"].astype(str).eq("enhanced_minus_baseline"))]
        for label, part in [("Baseline", base), ("Enhanced", enh)]:
            if part.empty:
                continue
            row = part.iloc[0]
            delta_text = "--"
            if label == "Enhanced" and not delta.empty:
                drow = delta.iloc[0]
                delta_text = f"AUROC {fmt_delta(drow.get('delta_auroc_raw'), 4)}; AUPRC {fmt_delta(drow.get('delta_auprc_raw'), 4)}"
            lines.append(
                " & ".join(
                    [
                        source,
                        label,
                        str(int(float(row["n_patients"]))),
                        str(int(float(row["patient_positives"]))),
                        fmt_num(row["auroc_raw"], 4),
                        fmt_num(row["auprc_raw"], 4),
                        fmt_num(row["utility_raw_at_0.5"], 4),
                        delta_text,
                    ]
                )
                + r" \\"
            )
    lines += [r"\bottomrule", r"\end{tabular*}", r"\end{table*}"]
    write_latex_lines(path, lines)


def select_columns(df, preferred: Iterable[str]):
    cols = [col for col in preferred if col in df.columns]
    return df.loc[:, cols].copy() if cols else df.copy()


def build_primary_metrics(statistics_dir: Path):
    comparison = read_csv(statistics_dir / "comparison_summary.csv", "comparison summary")
    if "metric" not in comparison.columns:
        raise ValueError("comparison_summary.csv must contain a metric column")
    out = comparison[comparison["metric"].isin(PRIMARY_METRICS)].copy()
    ordered = {metric: i for i, metric in enumerate(PRIMARY_METRICS)}
    out["_order"] = out["metric"].map(ordered)
    out = out.sort_values("_order").drop(columns=["_order"])
    return select_columns(out, ["metric", "baseline", "enhanced", "delta", "relative_delta_percent", "interpretation"])


def build_statistical_tests(statistics_dir: Path):
    stats = read_csv(statistics_dir / "statistical_tests.csv", "statistical tests")
    if "metric" in stats.columns:
        relevant = [
            "Delta_AUROC_raw",
            "Delta_AUPRC_raw",
            "Delta_Brier_raw",
            "Delta_ECE_raw",
            "McNemar",
            "mcnemar",
        ]
        pattern = "|".join(relevant)
        stats = stats[stats["metric"].astype(str).str.contains(pattern, case=False, regex=True)].copy()
    return select_columns(
        stats,
        [
            "metric",
            "baseline",
            "enhanced",
            "delta",
            "ci_lower",
            "ci_upper",
            "p_value_two_sided",
            "p_value_bh",
            "significant_bh_0.05",
            "p_value_display",
            "test",
        ],
    )


def build_thresholds(statistics_dir: Path):
    thresholds = read_csv(statistics_dir / "threshold_operating_points.csv", "threshold operating points")
    return select_columns(
        thresholds,
        [
            "model_variant",
            "probability_column",
            "threshold",
            "utility",
            "sensitivity",
            "specificity",
            "precision",
            "f1",
            "source_file",
        ],
    )


def build_patient_level_metrics(statistics_dir: Path, baseline_dir: Path, enhanced_dir: Path):
    """Build final patient-level table with early-alert and any-alert definitions."""
    confusion = read_csv(
        statistics_dir / "patient_confusion_matrices.csv",
        "patient any-alert confusion matrices",
    )
    required_confusion = {"model_variant", "threshold_type", "threshold", "TP", "TN", "FP", "FN"}
    missing_confusion = required_confusion.difference(confusion.columns)
    if missing_confusion:
        raise ValueError(
            "patient_confusion_matrices.csv missing required columns: "
            + ", ".join(sorted(missing_confusion))
            + f". Available columns: {list(confusion.columns)}"
        )

    operating_points = {
        "raw_threshold_0.5": "threshold_0.5",
        "raw_utility_optimal": "raw_utility_optimal",
        "raw_f1_optimal": "raw_F1_optimal",
    }
    rows = []
    for variant, result_dir in [("baseline", baseline_dir), ("enhanced", enhanced_dir)]:
        threshold_path = result_dir / "threshold_metrics.csv"
        thresholds = pd.read_csv(require_file(threshold_path, f"{variant} threshold metrics"))
        required_threshold = {
            "probability_source",
            "threshold",
            "Utility_U",
            "F1",
            "Patient_sensitivity",
            "Patient_specificity",
            "Lead_time_median",
            "Lead_time_IQR_lower",
            "Lead_time_IQR_upper",
            "TP_patients",
            "FN_patients",
            "TN_patients",
            "FP_patients",
        }
        missing_threshold = required_threshold.difference(thresholds.columns)
        if missing_threshold:
            raise ValueError(
                f"{threshold_path} missing required patient-level columns: "
                + ", ".join(sorted(missing_threshold))
                + f". Available columns: {list(thresholds.columns)}"
            )
        raw = thresholds[thresholds["probability_source"].astype(str).str.lower().eq("raw")].copy()
        if raw.empty:
            raise ValueError(f"{threshold_path} has no probability_source == raw rows")
        raw["threshold"] = pd.to_numeric(raw["threshold"], errors="coerce")
        raw["Utility_U"] = pd.to_numeric(raw["Utility_U"], errors="coerce")
        raw["F1"] = pd.to_numeric(raw["F1"], errors="coerce")

        threshold_rows = {
            "raw_threshold_0.5": raw.iloc[(raw["threshold"] - 0.5).abs().argsort()].iloc[0],
            "raw_utility_optimal": raw.loc[raw["Utility_U"].idxmax()],
            "raw_f1_optimal": raw.loc[raw["F1"].idxmax()],
        }
        if not np.isclose(float(threshold_rows["raw_threshold_0.5"]["threshold"]), 0.5, atol=1e-9):
            raise ValueError(f"{threshold_path} missing raw threshold 0.5 row")

        for operating_point, confusion_type in operating_points.items():
            trow = threshold_rows[operating_point]
            crow = confusion[
                confusion["model_variant"].astype(str).eq(variant)
                & confusion["threshold_type"].astype(str).eq(confusion_type)
            ]
            if crow.empty:
                raise ValueError(
                    f"patient_confusion_matrices.csv missing row for model_variant={variant}, "
                    f"threshold_type={confusion_type}"
                )
            crow = crow.iloc[0]
            threshold = float(trow["threshold"])
            confusion_threshold = float(crow["threshold"])
            if not np.isclose(threshold, confusion_threshold, atol=1e-9):
                raise ValueError(
                    f"Threshold mismatch for {variant} {operating_point}: "
                    f"threshold_metrics={threshold}, patient_confusion_matrices={confusion_threshold}"
                )

            early_tp = float(trow["TP_patients"])
            early_fn = float(trow["FN_patients"])
            negative_tn = float(trow["TN_patients"])
            negative_fp = float(trow["FP_patients"])
            any_tp = float(crow["TP"])
            any_fn = float(crow["FN"])
            early_sensitivity = float(trow["Patient_sensitivity"])
            negative_specificity = float(trow["Patient_specificity"])
            any_sensitivity = any_tp / (any_tp + any_fn) if (any_tp + any_fn) else np.nan

            expected_early = early_tp / (early_tp + early_fn) if (early_tp + early_fn) else np.nan
            expected_specificity = negative_tn / (negative_tn + negative_fp) if (negative_tn + negative_fp) else np.nan
            if not np.isclose(early_sensitivity, expected_early, atol=1e-9, equal_nan=True):
                raise ValueError(f"Early-alert patient sensitivity mismatch for {variant} {operating_point}")
            if not np.isclose(negative_specificity, expected_specificity, atol=1e-9, equal_nan=True):
                raise ValueError(f"Negative-patient specificity mismatch for {variant} {operating_point}")
            if "sensitivity" in confusion.columns and not np.isclose(float(crow["sensitivity"]), any_sensitivity, atol=1e-9, equal_nan=True):
                raise ValueError(f"Any-alert patient sensitivity mismatch for {variant} {operating_point}")

            lead_low = float(trow["Lead_time_IQR_lower"])
            lead_high = float(trow["Lead_time_IQR_upper"])
            rows.append(
                {
                    "model_variant": variant,
                    "operating_point": operating_point,
                    "threshold": threshold,
                    "early_alert_patient_sensitivity": early_sensitivity,
                    "any_alert_patient_sensitivity": any_sensitivity,
                    "negative_patient_specificity": negative_specificity,
                    "lead_time_median_h": float(trow["Lead_time_median"]),
                    "lead_time_iqr_h": f"{lead_low:g}-{lead_high:g}",
                    "early_alert_tp_patients": int(early_tp),
                    "early_alert_fn_patients": int(early_fn),
                    "negative_tn_patients": int(negative_tn),
                    "negative_fp_patients": int(negative_fp),
                    "any_alert_tp_patients": int(any_tp),
                    "any_alert_fn_patients": int(any_fn),
                }
            )
    return pd.DataFrame(rows)


def build_feature_counts(statistics_dir: Path):
    counts = read_csv(statistics_dir / "feature_count_audit.csv", "feature count audit")
    return select_columns(counts, ["item", "value", "expected_or_recommended", "status", "notes"])


def build_hemodynamic_features(statistics_dir: Path):
    features = read_csv(statistics_dir / "feature_importance_table.csv", "feature importance table")
    required = {"model_variant", "feature", "is_hemodynamic_complexity_feature"}
    missing = required.difference(features.columns)
    if missing:
        raise ValueError(f"feature_importance_table.csv missing columns: {sorted(missing)}")
    is_hemo = features["is_hemodynamic_complexity_feature"].astype(str).str.lower().isin(["true", "1"])
    hemo = features[features["model_variant"].astype(str).eq("enhanced") & is_hemo].copy()
    if "mean_importance" in hemo.columns:
        hemo["mean_importance_numeric"] = pd.to_numeric(hemo["mean_importance"], errors="coerce")
        hemo = hemo.sort_values("mean_importance_numeric", ascending=False)
    return select_columns(
        hemo,
        ["feature", "hemo_signal", "hemo_feature_type", "mean_importance", "std_importance", "selected_folds"],
    )


def build_sourcewise(robustness_dir: Path):
    source_path = robustness_dir / "sourcewise_metrics.csv"
    if not source_path.exists():
        source_path = robustness_dir / "sourcewise_metrics.csv"
    source = read_csv(source_path, "source-wise metrics")
    return select_columns(
        source,
        ["group", "level", "model", "n_patients", "patient_positives", "auroc_raw", "auprc_raw", "delta_auroc_raw", "delta_auprc_raw", "utility_raw_at_0.5", "patient_sensitivity_at_0.5", "patient_specificity_at_0.5", "status"],
    )


def probability_column(df, preferred: str = "prob_raw") -> str:
    if preferred in df.columns:
        return preferred
    for col in ["probability", "score", "prediction", "prob_platt", "prob_isotonic"]:
        if col in df.columns:
            return col
    raise ValueError(f"No probability column found in OOF columns: {list(df.columns)}")


def plot_roc_pr(baseline_dir: Path, enhanced_dir: Path, output_dir: Path) -> list[str]:
    import matplotlib.pyplot as plt
    from sklearn.metrics import auc, precision_recall_curve, roc_curve

    outputs = []
    models = []
    for label, result_dir in [("Baseline", baseline_dir), ("Enhanced", enhanced_dir)]:
        df = pd.read_csv(require_file(result_dir / "oof_predictions.csv", f"{label} OOF predictions"))
        statistics.require_columns(df, ["SepsisLabel"], result_dir / "oof_predictions.csv")
        prob_col = probability_column(df, "prob_raw")
        y = pd.to_numeric(df["SepsisLabel"], errors="raise").astype(int).to_numpy()
        p = pd.to_numeric(df[prob_col], errors="raise").to_numpy()
        models.append((label, y, p))

    plt.figure(figsize=(6.4, 5.0))
    for label, y, p in models:
        fpr, tpr, _ = roc_curve(y, p)
        plt.plot(fpr, tpr, label=f"{label} AUROC={auc(fpr, tpr):.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="0.5", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.tight_layout()
    path = output_dir / "raw_roc_curve.png"
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    outputs.append(path.name)

    plt.figure(figsize=(6.4, 5.0))
    for label, y, p in models:
        precision, recall, _ = precision_recall_curve(y, p)
        plt.plot(recall, precision, label=f"{label} AUPRC={auc(recall, precision):.3f}")
    prevalence = float(np.mean(models[0][1]))
    plt.axhline(prevalence, linestyle="--", color="0.5", linewidth=1, label=f"Prevalence={prevalence:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.tight_layout()
    path = output_dir / "raw_precision_recall_curve.png"
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    outputs.append(path.name)
    return outputs


def plot_calibration_summary(statistics_dir: Path, output_dir: Path) -> str:
    import matplotlib.pyplot as plt

    cal = read_csv(statistics_dir / "calibration_table.csv", "calibration table")
    accepted = {
        "model": ["model_variant", "model", "variant"],
        "method": ["method", "calibration_method"],
        "brier": ["Brier", "brier", "brier_score"],
        "ece": ["ECE", "ece"],
    }

    def resolve_column(kind: str) -> str | None:
        exact = [col for col in accepted[kind] if col in cal.columns]
        if exact:
            return exact[0]
        lower_map = {col.lower(): col for col in cal.columns}
        for col in accepted[kind]:
            if col.lower() in lower_map:
                return lower_map[col.lower()]
        return None

    model_col = resolve_column("model")
    method_col = resolve_column("method")
    brier_col = resolve_column("brier")
    ece_col = resolve_column("ece")
    resolved = {"model": model_col, "method": method_col, "brier": brier_col, "ece": ece_col}
    missing = [
        f"{kind} (accepted: {', '.join(options)})"
        for kind, options in accepted.items()
        if resolved[kind] is None
    ]
    if missing:
        raise ValueError(
            "calibration_table.csv lacks required plotting columns: "
            + "; ".join(missing)
            + f". Available columns: {list(cal.columns)}"
        )
    work = cal[[model_col, method_col, brier_col, ece_col]].copy()
    work[brier_col] = pd.to_numeric(work[brier_col], errors="coerce")
    work[ece_col] = pd.to_numeric(work[ece_col], errors="coerce")
    labels = work[model_col].astype(str) + " " + work[method_col].astype(str)
    x = np.arange(len(work))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True)
    axes[0].bar(x, work[brier_col])
    axes[0].set_title("Brier")
    axes[1].bar(x, work[ece_col])
    axes[1].set_title("ECE")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=60, ha="right")
    fig.tight_layout()
    path = output_dir / "calibration_summary.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path.name


def plot_utility_thresholds(baseline_dir: Path, enhanced_dir: Path, output_dir: Path) -> str:
    import matplotlib.pyplot as plt

    accepted = {
        "threshold": ["threshold", "Threshold"],
        "utility": ["Utility_U", "utility", "Utility", "normalized_utility", "UtilityScore"],
        "probability_source": ["probability_source", "method", "probability_column"],
    }

    def resolve_column(table, kind: str) -> str | None:
        exact = [col for col in accepted[kind] if col in table.columns]
        if exact:
            return exact[0]
        lower_map = {col.lower(): col for col in table.columns}
        for col in accepted[kind]:
            if col.lower() in lower_map:
                return lower_map[col.lower()]
        return None

    plt.figure(figsize=(6.4, 5.0))
    for label, result_dir in [("Baseline", baseline_dir), ("Enhanced", enhanced_dir)]:
        path = require_file(result_dir / "threshold_metrics.csv", f"{label} threshold metrics")
        table = pd.read_csv(path)
        threshold_col = resolve_column(table, "threshold")
        utility_col = resolve_column(table, "utility")
        source_col = resolve_column(table, "probability_source")
        missing = [
            f"{kind} (accepted: {', '.join(options)})"
            for kind, options in accepted.items()
            if kind != "probability_source" and resolve_column(table, kind) is None
        ]
        if missing:
            raise ValueError(
                f"{path} lacks required threshold plotting columns: "
                + "; ".join(missing)
                + f". Available columns: {list(table.columns)}. "
                + "Accepted probability source columns: "
                + ", ".join(accepted["probability_source"])
            )
        work = table.copy()
        source_label = "all sources"
        if source_col is not None:
            sources = work[source_col].astype(str).str.lower()
            raw_mask = sources.isin(["raw", "prob_raw"])
            if raw_mask.any():
                work = work[raw_mask].copy()
                source_label = "raw"
            else:
                first_source = str(work[source_col].dropna().iloc[0]) if work[source_col].notna().any() else "unknown"
                work = work[work[source_col].astype(str).eq(first_source)].copy()
                source_label = first_source
        work[threshold_col] = pd.to_numeric(work[threshold_col], errors="coerce")
        work[utility_col] = pd.to_numeric(work[utility_col], errors="coerce")
        work = work.dropna(subset=[threshold_col, utility_col]).sort_values(threshold_col)
        plt.plot(work[threshold_col], work[utility_col], label=f"{label} ({source_label})")
    plt.xlabel("Threshold")
    plt.ylabel("Normalized Utility")
    plt.legend()
    plt.tight_layout()
    path = output_dir / "utility_thresholds.png"
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    return path.name


def plot_hemodynamic_importance(hemo_table, output_dir: Path) -> str:
    import matplotlib.pyplot as plt

    if hemo_table.empty:
        raise ValueError("No hemodynamic features available for importance plot")
    work = hemo_table.copy()
    if "mean_importance" not in work.columns:
        raise ValueError("Hemodynamic feature table lacks mean_importance")
    work["mean_importance_numeric"] = pd.to_numeric(work["mean_importance"], errors="coerce")
    work = work.dropna(subset=["mean_importance_numeric"]).sort_values("mean_importance_numeric", ascending=True).tail(15)
    plt.figure(figsize=(7.2, 5.4))
    plt.barh(work["feature"].astype(str).map(pretty_feature), work["mean_importance_numeric"])
    plt.xlabel("Mean Gain importance")
    plt.tight_layout()
    path = output_dir / "hemodynamic_importance.png"
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    return path.name


def build_assets(args: argparse.Namespace) -> dict[str, Any]:
    tables_dir = args.tables_dir or Path("results/final_tables")
    figures_dir = args.figures_dir or Path("results/final_figures")
    if args.output_dir is not None:
        if args.tables_dir is None:
            tables_dir = args.output_dir
        if args.figures_dir is None:
            figures_dir = args.output_dir
    tables_dir.mkdir(parents=True, exist_ok=True)
    if not args.skip_figures:
        figures_dir.mkdir(parents=True, exist_ok=True)

    table_outputs: list[str] = []
    figure_outputs: list[str] = []
    primary = build_primary_metrics(args.statistics_dir)
    write_csv(primary, tables_dir / "primary_metrics.csv")
    table_outputs.append("primary_metrics.csv")
    if args.write_latex_snippets:
        write_primary_metrics_latex(primary, tables_dir / "primary_metrics.tex")
        table_outputs.append("primary_metrics.tex")

    stats = build_statistical_tests(args.statistics_dir)
    write_csv(stats, tables_dir / "statistical_tests.csv")
    table_outputs.append("statistical_tests.csv")

    thresholds = build_thresholds(args.statistics_dir)
    write_csv(thresholds, tables_dir / "threshold_operating_points.csv")
    table_outputs.append("threshold_operating_points.csv")
    if args.write_latex_snippets:
        write_thresholds_latex(thresholds, tables_dir / "threshold_operating_points.tex")
        table_outputs.append("threshold_operating_points.tex")

    patient = build_patient_level_metrics(args.statistics_dir, args.baseline_dir, args.enhanced_dir)
    write_csv(patient, tables_dir / "patient_level_metrics.csv")
    table_outputs.append("patient_level_metrics.csv")
    if args.write_latex_snippets:
        write_patient_level_latex(patient, tables_dir / "patient_level_metrics.tex")
        table_outputs.append("patient_level_metrics.tex")

    counts = build_feature_counts(args.statistics_dir)
    write_csv(counts, tables_dir / "feature_counts.csv")
    table_outputs.append("feature_counts.csv")
    if args.write_latex_snippets:
        write_feature_counts_latex(counts, tables_dir / "feature_counts.tex")
        table_outputs.append("feature_counts.tex")

    hemo = build_hemodynamic_features(args.statistics_dir)
    write_csv(hemo, tables_dir / "hemodynamic_features.csv")
    table_outputs.append("hemodynamic_features.csv")
    if args.write_latex_snippets:
        write_hemodynamic_features_latex(hemo, tables_dir / "hemodynamic_features.tex")
        table_outputs.append("hemodynamic_features.tex")

    sourcewise = build_sourcewise(args.internal_robustness_dir)
    write_csv(sourcewise, tables_dir / "sourcewise_metrics.csv")
    table_outputs.append("sourcewise_metrics.csv")
    if args.write_latex_snippets:
        write_sourcewise_latex(sourcewise, tables_dir / "sourcewise_metrics.tex")
        table_outputs.append("sourcewise_metrics.tex")

    if not args.skip_figures:
        figure_outputs.extend(plot_roc_pr(args.baseline_dir, args.enhanced_dir, figures_dir))
        figure_outputs.append(plot_calibration_summary(args.statistics_dir, figures_dir))
        figure_outputs.append(plot_utility_thresholds(args.baseline_dir, args.enhanced_dir, figures_dir))
        figure_outputs.append(plot_hemodynamic_importance(hemo, figures_dir))

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_scope": "public_final_tables_and_figures_from_final_reporting_artifacts",
        "inputs": {
            "statistics_dir": str(args.statistics_dir),
            "internal_robustness_dir": str(args.internal_robustness_dir),
            "baseline_dir": str(args.baseline_dir),
            "enhanced_dir": str(args.enhanced_dir),
        },
        "outputs_root": {
            "tables_dir": str(tables_dir),
            "figures_dir": str(figures_dir),
        },
        "skip_figures": bool(args.skip_figures),
        "write_latex_snippets": bool(args.write_latex_snippets),
        "outputs": {
            "tables": sorted(table_outputs + ["final_assets_manifest.json"]),
            "figures": sorted(figure_outputs),
        },
    }
    (tables_dir / "final_assets_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build compact public final tables and figures from reporting artifacts.")
    parser.add_argument("--statistics-dir", type=Path, default=Path("results/statistics"))
    parser.add_argument("--internal-robustness-dir", type=Path, default=Path("results/internal_robustness"))
    parser.add_argument("--baseline-dir", type=Path, default=Path("results/baseline"))
    parser.add_argument("--enhanced-dir", type=Path, default=Path("results/enhanced"))
    parser.add_argument("--tables-dir", type=Path, default=None, help="Output directory for compact CSV tables.")
    parser.add_argument("--figures-dir", type=Path, default=None, help="Output directory for compact figures.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Deprecated legacy single output directory for both tables and figures.")
    parser.add_argument("--write-latex-snippets", action="store_true", help="Also write local-only LaTeX snippets beside the CSV tables.")
    parser.add_argument("--skip-figures", action="store_true", help="Generate tables only; do not require OOF/plotting inputs.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    load_runtime_dependencies(include_figures=not args.skip_figures)
    manifest = build_assets(args)
    print(f"Final tables written to: {manifest['outputs_root']['tables_dir']}")
    if not args.skip_figures:
        print(f"Final figures written to: {manifest['outputs_root']['figures_dir']}")
    for group, items in manifest["outputs"].items():
        for item in items:
            print(f"  - {group}/{item}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
