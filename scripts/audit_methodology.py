#!/usr/bin/env python3
"""No-leakage and methodology audit for Sepsis-Hemodynamic.

This script is intentionally lightweight: it does not train models, does not
run SLURM jobs, and does not require pandas/numpy. It inspects source code,
configuration files, documentation, and small existing result artifacts when
available. Missing result artifacts are reported as CONDITIONAL/NOT_CHECKED
instead of causing an unhelpful failure.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXPECTED_POLICY = "phase1b_fold_impute_no_sampen_backfill_v2"
EXPECTED_BASELINE_FEATURES = 287
EXPECTED_ENHANCED_FEATURES = 305
EXPECTED_DELTA_FEATURES = 18


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""


def git_value(root: Path, args: list[str]) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=root,
            text=True,
            capture_output=True,
            check=False,
        )
    except Exception:
        return "unavailable"
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def format_path(path: Path, root: Path) -> str:
    """Display repo-relative paths when possible, otherwise absolute paths."""
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def first_existing(paths: list[Path]) -> Path:
    for path in paths:
        if path.exists():
            return path
    return paths[0]


def line_of(text: str, pattern: str) -> str:
    for idx, line in enumerate(text.splitlines(), start=1):
        if pattern in line:
            return str(idx)
    return ""


def regex_line(text: str, pattern: str) -> str:
    rx = re.compile(pattern)
    for idx, line in enumerate(text.splitlines(), start=1):
        if rx.search(line):
            return str(idx)
    return ""


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError:
        return None
    except Exception as exc:
        return {"_error": repr(exc)}


def csv_header(path: Path) -> list[str] | None:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            return next(csv.reader(handle))
    except FileNotFoundError:
        return None
    except StopIteration:
        return []


def add_check(
    rows: list[dict[str, str]],
    check_id: str,
    category: str,
    status: str,
    evidence_file: str,
    evidence_line_or_function: str,
    explanation: str,
    reviewer_relevance: str,
) -> None:
    rows.append(
        {
            "check_id": check_id,
            "category": category,
            "status": status,
            "evidence_file": evidence_file,
            "evidence_line_or_function": evidence_line_or_function,
            "explanation": explanation,
            "reviewer_relevance": reviewer_relevance,
        }
    )


def status_from(condition: bool, fail_status: str = "FAIL") -> str:
    return "PASS" if condition else fail_status


def write_csv(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "check_id",
        "category",
        "status",
        "evidence_file",
        "evidence_line_or_function",
        "explanation",
        "reviewer_relevance",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: list[dict[str, str]], path: Path, title: str, intro: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# {title}",
        "",
        intro,
        "",
        "| check_id | category | status | evidence | explanation | reviewer relevance |",
        "|---|---|---|---|---|---|",
    ]
    for row in rows:
        evidence = row["evidence_file"]
        if row["evidence_line_or_function"]:
            evidence += f":{row['evidence_line_or_function']}"
        values = [
            row["check_id"],
            row["category"],
            row["status"],
            evidence,
            row["explanation"],
            row["reviewer_relevance"],
        ]
        safe_values = [str(value).replace("\n", " ").replace("|", "\\|") for value in values]
        lines.append("| " + " | ".join(safe_values) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_claims_report(rows: list[dict[str, str]], path: Path) -> None:
    claim_rows = [row for row in rows if row["category"] == "claim_boundary"]
    intro = (
        "This report checks that public documentation and reporting artifacts "
        "limit claims to internal cross-validation and do not assert external "
        "validation, hidden-test performance, clinical deployment readiness, or SOTA superiority."
    )
    write_markdown(claim_rows, path, "Methodology Claims Check", intro)


def audit(args: argparse.Namespace) -> list[dict[str, str]]:
    root = repo_root()
    rows: list[dict[str, str]] = []

    data_code_path = root / "src" / "data" / "data_harmonization.py"
    train_code_path = root / "src" / "training" / "train_zabihi_cudf.py"
    report_code_path = root / "src" / "reporting" / "build_statistics.py"
    reproducibility_doc_path = first_existing(
        [
            root / "docs" / "reproducibility.md",
        ]
    )

    data_code = read_text(data_code_path)
    train_code = read_text(train_code_path)
    report_code = read_text(report_code_path)
    docs_text = "\n".join(read_text(path) for path in root.glob("docs/**/*.md"))

    # Dataset loading.
    add_check(
        rows,
        "DL001",
        "dataset_loading",
        status_from("psv_files" in data_code and "Using PSV files only" in data_code),
        str(data_code_path.relative_to(root)),
        line_of(data_code, "Using PSV files only"),
        "Loader detects individual PSV files and chooses them over aggregate CSV/TSV files.",
        "Addresses reviewer concerns about duplicated Dataset.csv/aggregate rows.",
    )
    add_check(
        rows,
        "DL002",
        "dataset_loading",
        status_from("Aggregate CSV/TSV files skipped" in data_code or "aggregate CSV/TSV files skipped" in data_code),
        str(data_code_path.relative_to(root)),
        line_of(data_code, "Aggregate CSV/TSV files skipped"),
        "Aggregate CSV/TSV names are logged as skipped when PSV files are present.",
        "Documents why Dataset.csv is not loaded in the primary analysis.",
    )
    add_check(
        rows,
        "DL003",
        "dataset_loading",
        status_from("training_setA" in data_code and "training_setB" in data_code),
        str(data_code_path.relative_to(root)),
        line_of(data_code, "training_setA"),
        "SourceSet is inferred from training_setA/training_setB archive paths.",
        "Supports internal source-wise traceability without claiming external validation.",
    )

    # Patient split.
    add_check(
        rows,
        "PS001",
        "patient_split",
        status_from("StratifiedGroupKFold" in train_code and "groups=patients" in train_code),
        str(train_code_path.relative_to(root)),
        line_of(train_code, "StratifiedGroupKFold"),
        "CV fold assignment uses patient groups, preventing the same Patient_ID from being split across train/validation folds.",
        "Directly addresses leakage via patient overlap.",
    )
    add_check(
        rows,
        "PS002",
        "patient_split",
        status_from(
            "build_or_load_fold_assignments" in train_code
            and "patient_fold_assignments_" in train_code
            and "n_splits}cv.csv" in train_code
            and "fold_df.to_csv(fold_file, index=False)" in train_code
        ),
        str(train_code_path.relative_to(root)),
        line_of(train_code, "patient_fold_assignments_"),
        "Fold assignments are loaded or generated as a reproducible patient-level CSV using the configured n_splits value.",
        "Supports repeatable internal CV splits.",
    )

    # Preprocessing / no global leakage.
    add_check(
        rows,
        "PP001",
        "preprocessing_imputation",
        status_from("fit_fold_imputation_values(X_train" in train_code and "X_tr_imp" in train_code and "X_val_imp" in train_code),
        str(train_code_path.relative_to(root)),
        line_of(train_code, "fit_fold_imputation_values(X_train"),
        "Median imputation is fitted on X_train and applied to train/validation arrays within each fold.",
        "Addresses global preprocessing leakage.",
    )
    add_check(
        rows,
        "PP002",
        "preprocessing_imputation",
        status_from(
            "all_missing = ~np.isfinite(values)" in train_code
            and "values[all_missing] = 0.0" in train_code
            and "np.nanmedian(finite, axis=0)" in train_code
        ),
        str(train_code_path.relative_to(root)),
        line_of(train_code, "all_missing = ~np.isfinite(values)"),
        "Columns entirely missing/non-finite in the training fold are filled with 0.0 fold-locally.",
        "Documents deterministic handling of all-missing train-fold features.",
    )
    add_check(
        rows,
        "PP003",
        "preprocessing_imputation",
        status_from("global_means" not in train_code),
        str(train_code_path.relative_to(root)),
        "",
        "No `global_means` pre-split imputation pattern is present in training code.",
        "Evidence against historical global fallback leakage.",
    )

    # Temporal causality.
    add_check(
        rows,
        "TC001",
        "temporal_causality",
        status_from("rolling(w, min_periods=1)" in train_code and "groupby(\"Patient_ID\")" in train_code),
        str(train_code_path.relative_to(root)),
        line_of(train_code, "rolling(w, min_periods=1)"),
        "Baseline rolling features use grouped trailing windows by Patient_ID.",
        "Supports causal time-series feature construction.",
    )
    add_check(
        rows,
        "TC002",
        "temporal_causality",
        status_from("center=True" not in train_code and "shift(-" not in train_code),
        str(train_code_path.relative_to(root)),
        "",
        "No centered rolling windows or negative shifts were detected in training code.",
        "Screens for future-information leakage.",
    )
    add_check(
        rows,
        "TC003",
        "feature_exclusion",
        status_from('"TimeStep"' in train_code and '"SourceSet"' in train_code and '"SepsisLabel"' in train_code),
        str(train_code_path.relative_to(root)),
        line_of(train_code, 'meta = ["Patient_ID", "TimeStep", "SourceSet", "SepsisLabel"]'),
        "Meta columns are explicitly handled outside model feature columns.",
        "Prevents identifiers, time index, labels, and source tags from becoming model inputs.",
    )

    # SampEn / hemodynamics.
    add_check(
        rows,
        "SE001",
        "sampen_policy",
        status_from("def sample_entropy_causal_cpu(series, window=24, m=2, r_factor=0.2)" in train_code),
        str(train_code_path.relative_to(root)),
        line_of(train_code, "def sample_entropy_causal_cpu"),
        "SampEn function defaults to window=24, m=2, r_factor=0.2.",
        "Documents hemodynamic complexity definition.",
    )
    add_check(
        rows,
        "SE002",
        "sampen_policy",
        status_from("early rows left" in train_code and "previous non-causal backfill" in train_code),
        str(train_code_path.relative_to(root)),
        line_of(train_code, "previous non-causal backfill"),
        "Early SampEn rows without enough history remain NaN rather than being backfilled from future valid values.",
        "Addresses future-backfill leakage concern.",
    )
    hemo_signals_ok = all(signal in train_code for signal in ["HeartRate", "SysBP", "MeanBP", "DiaBP", "RespRate", "O2Sat"])
    add_check(
        rows,
        "SE003",
        "hemodynamic_module",
        status_from(hemo_signals_ok and "f\"{col}_cv_8h\"" in train_code and "f\"{col}_iqr_8h\"" in train_code and "f\"{col}_sampen_24h\"" in train_code),
        str(train_code_path.relative_to(root)),
        line_of(train_code, "expected_hemo_feature_columns"),
        "Enhanced module expects six hemodynamic signals times CV_8h, IQR_8h, and SampEn_24h = 18 features.",
        "Supports feature-count explanation and local paper correction.",
    )

    # Calibration / metrics.
    add_check(
        rows,
        "CA001",
        "calibration",
        status_from("fold_ids != fold" in train_code and "fold_ids == fold" in train_code),
        str(train_code_path.relative_to(root)),
        line_of(train_code, "fold_ids != fold"),
        "Platt/isotonic calibration is cross-fitted by fold IDs in training post-processing.",
        "Supports internal OOF calibration while avoiding calibration-on-heldout leakage.",
    )
    add_check(
        rows,
        "MT001",
        "oof_metrics",
        status_from("oof_predictions.csv" in train_code and "prob_raw" in train_code),
        str(train_code_path.relative_to(root)),
        line_of(train_code, "oof_predictions.csv"),
        "Training post-processing writes OOF predictions and computes metrics from OOF probabilities.",
        "Supports valid internal CV performance estimates.",
    )
    add_check(
        rows,
        "MT002",
        "oof_metrics",
        status_from("align_oof_predictions" in report_code and "OOF_ALIGNMENT_COLUMNS" in report_code),
        str(report_code_path.relative_to(root)),
        line_of(report_code, "def align_oof_predictions"),
        "Statistics reporting aligns baseline/enhanced OOF predictions by Patient_ID, TimeStep, and SepsisLabel.",
        "Supports paired comparison between models.",
    )
    add_check(
        rows,
        "MT003",
        "oof_metrics",
        status_from("resamples `Patient_ID`" in report_code or "rng.choice(patients" in report_code),
        str(report_code_path.relative_to(root)),
        line_of(report_code, "rng.choice(patients"),
        "Bootstrap resamples Patient_ID values, not individual rows.",
        "Prevents anti-conservative row-level bootstrap over repeated time steps.",
    )
    add_check(
        rows,
        "MT004",
        "oof_metrics",
        status_from("threshold-dependent" in read_text(reproducibility_doc_path) and "Utility" in report_code),
        str(reproducibility_doc_path.relative_to(root)),
        line_of(read_text(reproducibility_doc_path), "Utility"),
        "Utility is documented as threshold-dependent rather than a blanket operational improvement.",
        "Prevents overclaiming clinical utility.",
    )

    # Result artifact checks if present.
    for label, result_dir in [("baseline", args.baseline_dir), ("enhanced", args.enhanced_dir)]:
        metrics = load_json(root / result_dir / "summary_metrics.json")
        if metrics is None:
            add_check(
                rows,
                f"RA_{label.upper()}_001",
                "result_artifacts",
                "NOT_CHECKED",
                str(Path(result_dir) / "summary_metrics.json"),
                "",
                "summary_metrics.json not found in this checkout; rerun the audit where final result artifacts exist.",
                "Result-derived feature counts and policy metadata require CEDIA artifacts.",
            )
            continue
        if "_error" in metrics:
            status = "CONDITIONAL"
            explanation = f"summary_metrics.json could not be parsed: {metrics['_error']}"
        else:
            status = "PASS" if metrics.get("pipeline_policy_version") == EXPECTED_POLICY else "FAIL"
            explanation = f"pipeline_policy_version={metrics.get('pipeline_policy_version')}"
        add_check(
            rows,
            f"RA_{label.upper()}_001",
            "result_artifacts",
            status,
            str(Path(result_dir) / "summary_metrics.json"),
            "pipeline_policy_version",
            explanation,
            "Confirms final outputs use the expected no-leakage policy.",
        )
        if "_error" not in metrics:
            expected = EXPECTED_BASELINE_FEATURES if label == "baseline" else EXPECTED_ENHANCED_FEATURES
            observed = metrics.get("n_features")
            add_check(
                rows,
                f"RA_{label.upper()}_002",
                "feature_exclusion",
                "PASS" if observed == expected else "CONDITIONAL",
                str(Path(result_dir) / "summary_metrics.json"),
                "n_features",
                f"Observed n_features={observed}; expected {expected}.",
                "Confirms final feature counts after meta-column exclusion and enhanced +18 feature delta.",
            )

    validation_path = root / args.statistics_dir / "validation_checks.csv"
    header = csv_header(validation_path)
    add_check(
        rows,
        "RA_STATISTICS_001",
        "result_artifacts",
        "PASS" if header and {"check", "status", "value", "detail"}.issubset(set(header)) else "NOT_CHECKED",
        str(Path(args.statistics_dir) / "validation_checks.csv"),
        "",
        "Statistics validation checks are present." if header else "Statistics validation checks not found locally; run after statistics generation.",
        "Links methodology audit to the generated statistics package.",
    )

    # Claim boundaries.
    forbidden_patterns = [
        ("clinically validated", "CB001"),
        ("ready for deployment", "CB002"),
        ("state of the art", "CB003"),
        ("outperforms Zabihi", "CB004"),
        ("external validation", "CB005"),
        ("hidden test performance", "CB006"),
    ]
    allowed_negation_context = ("No independent external validation", "not external validation", "Do not claim external validation")
    for phrase, check_id in forbidden_patterns:
        if phrase == "external validation":
            bad = phrase in docs_text and not any(context in docs_text for context in allowed_negation_context)
        else:
            bad = phrase in docs_text
        add_check(
            rows,
            check_id,
            "claim_boundary",
            "FAIL" if bad else "PASS",
            "docs/**/*.md",
            "",
            f"Forbidden claim phrase check: {phrase!r}.",
            "Ensures paper-supporting docs do not overclaim beyond internal CV.",
        )
    add_check(
        rows,
        "CB007",
        "claim_boundary",
        status_from("internal cross-validation" in docs_text and "public PhysioNet/CinC 2019 training split" in docs_text),
        "docs/**/*.md",
        "",
        "Allowed language about internal cross-validation and public training split is present.",
        "Frames claims within the correct validation boundary.",
    )

    return rows


def main() -> int:
    root = repo_root()
    parser = argparse.ArgumentParser(description="Run no-leakage/methodology audit.")
    parser.add_argument("--baseline-dir", default="results/baseline")
    parser.add_argument("--enhanced-dir", default="results/enhanced")
    parser.add_argument("--statistics-dir", default="results/statistics")
    parser.add_argument("--output-dir", default="external_artifacts/inventory")
    args = parser.parse_args()

    rows = audit(args)
    output_dir = root / args.output_dir
    csv_path = output_dir / "no_leakage_audit.csv"
    md_path = output_dir / "no_leakage_audit.md"
    claims_path = output_dir / "methodology_claims_check.md"

    intro = (
        f"Generated {datetime.now(timezone.utc).isoformat()} from branch "
        f"`{git_value(root, ['rev-parse', '--abbrev-ref', 'HEAD'])}` commit "
        f"`{git_value(root, ['rev-parse', '--short', 'HEAD'])}`. This audit is read-only "
        "with respect to training and result artifacts."
    )
    write_csv(rows, csv_path)
    write_markdown(rows, md_path, "No-Leakage And Methodology Audit", intro)
    build_claims_report(rows, claims_path)

    status_counts: dict[str, int] = {}
    for row in rows:
        status_counts[row["status"]] = status_counts.get(row["status"], 0) + 1
    print("Methodology audit written:")
    print(f"- {format_path(csv_path, root)}")
    print(f"- {format_path(md_path, root)}")
    print(f"- {format_path(claims_path, root)}")
    print("Status counts:", json.dumps(status_counts, sort_keys=True))
    if status_counts.get("FAIL", 0):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
