"""Canonical scientific pipeline for the public PhysioNet/CinC 2019 experiment.

The module deliberately has one numerical backend: NumPy/pandas for features
and scikit-learn/XGBoost for models.  Challenge Utility is delegated to the
unmodified official PhysioNet scorer vendored in ``vendor/physionet2019``.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
import warnings
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (
    auc,
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from vendor.physionet2019 import evaluate_sepsis_score as official_utility


PIPELINE_VERSION = "scientific-pipeline-v2"
SEED = 20260906
OFFICIAL_UTILITY_SHA256 = "26b8b26267ed32e8b7a7a27e45201cfc8c6640e717ba4cdc1f452b32f12b99e5"
DATA_POLICY = {
    "archive_sha256": "1a0eb8040c76fdab84ee6c7dd6afdab4ad457a33d363cb7e4e200af713345897",
    "psv_file_count": 40336,
    "patient_count": 40336,
    "row_count": 1552210,
    "source_patient_counts": {"A": 20336, "B": 20000},
}
CHALLENGE_COLUMNS = (
    "HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2",
    "BaseExcess", "HCO3", "FiO2", "pH", "PaCO2", "SaO2", "AST", "BUN",
    "Alkalinephos", "Calcium", "Chloride", "Creatinine", "Bilirubin_direct",
    "Glucose", "Lactate", "Magnesium", "Phosphate", "Potassium",
    "Bilirubin_total", "TroponinI", "Hct", "Hgb", "PTT", "WBC",
    "Fibrinogen", "Platelets", "Age", "Gender", "Unit1", "Unit2",
    "HospAdmTime", "ICULOS", "SepsisLabel",
)
PREDICTOR_COLUMNS = CHALLENGE_COLUMNS[:-1]
STATIC_COLUMNS = ("Age", "Gender", "Unit1", "Unit2", "HospAdmTime")
DYNAMIC_COLUMNS = tuple(c for c in PREDICTOR_COLUMNS if c not in STATIC_COLUMNS + ("ICULOS",))
HEMODYNAMIC_COLUMNS = ("HR", "O2Sat", "SBP", "MAP", "DBP", "Resp")
HEADER_ALIASES = {"HCT": "Hct", "Hematocrit": "Hct"}
FEATURE_POLICY = {
    "last_observation_max_age_hours": 24,
    "rolling_windows_hours": {"cv_iqr": 8, "sampen": 24},
    "rolling_min_observations": 2,
    "sampen": {"window_hours": 24, "m": 2, "r_factor": 0.2, "min_observations": 4},
    "early_warning": {"start_hours_before_onset": 12, "end_hours_before_onset": 1, "refractory_hours": 6},
    "calibration": "pre-specified_platt_nested_inner_oof",
    "calibration_fit_weighting": "equal total weight per patient",
    "ece_equal_width_bins": 10,
    "calibration_patient_cluster_bootstrap_repeats": 300,
    "dca_patient_cluster_bootstrap_repeats": 300,
    "paired_inference_repeats": 200,
    "split_stability_seeds": (SEED, SEED + 101, SEED + 202),
    "threshold_grid": tuple(round(x, 2) for x in np.arange(0.05, 1.00, 0.05)),
}


class PipelineError(RuntimeError):
    """An input, provenance, or scientific validity gate failed."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        frame.to_csv(handle, index=False)
        temporary = Path(handle.name)
    temporary.replace(path)


def git_value(root: Path, *args: str) -> str:
    completed = subprocess.run(["git", *args], cwd=root, text=True, capture_output=True, check=False)
    return completed.stdout.strip() if completed.returncode == 0 else "unavailable"


def dependency_versions() -> dict[str, str]:
    import sklearn
    import xgboost

    versions = {
        "python": sys.version.replace("\n", " "),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "sklearn": sklearn.__version__,
        "xgboost": xgboost.__version__,
    }
    for optional in ("cupy", "cudf"):
        try:
            module = __import__(optional)
            versions[optional] = getattr(module, "__version__", "installed")
        except ImportError:
            versions[optional] = "not-installed"
    return versions


def gpu_runtime() -> dict[str, Any]:
    """Return usable GPU state; importing CuPy alone is never evidence of a GPU."""
    state: dict[str, Any] = {"available": False, "reason": "CuPy unavailable", "device_count": 0, "n_gpus_used": 0, "device_backend": "cpu", "gpu_model": None, "cuda_runtime_version": None, "cuda_driver_version": None}
    try:
        import cupy as cp
    except Exception as exc:
        state["reason"] = f"CuPy import failed: {type(exc).__name__}"
        return state
    try:
        count = int(cp.cuda.runtime.getDeviceCount())
        state.update({"device_count": count, "cuda_runtime_version": int(cp.cuda.runtime.runtimeGetVersion()), "cuda_driver_version": int(cp.cuda.runtime.driverGetVersion())})
        if count < 1:
            state["reason"] = "CUDA runtime reported zero devices"
            return state
        test = cp.asarray([1.0], dtype=cp.float32)
        if float((test + 1).sum().get()) != 2.0:
            state["reason"] = "CUDA minimum operation returned an unexpected value"
            return state
        properties = cp.cuda.runtime.getDeviceProperties(0)
        name = properties.get("name", b"unknown")
        state.update({"available": True, "reason": "validated", "n_gpus_used": 1, "device_backend": "cuda", "gpu_model": name.decode() if isinstance(name, bytes) else str(name)})
        return state
    except Exception as exc:
        state["reason"] = f"CUDA runtime validation failed: {type(exc).__name__}: {exc}"
        return state


def runtime_manifest(root: Path, run_id: str, command: list[str], archive: Path) -> dict[str, Any]:
    utility_path = root / "vendor" / "physionet2019" / "evaluate_sepsis_score.py"
    utility_hash = sha256_file(utility_path)
    if utility_hash != OFFICIAL_UTILITY_SHA256:
        raise PipelineError("The vendored official Utility scorer hash does not match its pinned source.")
    archive_hash = sha256_file(archive)
    if archive_hash != DATA_POLICY["archive_sha256"]:
        raise PipelineError("Raw archive hash does not match the pinned complete public A/B input")
    gpu = gpu_runtime()
    return {
        "timestamp_utc": utc_now(),
        "hostname": platform.node(),
        "run_id": run_id,
        "pipeline_version": PIPELINE_VERSION,
        "git_commit": os.environ.get("SOURCE_GIT_COMMIT") or git_value(root, "rev-parse", "HEAD"),
        "git_dirty": os.environ.get("SOURCE_GIT_DIRTY", "false").lower() == "true",
        "command_line": command,
        "run_sh_sha256": sha256_file(root / "run.sh") if (root / "run.sh").is_file() else "not-created-yet",
        "data_archive_path": str(archive.resolve()),
        "data_archive_sha256": archive_hash,
        "data_policy": DATA_POLICY,
        "data_policy_hash": stable_hash(DATA_POLICY),
        "schema_version": "PhysioNet-CinC-2019-v1.0.0-40-predictors",
        "feature_policy": FEATURE_POLICY,
        "feature_policy_hash": stable_hash(FEATURE_POLICY),
        "seed": SEED,
        "pythonhashseed": os.environ.get("PYTHONHASHSEED", "unset"),
        "dependencies": dependency_versions(),
        "execution_environment": {
            "container_path": os.environ.get("SOURCE_CONTAINER_PATH", "not-scheduled"),
            "container_sha256": os.environ.get("SOURCE_CONTAINER_SHA256", "not-scheduled"),
            "container_runtime": os.environ.get("SOURCE_CONTAINER_RUNTIME", "not-scheduled"),
        },
        "gpu": gpu,
        "xgboost_backend": xgb_backend(gpu),
        "official_utility": {
            "source": "physionetchallenges/evaluation-2019@467c49b514542be7a4a0bafe40fa2c3b064dda2e",
            "sha256": utility_hash,
        },
    }


def runtime_resume_context(manifest: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "git_commit", "run_sh_sha256", "pipeline_version", "data_archive_sha256", "data_policy_hash",
        "schema_version", "feature_policy_hash", "seed", "pythonhashseed", "dependencies",
        "execution_environment", "gpu", "xgboost_backend", "official_utility",
    )
    return {key: manifest.get(key) for key in keys}


def require_python_hash_seed() -> None:
    if os.environ.get("PYTHONHASHSEED") != str(SEED):
        raise PipelineError(
            f"PYTHONHASHSEED must be {SEED} before interpreter startup; got "
            f"{os.environ.get('PYTHONHASHSEED', 'unset')!r}. Use run.sh."
        )


def canonical_headers(headers: Iterable[str], member_name: str) -> list[str]:
    normalized = [HEADER_ALIASES.get(str(header).strip(), str(header).strip()) for header in headers]
    duplicates = sorted(header for header, count in Counter(normalized).items() if count > 1)
    unknown = sorted(set(normalized).difference(CHALLENGE_COLUMNS))
    missing = sorted(set(CHALLENGE_COLUMNS).difference(normalized))
    if duplicates or unknown or missing:
        raise PipelineError(
            f"{member_name}: invalid official schema; duplicates={duplicates}, "
            f"unknown={unknown}, missing={missing}. Only explicit HCT/Hematocrit-to-Hct aliases are accepted."
        )
    return normalized


def numeric_columns(frame: pd.DataFrame, columns: Iterable[str], context: str) -> pd.DataFrame:
    columns = list(columns)
    original = frame.loc[:, columns]
    numeric = original.apply(pd.to_numeric, errors="coerce")
    invalid = original.notna() & ~np.isfinite(numeric)
    if invalid.any().any():
        names = invalid.columns[invalid.any()].tolist()
        raise PipelineError(f"{context}: non-numeric or infinite values in {names}")
    return numeric


def source_and_patient(member_name: str) -> tuple[str, str]:
    parts = Path(member_name).parts
    expected_directories = {"training_setA": "training", "training_setB": "training_setB"}
    if len(parts) != 3 or parts[0] not in expected_directories or parts[1] != expected_directories[parts[0]]:
        raise PipelineError(f"Unexpected PSV path outside validated official training sets: {member_name}")
    stem = Path(parts[2]).stem
    if not stem.startswith("p") or not stem[1:].isdigit():
        raise PipelineError(f"Unexpected official patient filename: {member_name}")
    source = "A" if parts[0].endswith("A") else "B"
    return source, f"{source}:{stem}"


def validate_patient_frame(frame: pd.DataFrame, member_name: str) -> pd.DataFrame:
    if frame.empty:
        raise PipelineError(f"{member_name}: empty patient file")
    frame = frame.copy()
    frame[list(CHALLENGE_COLUMNS)] = numeric_columns(frame, CHALLENGE_COLUMNS, member_name)
    if not np.isfinite(frame["ICULOS"]).all() or (frame["ICULOS"].diff().iloc[1:] <= 0).any():
        raise PipelineError(f"{member_name}: ICULOS must be finite and strictly increasing within patient")
    labels = frame["SepsisLabel"]
    if not labels.isin([0, 1]).all():
        raise PipelineError(f"{member_name}: SepsisLabel must be binary")
    first_positive = np.flatnonzero(labels.to_numpy(dtype=int))
    if len(first_positive) and not (labels.iloc[first_positive[0]:] == 1).all():
        raise PipelineError(f"{member_name}: Challenge shifted labels must be persistent after first positive")
    for column in STATIC_COLUMNS:
        observed = frame[column].dropna().unique()
        if len(observed) > 1:
            raise PipelineError(f"{member_name}: static predictor {column} changes within patient")
    return frame


def reconstruct_true_onset(labels: Iterable[int], times: Iterable[float]) -> tuple[float, str]:
    label_array = np.asarray(list(labels), dtype=int)
    time_array = np.asarray(list(times), dtype=float)
    first_positive = np.flatnonzero(label_array)
    if not len(first_positive):
        return math.nan, "nonseptic"
    if first_positive[0] == 0:
        return math.nan, "septic_onset_left_censored"
    return float(time_array[first_positive[0]] + 6), "exact_from_shift_transition"


def archive_inventory(archive: Path) -> tuple[list[zipfile.ZipInfo], dict[str, Any]]:
    with zipfile.ZipFile(archive) as zf:
        members = sorted((info for info in zf.infolist() if not info.is_dir()), key=lambda info: info.filename)
    psv = [info for info in members if info.filename.lower().endswith(".psv")]
    if not psv:
        raise PipelineError("Archive contains no PSV patient files; CSV/TSV fallback is prohibited.")
    inventory = [{"name": info.filename, "size": info.file_size, "crc": info.CRC} for info in members]
    return psv, {"member_count": len(members), "psv_file_count": len(psv), "inventory_hash": stable_hash(inventory)}


def validate_cohort_identity(frame: pd.DataFrame, inventory: dict[str, Any]) -> dict[str, int]:
    source_counts = {key: int(value) for key, value in frame.groupby("SourceSet")["Patient_ID"].nunique().to_dict().items()}
    observed = {
        "psv_file_count": int(inventory["psv_file_count"]),
        "patient_count": int(frame["Patient_ID"].nunique()),
        "row_count": int(len(frame)),
        "source_patient_counts": source_counts,
    }
    expected = {key: DATA_POLICY[key] for key in observed}
    if observed != expected:
        raise PipelineError(f"Validated cohort identity mismatch: observed={observed}, expected={expected}")
    return source_counts


def harmonize_archive(archive: Path, output: Path) -> dict[str, Any]:
    if not archive.is_file():
        raise PipelineError(f"Missing data archive: {archive}")
    psv_members, inventory = archive_inventory(archive)
    frames: list[pd.DataFrame] = []
    seen_patients: set[str] = set()
    with zipfile.ZipFile(archive) as zf:
        for info in psv_members:
            source, patient = source_and_patient(info.filename)
            if patient in seen_patients:
                raise PipelineError(f"Duplicate source-qualified patient id: {patient}")
            seen_patients.add(patient)
            with zf.open(info) as handle:
                frame = pd.read_csv(handle, sep="|", dtype=str)
            frame.columns = canonical_headers(frame.columns, info.filename)
            frame = frame.loc[:, list(CHALLENGE_COLUMNS)].copy()
            frame = validate_patient_frame(frame, info.filename)
            frame.insert(0, "Patient_ID", patient)
            frame.insert(1, "SourceSet", source)
            onset, onset_status = reconstruct_true_onset(frame["SepsisLabel"], frame["ICULOS"])
            frame["TrueSepsisOnset_ICULOS"] = onset
            frame["OnsetReconstructionStatus"] = onset_status
            frames.append(frame)
    harmonized = pd.concat(frames, ignore_index=True).sort_values(
        ["SourceSet", "Patient_ID", "ICULOS"], kind="mergesort"
    ).reset_index(drop=True)
    if harmonized.duplicated(["Patient_ID", "ICULOS"]).any():
        raise PipelineError("Harmonization produced duplicate Patient_ID/ICULOS rows")
    source_counts = validate_cohort_identity(harmonized, inventory)
    atomic_csv(harmonized, output)
    return {
        "stage": "harmonized",
        "created_at_utc": utc_now(),
        "artifact": str(output),
        "artifact_sha256": sha256_file(output),
        "archive_sha256": sha256_file(archive),
        "row_count": int(len(harmonized)),
        "patient_count": int(harmonized["Patient_ID"].nunique()),
        "source_patient_counts": source_counts,
        **inventory,
    }


def causal_last_observation(values: pd.Series, times: pd.Series, max_age_hours: int) -> pd.Series:
    latest_value = values.ffill()
    observed_at = times.where(values.notna()).ffill()
    return latest_value.where((times - observed_at) <= max_age_hours)


def sample_entropy(values: Iterable[float], m: int = 2, r_factor: float = 0.2) -> float:
    """Canonical SampEn on observed values, with explicit undefined/zero-match states."""
    series = np.asarray(list(values), dtype=float)
    series = series[np.isfinite(series)]
    if len(series) < m + 2:
        return math.nan
    tolerance = r_factor * float(np.std(series, ddof=0))
    # Both match counts use the same N-m starting positions. The terminal
    # m-template has no corresponding (m+1)-template and must not enter B.
    templates_m = np.lib.stride_tricks.sliding_window_view(series, m)[:-1]
    templates_m1 = np.lib.stride_tricks.sliding_window_view(series, m + 1)
    matches_m = np.max(np.abs(templates_m[:, None, :] - templates_m[None, :, :]), axis=2) <= tolerance
    matches_m1 = np.max(np.abs(templates_m1[:, None, :] - templates_m1[None, :, :]), axis=2) <= tolerance
    denominator = int(np.triu(matches_m, k=1).sum())
    numerator = int(np.triu(matches_m1, k=1).sum())
    if denominator == 0:
        return math.nan  # SampEn is undefined: no m-template matches exist.
    if numerator == 0:
        return math.inf  # Mathematically defined zero-match result: -log(0/B).
    return float(-math.log(numerator / denominator))


def shannon_entropy(values: Iterable[float], bins: int = 10) -> float:
    observed = np.asarray(list(values), dtype=float)
    observed = observed[np.isfinite(observed)]
    if len(observed) < 2 or np.all(observed == observed[0]):
        return 0.0 if len(observed) else math.nan
    counts, _ = np.histogram(observed, bins=bins)
    probabilities = counts[counts > 0] / counts.sum()
    return float(-(probabilities * np.log(probabilities)).sum())


def rolling_feature(series: pd.Series, hours: int, function: str) -> pd.Series:
    index = pd.to_timedelta(series.index.to_numpy(dtype=float), unit="h")
    observed = pd.Series(series.to_numpy(dtype=float), index=index)
    window = observed.rolling(f"{hours}h", closed="right", min_periods=FEATURE_POLICY["rolling_min_observations"])
    if function == "mean":
        out = window.mean()
    elif function == "std":
        out = window.std(ddof=1)
    elif function == "iqr":
        out = window.quantile(0.75) - window.quantile(0.25)
    elif function == "shannon":
        out = window.apply(shannon_entropy, raw=True)
    else:
        raise ValueError(function)
    return pd.Series(out.to_numpy(), index=series.index)


def causal_sampen(series: pd.Series, times: pd.Series) -> pd.Series:
    result = np.full(len(series), np.nan, dtype=float)
    values = series.to_numpy(dtype=float)
    hour = times.to_numpy(dtype=float)
    policy = FEATURE_POLICY["sampen"]
    for end in range(len(series)):
        start = np.searchsorted(hour, hour[end] - policy["window_hours"], side="right")
        result[end] = sample_entropy(values[start : end + 1], policy["m"], policy["r_factor"])
    return pd.Series(result, index=series.index)


def feature_patient(patient: pd.DataFrame, include_hemodynamics: bool) -> pd.DataFrame:
    patient = patient.sort_values("ICULOS", kind="mergesort").copy()
    times = patient["ICULOS"]
    identity = patient[["Patient_ID", "SourceSet", "ICULOS", "SepsisLabel", "TrueSepsisOnset_ICULOS", "OnsetReconstructionStatus"]].copy()
    engineered: dict[str, Any] = {column: patient[column].to_numpy() for column in STATIC_COLUMNS}
    engineered["Measurement_Count"] = patient.loc[:, DYNAMIC_COLUMNS].notna().sum(axis=1).to_numpy(dtype="int16")
    for column in DYNAMIC_COLUMNS:
        raw = pd.to_numeric(patient[column], errors="coerce")
        engineered[f"{column}_is_missing"] = raw.isna().to_numpy(dtype="int8")
        observed_at = times.where(raw.notna()).ffill()
        engineered[f"{column}_observation_age_hours"] = (times - observed_at).to_numpy()
        engineered[f"{column}_last_obs"] = causal_last_observation(
            raw, times, FEATURE_POLICY["last_observation_max_age_hours"]
        ).to_numpy()
    if include_hemodynamics:
        for column in HEMODYNAMIC_COLUMNS:
            raw = pd.to_numeric(patient[column], errors="coerce")
            raw.index = times.to_numpy(dtype=float)
            variability_hours = FEATURE_POLICY["rolling_windows_hours"]["cv_iqr"]
            mean_8h = rolling_feature(raw, variability_hours, "mean")
            std_8h = rolling_feature(raw, variability_hours, "std")
            engineered[f"{column}_cv_8h"] = (std_8h / mean_8h.abs()).replace([np.inf, -np.inf], np.nan).to_numpy()
            engineered[f"{column}_iqr_8h"] = rolling_feature(raw, variability_hours, "iqr").to_numpy()
            observed = pd.Series(raw.notna().to_numpy(dtype="int8"), index=pd.to_timedelta(times.to_numpy(dtype=float), unit="h"))
            engineered[f"{column}_sampen_effective_n_24h"] = observed.rolling("24h", closed="right").sum().to_numpy(dtype="int16")
            sampen = causal_sampen(raw, times)
            engineered[f"{column}_sampen_24h_zero_match"] = np.isposinf(sampen).astype("int8")
            engineered[f"{column}_sampen_24h"] = sampen.replace([np.inf, -np.inf], np.nan).to_numpy()
    return pd.concat([identity, pd.DataFrame(engineered, index=patient.index)], axis=1)


def build_features(harmonized: Path, output: Path) -> dict[str, Any]:
    frame = pd.read_csv(harmonized)
    required = {"Patient_ID", "SourceSet", *CHALLENGE_COLUMNS, "TrueSepsisOnset_ICULOS", "OnsetReconstructionStatus"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise PipelineError(f"Harmonized artifact is invalid; missing {missing}")
    frame[list(CHALLENGE_COLUMNS)] = numeric_columns(frame, CHALLENGE_COLUMNS, "harmonized artifact")
    frame[["TrueSepsisOnset_ICULOS"]] = numeric_columns(frame, ["TrueSepsisOnset_ICULOS"], "harmonized artifact")
    features = pd.concat(
        [feature_patient(group, include_hemodynamics=True) for _, group in frame.groupby("Patient_ID", sort=False)],
        ignore_index=True,
    )
    if len(features) != len(frame) or features.duplicated(["Patient_ID", "ICULOS"]).any():
        raise PipelineError("Feature construction changed row identity")
    feature_columns = model_features(features, "enhanced")
    atomic_csv(features, output)
    return {
        "stage": "features",
        "created_at_utc": utc_now(),
        "artifact": str(output),
        "artifact_sha256": sha256_file(output),
        "input_sha256": sha256_file(harmonized),
        "row_count": int(len(features)),
        "feature_columns": feature_columns,
        "feature_column_hash": stable_hash(feature_columns),
    }


def model_features(frame: pd.DataFrame, variant: str) -> list[str]:
    # ICULOS is an observed official predictor, not an identifier. SourceSet is
    # administrative provenance and is never a model input.
    if frame.columns.duplicated().any():
        raise PipelineError("Feature schema contains duplicate columns")
    excluded = {"Patient_ID", "SourceSet", "SepsisLabel", "TrueSepsisOnset_ICULOS", "OnsetReconstructionStatus", "Fold"}
    columns = [column for column in frame.columns if column not in excluded and not column.endswith("_sampen_effective_n_24h")]
    enhanced_only = [column for column in columns if column.endswith(("_cv_8h", "_iqr_8h", "_sampen_24h", "_sampen_24h_zero_match"))]
    if variant == "baseline":
        columns = [column for column in columns if column not in enhanced_only]
    elif variant != "enhanced":
        raise PipelineError(f"Unknown model variant: {variant}")
    if not columns or "Hct_last_obs" not in columns:
        raise PipelineError("Feature policy failed: canonical Hct is not a model feature")
    return columns


def write_folds(features: pd.DataFrame, output: Path, n_splits: int = 5, split_seed: int = SEED) -> dict[str, Any]:
    patient = features.groupby("Patient_ID", sort=True)["SepsisLabel"].max().astype(int).reset_index()
    if patient["SepsisLabel"].value_counts().min() < n_splits:
        raise PipelineError("Insufficient septic or non-septic patients for requested outer folds")
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=split_seed)
    folds = np.full(len(patient), -1, dtype=int)
    for fold, (_, held_out) in enumerate(splitter.split(patient, patient["SepsisLabel"], groups=patient["Patient_ID"])):
        folds[held_out] = fold
    patient["Fold"] = folds
    if (patient["Fold"] < 0).any() or patient.groupby("Patient_ID")["Fold"].nunique().max() != 1:
        raise PipelineError("Failed to create one isolated fold per patient")
    atomic_csv(patient, output)
    return {
        "stage": "folds",
        "created_at_utc": utc_now(),
        "artifact": str(output),
        "artifact_sha256": sha256_file(output),
        "patient_count": int(len(patient)),
        "fold_count": n_splits,
        "seed": split_seed,
        "patient_inventory_hash": stable_hash(patient[["Patient_ID", "SepsisLabel"]].to_dict("records")),
    }


def require_fold_context(features: pd.DataFrame, folds: pd.DataFrame) -> pd.DataFrame:
    required = {"Patient_ID", "SepsisLabel", "Fold"}
    if missing := required.difference(folds.columns):
        raise PipelineError(f"Fold provenance missing columns: {sorted(missing)}")
    feature_patients = set(features["Patient_ID"])
    fold_patients = set(folds["Patient_ID"])
    if folds["Patient_ID"].duplicated().any() or feature_patients != fold_patients:
        raise PipelineError("Fold artifact does not isolate every feature patient exactly once")
    expected_labels = features.groupby("Patient_ID", sort=True)["SepsisLabel"].max().astype(int)
    observed_labels = pd.to_numeric(folds.set_index("Patient_ID")["SepsisLabel"], errors="coerce").sort_index()
    if observed_labels.isna().any() or not observed_labels.astype(int).equals(expected_labels):
        raise PipelineError("Fold outcome provenance does not match the feature cohort")
    merged = features.merge(folds[["Patient_ID", "Fold"]], on="Patient_ID", how="left", validate="many_to_one")
    if merged["Fold"].isna().any():
        raise PipelineError("Feature rows lack fold provenance")
    if not (merged.groupby("Patient_ID")["Fold"].nunique() == 1).all():
        raise PipelineError("A patient spans multiple outer folds")
    return merged


def matrix(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    # XGBoost receives genuine missing values only; corrupt or infinite observed
    # values are never silently recoded as missingness.
    return numeric_columns(frame, columns, "model matrix").to_numpy(dtype=np.float32)


def xgb_backend(gpu: dict[str, Any]) -> dict[str, str]:
    import xgboost as xgb

    major = int(xgb.__version__.split(".", 1)[0])
    if not gpu["available"]:
        return {"tree_method": "hist"}
    return {"tree_method": "hist", "device": "cuda"} if major >= 2 else {"tree_method": "gpu_hist", "predictor": "gpu_predictor"}


def xgb_model(params: dict[str, Any], seed: int, gpu: dict[str, Any], n_estimators: int, early_stopping: bool = False):
    import xgboost as xgb

    kwargs: dict[str, Any] = {
        "n_estimators": int(n_estimators),
        "max_depth": params["max_depth"],
        "learning_rate": params["learning_rate"],
        "min_child_weight": params["min_child_weight"],
        "subsample": params["subsample"],
        "colsample_bytree": params["colsample_bytree"],
        "objective": "binary:logistic",
        "eval_metric": "logloss",  # XGBoost aucpr is not sklearn Average Precision.
        "random_state": int(seed),
        "n_jobs": 8,
        **xgb_backend(gpu),
    }
    if early_stopping:
        kwargs["early_stopping_rounds"] = 30
    return xgb.XGBClassifier(**kwargs)


def fit_xgb(model: Any, train: pd.DataFrame, columns: list[str], validation: pd.DataFrame | None = None) -> Any:
    kwargs: dict[str, Any] = {"sample_weight": equal_patient_weights(train), "verbose": False}
    if validation is not None:
        kwargs.update({
            "eval_set": [(matrix(validation, columns), validation["SepsisLabel"])],
            "sample_weight_eval_set": [equal_patient_weights(validation)],
        })
    return model.fit(matrix(train, columns), train["SepsisLabel"], **kwargs)


MODEL_CANDIDATES = (
    {"id": "depth3", "max_depth": 3, "learning_rate": 0.05, "min_child_weight": 1, "subsample": 0.8, "colsample_bytree": 0.8},
    {"id": "depth5", "max_depth": 5, "learning_rate": 0.05, "min_child_weight": 1, "subsample": 0.8, "colsample_bytree": 0.8},
)

OOF_OUTPUT_COLUMNS = [
    "Patient_ID", "SourceSet", "ICULOS", "Age", "SepsisLabel", "TrueSepsisOnset_ICULOS",
    "OnsetReconstructionStatus", "Fold", "prob_raw", "prob_platt", "nested_threshold",
]


def inner_patient_splits(patient: pd.DataFrame, n_splits: int = 3, seed_offset: int = 0, split_seed: int = SEED):
    if patient["SepsisLabel"].value_counts().min() < n_splits:
        raise PipelineError("Insufficient class count for nested inner grouped folds")
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=split_seed + seed_offset)
    yield from splitter.split(patient, patient["SepsisLabel"], groups=patient["Patient_ID"])


def patient_mask(frame: pd.DataFrame, patients: Iterable[str]) -> np.ndarray:
    return frame["Patient_ID"].isin(set(patients)).to_numpy()


def select_inner_model(train: pd.DataFrame, columns: list[str], gpu: dict[str, Any], outer_fold: int, split_seed: int = SEED) -> tuple[dict[str, Any], int, pd.DataFrame]:
    """Select only inside an outer training partition, then emit unbiased inner OOF raw scores."""
    patient = train.groupby("Patient_ID", sort=True)["SepsisLabel"].max().astype(int).reset_index()
    splits = list(inner_patient_splits(patient, seed_offset=outer_fold + 1, split_seed=split_seed))
    scores: dict[str, list[float]] = {candidate["id"]: [] for candidate in MODEL_CANDIDATES}
    rounds: dict[str, list[int]] = {candidate["id"]: [] for candidate in MODEL_CANDIDATES}
    for candidate_index, candidate in enumerate(MODEL_CANDIDATES):
        for inner_fold, (fit_idx, valid_idx) in enumerate(splits):
            fit_patients = patient.iloc[fit_idx]["Patient_ID"]
            valid_patients = patient.iloc[valid_idx]["Patient_ID"]
            fit = train.loc[patient_mask(train, fit_patients)]
            valid = train.loc[patient_mask(train, valid_patients)]
            model = xgb_model(candidate, split_seed + outer_fold * 100 + candidate_index * 10 + inner_fold, gpu, 600, early_stopping=True)
            fit_xgb(model, fit, columns, valid)
            score = average_precision_score(valid["SepsisLabel"], model.predict_proba(matrix(valid, columns))[:, 1])
            scores[candidate["id"]].append(float(score))
            rounds[candidate["id"]].append(int(getattr(model, "best_iteration", model.n_estimators - 1)) + 1)
    winner = max(MODEL_CANDIDATES, key=lambda candidate: (float(np.mean(scores[candidate["id"]])), -candidate["max_depth"]))
    selected_rounds = max(1, int(round(float(np.median(rounds[winner["id"]])))))
    # Refit inner models without validation/early stopping.  Their held-out
    # probabilities are suitable for calibrator and threshold fitting.
    inner_rows: list[pd.DataFrame] = []
    for inner_fold, (fit_idx, valid_idx) in enumerate(splits):
        fit_patients = patient.iloc[fit_idx]["Patient_ID"]
        valid_patients = patient.iloc[valid_idx]["Patient_ID"]
        fit = train.loc[patient_mask(train, fit_patients)]
        valid = train.loc[patient_mask(train, valid_patients)].copy()
        model = xgb_model(winner, split_seed + outer_fold * 1000 + inner_fold, gpu, selected_rounds)
        fit_xgb(model, fit, columns)
        valid["inner_prob_raw"] = model.predict_proba(matrix(valid, columns))[:, 1]
        valid["InnerFold"] = inner_fold
        inner_rows.append(valid[["Patient_ID", "ICULOS", "SepsisLabel", "inner_prob_raw", "InnerFold"]])
    inner_oof = pd.concat(inner_rows, ignore_index=True)
    if inner_oof.duplicated(["Patient_ID", "ICULOS"]).any() or len(inner_oof) != len(train):
        raise PipelineError("Nested inner OOF predictions do not cover outer-train rows exactly once")
    return winner, selected_rounds, inner_oof


def fitted_platt(inner_oof: pd.DataFrame, split_seed: int = SEED) -> LogisticRegression:
    y = inner_oof["SepsisLabel"].to_numpy(dtype=int)
    if set(y) != {0, 1}:
        raise PipelineError("Nested calibration requires both classes in inner OOF predictions")
    calibrator = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000, random_state=split_seed)
    calibrator.fit(inner_oof["inner_prob_raw"].to_numpy(dtype=float).reshape(-1, 1), y, sample_weight=equal_patient_weights(inner_oof))
    return calibrator


def platt_probabilities(calibrator: LogisticRegression, probability: Any) -> np.ndarray:
    """Apply the one-dimensional nested calibrator without DataFrame-name coupling."""
    return calibrator.predict_proba(np.asarray(probability, dtype=float).reshape(-1, 1))[:, 1]


def threshold_from_inner_oof(inner_oof: pd.DataFrame, probability_column: str) -> float:
    utilities = []
    for threshold in FEATURE_POLICY["threshold_grid"]:
        utilities.append((challenge_utility(inner_oof, probability_column, threshold), threshold))
    maximum = max(value for value, _ in utilities)
    return min(threshold for value, threshold in utilities if np.isclose(value, maximum))


def challenge_utility(frame: pd.DataFrame, probability_column: str, threshold: float) -> float:
    """Use the official scorer's sole utility definition over source-qualified patients."""
    required = {"Patient_ID", "ICULOS", "SepsisLabel", probability_column}
    if missing := required.difference(frame.columns):
        raise PipelineError(f"Utility requires {sorted(missing)}")
    observed_total = 0.0
    best_total = 0.0
    inaction_total = 0.0
    for _, patient in frame.sort_values(["Patient_ID", "ICULOS"], kind="mergesort").groupby("Patient_ID", sort=False):
        labels = patient["SepsisLabel"].to_numpy(dtype=int)
        predictions = (patient[probability_column].to_numpy(dtype=float) >= threshold).astype(int)
        if not np.isfinite(patient[probability_column]).all():
            raise PipelineError("Utility cannot score missing probability values")
        best = np.zeros(len(labels), dtype=int)
        if labels.any():
            onset = int(np.argmax(labels) - (-6))
            best[max(0, onset - 12): min(len(labels), onset + 3 + 1)] = 1
        observed_total += float(official_utility.compute_prediction_utility(labels, predictions))
        best_total += float(official_utility.compute_prediction_utility(labels, best))
        inaction_total += float(official_utility.compute_prediction_utility(labels, np.zeros(len(labels), dtype=int)))
    denominator = best_total - inaction_total
    if denominator <= 0:
        raise PipelineError("Official Utility normalization denominator is non-positive")
    return float((observed_total - inaction_total) / denominator)


def equal_patient_weights(frame: pd.DataFrame) -> np.ndarray:
    if "Patient_ID" not in frame:
        raise PipelineError("Patient-balanced calibration requires Patient_ID")
    counts = frame.groupby("Patient_ID", sort=False)["Patient_ID"].transform("size").to_numpy(dtype=float)
    return 1.0 / counts


def calibration_metrics(y: np.ndarray, probability: np.ndarray, sample_weight: np.ndarray | None = None) -> dict[str, float]:
    probability = np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)
    y = np.asarray(y, dtype=int)
    weights = np.ones(len(y), dtype=float) if sample_weight is None else np.asarray(sample_weight, dtype=float)
    if len(y) != len(probability) or len(y) != len(weights) or not np.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
        raise PipelineError("Invalid calibration inputs or weights")
    bins = FEATURE_POLICY["ece_equal_width_bins"]
    ece = 0.0
    bin_ids = np.minimum((probability * bins).astype(int), bins - 1)
    for bin_id in range(bins):
        mask = bin_ids == bin_id
        weight = weights[mask]
        if weight.sum() > 0:
            ece += float(weight.sum() / weights.sum() * abs(np.average(y[mask], weights=weight) - np.average(probability[mask], weights=weight)))
    logit = np.log(probability / (1 - probability))
    logit_mean = float(np.average(logit, weights=weights))
    logit_scale = float(np.sqrt(np.average((logit - logit_mean) ** 2, weights=weights)))
    if not np.isfinite(logit_scale) or logit_scale <= np.finfo(float).eps:
        raise PipelineError("Calibration slope is not identifiable from a constant logit")
    standardized_logit = ((logit - logit_mean) / logit_scale).reshape(-1, 1)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            model = LogisticRegression(penalty="none", solver="lbfgs", max_iter=1000).fit(
                standardized_logit, y, sample_weight=weights
            )
    except ConvergenceWarning as exc:
        raise PipelineError("Unpenalized calibration regression did not converge after weighted logit standardization") from exc
    calibration_slope = float(model.coef_[0, 0] / logit_scale)
    calibration_intercept = float(model.intercept_[0] - model.coef_[0, 0] * logit_mean / logit_scale)
    lower, upper = -40.0, 40.0
    for _ in range(100):
        midpoint = (lower + upper) / 2
        fitted = 1 / (1 + np.exp(-(logit + midpoint)))
        if np.average(y - fitted, weights=weights) > 0:
            lower = midpoint
        else:
            upper = midpoint
    return {
        "brier": float(brier_score_loss(y, probability, sample_weight=weights)),
        "ece_fixed_10_bins": float(ece),
        "calibration_in_the_large_intercept_slope_fixed_1": float((lower + upper) / 2),
        "calibration_intercept_with_slope": calibration_intercept,
        "calibration_slope": calibration_slope,
    }


def calibration_metrics_with_patient_uncertainty(frame: pd.DataFrame, probability_column: str, repeats: int | None = None) -> dict[str, Any]:
    y = frame["SepsisLabel"].to_numpy(dtype=int)
    probability = frame[probability_column].to_numpy(dtype=float)
    point = calibration_metrics(y, probability)
    repeats = FEATURE_POLICY["calibration_patient_cluster_bootstrap_repeats"] if repeats is None else repeats
    codes, patients = pd.factorize(frame["Patient_ID"], sort=True)
    if repeats < 1 or (codes < 0).any() or len(patients) < 2:
        raise PipelineError("Invalid patient-cluster calibration bootstrap context")
    rng = np.random.default_rng(SEED)
    samples = {name: [] for name in point}
    for _ in range(repeats):
        weights = rng.multinomial(len(patients), np.full(len(patients), 1 / len(patients)))[codes].astype(float)
        if set(y[weights > 0]) != {0, 1}:
            raise PipelineError("Patient-cluster calibration bootstrap draw lacks an outcome class")
        for name, value in calibration_metrics(y, probability, sample_weight=weights).items():
            samples[name].append(value)
    result = {**point, "calibration_estimand_unit": "observed row-time", "point_estimate_weighting": "each observed row-time equally weighted", "uncertainty_method": "patient-cluster bootstrap percentile 95% CI", "uncertainty_unit": "patient; row-time calibration estimand preserved within resampled patients", "uncertainty_repeats": int(repeats), "uncertainty_confidence_level": 0.95}
    for name, values in samples.items():
        result[f"{name}_ci_95_low"] = float(np.quantile(values, 0.025))
        result[f"{name}_ci_95_high"] = float(np.quantile(values, 0.975))
    return result


def discrimination_metrics(y: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    if set(np.asarray(y, dtype=int)) != {0, 1}:
        raise PipelineError("Discrimination metrics require both classes")
    precision, recall, _ = precision_recall_curve(y, probability)
    return {
        "auroc": float(roc_auc_score(y, probability)),
        "average_precision": float(average_precision_score(y, probability)),
        "trapezoidal_pr_auc": float(auc(recall[::-1], precision[::-1])),
        "xgboost_training_eval_metric": "logloss",
    }


def reliability_rows(y: np.ndarray, probability: np.ndarray, model: str, probability_kind: str) -> list[dict[str, Any]]:
    probability = np.asarray(probability, dtype=float)
    y = np.asarray(y, dtype=int)
    bins = FEATURE_POLICY["ece_equal_width_bins"]
    bin_ids = np.minimum((probability * bins).astype(int), bins - 1)
    rows: list[dict[str, Any]] = []
    for bin_id in range(bins):
        mask = bin_ids == bin_id
        rows.append({
            "model": model,
            "probability_kind": probability_kind,
            "bin": bin_id,
            "lower": bin_id / bins,
            "upper": (bin_id + 1) / bins,
            "n": int(mask.sum()),
            "mean_prediction": float(probability[mask].mean()) if mask.any() else math.nan,
            "observed_frequency": float(y[mask].mean()) if mask.any() else math.nan,
        })
    return rows


def alarm_episodes(times: np.ndarray, positive: np.ndarray, refractory_hours: int) -> list[float]:
    episodes: list[float] = []
    previous_positive = False
    for time, is_positive in zip(times, positive):
        if is_positive and not previous_positive and (not episodes or float(time) - episodes[-1] >= refractory_hours):
            episodes.append(float(time))
        previous_positive = bool(is_positive)
    return episodes


def early_warning_metrics(frame: pd.DataFrame, probability_column: str, threshold_column: str) -> dict[str, Any]:
    policy = FEATURE_POLICY["early_warning"]
    patients = 0
    septic = 0
    nonseptic = 0
    onset_unidentifiable_septic = 0
    true_positive_patients = 0
    false_negative_patients = 0
    false_alert_episodes = 0
    repeated_alerts = 0
    post_onset_episodes = 0
    total_alert_decision_hours = 0
    total_alarm_episodes = 0
    total_observation_hours = 0.0
    lead_times: list[float] = []
    rows: list[dict[str, Any]] = []
    for patient_id, patient in frame.sort_values(["Patient_ID", "ICULOS"], kind="mergesort").groupby("Patient_ID", sort=False):
        patients += 1
        onset_values = patient["TrueSepsisOnset_ICULOS"].dropna().unique()
        threshold_values = patient[threshold_column].dropna().unique()
        if len(onset_values) > 1 or len(threshold_values) != 1 or not np.isfinite(patient[probability_column]).all():
            raise PipelineError("Early-warning input has inconsistent onset/threshold or non-finite probability")
        onset = float(onset_values[0]) if len(onset_values) else math.nan
        threshold = float(threshold_values[0])
        if not 0 <= threshold <= 1:
            raise PipelineError("Early-warning threshold must lie in [0,1]")
        times = patient["ICULOS"].to_numpy(dtype=float)
        positive = patient[probability_column].to_numpy(dtype=float) >= threshold
        alerts = times[positive]
        episodes = alarm_episodes(times, positive, policy["refractory_hours"])
        total_alert_decision_hours += len(alerts)
        total_alarm_episodes += len(episodes)
        repeated_alerts += max(0, len(alerts) - len(episodes))
        total_observation_hours += float(times[-1] - times[0] + 1)
        eligible: list[float] = []
        is_septic = bool(patient["SepsisLabel"].max())
        onset_eligible = is_septic and math.isfinite(onset)
        if onset_eligible:
            septic += 1
            lower = onset - policy["start_hours_before_onset"]
            upper = onset - policy["end_hours_before_onset"]
            eligible = [time for time in episodes if lower <= time <= upper]
            post_onset_episodes += sum(time >= onset for time in episodes)
            false_alert_episodes += sum(time < lower for time in episodes)
            if eligible:
                true_positive_patients += 1
                lead_times.append(onset - eligible[0])
                status = "TP_patient_useful_window"
            else:
                false_negative_patients += 1
                status = "FN_patient_no_useful_window_alert"
        elif is_septic:
            septic += 1
            onset_unidentifiable_septic += 1
            status = "EXCLUDED_septic_onset_left_censored"
        else:
            nonseptic += 1
            false_alert_episodes += len(episodes)
            status = "TN_patient" if not episodes else "FP_patient"
        rows.append({
            "Patient_ID": patient_id,
            "septic": is_septic,
            "onset_eligible": onset_eligible,
            "true_onset_iculos": onset,
            "threshold": threshold,
            "alert_rows": int(len(alerts)),
            "time_in_alert_observed_decision_hours": int(len(alerts)),
            "time_in_alert_fraction_observed": len(alerts) / len(times),
            "alarm_episodes": int(len(episodes)),
            "eligible_episodes": int(len(eligible)),
            "first_eligible_alert_iculos": eligible[0] if eligible else math.nan,
            "status": status,
        })
    return {
        "summary": {
            "estimand": "patient-level useful early-warning window [onset-12h,onset-1h]",
            "probability_source": probability_column,
            "threshold_source": threshold_column,
            "refractory_hours": policy["refractory_hours"],
            "n_patients": patients,
            "n_septic_patients": septic,
            "n_nonseptic_patients": nonseptic,
            "n_onset_eligible_septic_patients": septic - onset_unidentifiable_septic,
            "n_left_censored_septic_patients_excluded_from_onset_estimands": onset_unidentifiable_septic,
            "tp_patients": true_positive_patients,
            "fn_patients": false_negative_patients,
            "useful_early_alert_sensitivity": true_positive_patients / (septic - onset_unidentifiable_septic) if septic > onset_unidentifiable_septic else math.nan,
            "median_lead_time_hours": float(np.median(lead_times)) if lead_times else math.nan,
            "post_onset_alarm_episodes": post_onset_episodes,
            "false_alarm_episodes": false_alert_episodes,
            "repeated_alert_rows_suppressed_by_refractory_policy": repeated_alerts,
            "alarm_episode_policy": f"a negative-to-positive threshold crossing opens an episode and a {policy['refractory_hours']}h refractory period; persistence alone cannot open another episode",
            "n_alarm_episodes": total_alarm_episodes,
            "time_in_alert_observed_decision_hours": total_alert_decision_hours,
            "time_in_alert_fraction_observed": total_alert_decision_hours / total_observation_hours if total_observation_hours else math.nan,
            "mean_alert_decision_hours_per_episode": total_alert_decision_hours / total_alarm_episodes if total_alarm_episodes else math.nan,
            "alert_decision_hours_per_patient_day": total_alert_decision_hours / (total_observation_hours / 24) if total_observation_hours else math.nan,
            "false_alarm_episodes_per_patient_day": false_alert_episodes / (total_observation_hours / 24) if total_observation_hours else math.nan,
        },
        "patients": rows,
    }


def paired_early_warning_comparison(baseline: pd.DataFrame, enhanced: pd.DataFrame) -> dict[str, Any]:
    key = ["Patient_ID", "ICULOS", "SepsisLabel", "Fold"]
    base = baseline.sort_values(key).reset_index(drop=True)
    enh = enhanced.sort_values(key).reset_index(drop=True)
    if not base[key].equals(enh[key]):
        raise PipelineError("Paired early-warning comparison requires aligned OOF rows")
    patient_tables = []
    excluded_left_censored = None
    for name, frame in (("baseline", base), ("enhanced", enh)):
        rows = pd.DataFrame(early_warning_metrics(frame, "prob_platt", "nested_threshold")["patients"])
        excluded = int((rows["septic"] & ~rows["onset_eligible"]).sum())
        if excluded_left_censored is not None and excluded != excluded_left_censored:
            raise PipelineError("Paired early-warning models disagree on onset eligibility")
        excluded_left_censored = excluded
        rows = rows.loc[rows["onset_eligible"], ["Patient_ID", "true_onset_iculos", "first_eligible_alert_iculos"]]
        rows[f"{name}_lead_time_hours"] = rows["true_onset_iculos"] - rows.pop("first_eligible_alert_iculos")
        patient_tables.append(rows.drop(columns="true_onset_iculos") if name == "enhanced" else rows)
    paired = patient_tables[0].merge(patient_tables[1], on="Patient_ID", validate="one_to_one")
    both = paired.dropna(subset=["baseline_lead_time_hours", "enhanced_lead_time_hours"])
    differences = both["enhanced_lead_time_hours"] - both["baseline_lead_time_hours"]
    ci = bootstrap_ci(differences.to_numpy()) if len(differences) else (math.nan, math.nan)
    return {
        "estimand": "paired lead-time difference among septic patients detected by both models inside the fixed onset-12h to onset-1h window",
        "probability_kind": "nested_platt",
        "threshold_provenance": "fold-specific threshold selected on outer-train inner OOF official Utility only",
        "n_onset_eligible_septic_patients": int(len(paired)),
        "n_left_censored_septic_patients_excluded": int(excluded_left_censored or 0),
        "baseline_detected": int(paired["baseline_lead_time_hours"].notna().sum()),
        "enhanced_detected": int(paired["enhanced_lead_time_hours"].notna().sum()),
        "detected_by_both": int(len(both)),
        "baseline_only": int((paired["baseline_lead_time_hours"].notna() & paired["enhanced_lead_time_hours"].isna()).sum()),
        "enhanced_only": int((paired["baseline_lead_time_hours"].isna() & paired["enhanced_lead_time_hours"].notna()).sum()),
        "missed_by_both": int(paired[["baseline_lead_time_hours", "enhanced_lead_time_hours"]].isna().all(axis=1).sum()),
        "mean_enhanced_minus_baseline_lead_time_hours_among_both": float(differences.mean()) if len(differences) else math.nan,
        "median_enhanced_minus_baseline_lead_time_hours_among_both": float(np.median(differences)) if len(differences) else math.nan,
        "paired_bootstrap_ci_95_low": ci[0],
        "paired_bootstrap_ci_95_high": ci[1],
        "interpretation": "Detection counts retain every onset-eligible septic patient; timing is secondary and conditional on detection by both models. The confidence interval is for the paired mean difference; the median is descriptive.",
    }


def cohort_flow_summary(features: pd.DataFrame, harmonized_stage: dict[str, Any]) -> dict[str, Any]:
    patients = features.groupby("Patient_ID", sort=True).agg(SourceSet=("SourceSet", "first"), septic=("SepsisLabel", "max"))
    if len(features) != harmonized_stage["row_count"] or len(patients) != harmonized_stage["patient_count"]:
        raise PipelineError("Cohort flow does not match harmonized provenance")
    sources = {}
    for source, group in features.groupby("SourceSet", sort=True):
        source_patients = patients[patients["SourceSet"] == source]
        sources[source] = {"rows": int(len(group)), "patients": int(len(source_patients)), "septic_patients": int(source_patients["septic"].sum())}
    return {
        "population": "complete validated public PhysioNet/CinC 2019 training cohorts A and B",
        "available_patients": int(len(patients)), "included_patients": int(len(patients)), "excluded_patients": 0,
        "rows": int(len(features)), "septic_patients": int(patients["septic"].sum()),
        "nonseptic_patients": int((patients["septic"] == 0).sum()), "source_sets": sources,
        "calendar_period_available": False, "hospital_identity_beyond_SourceSet_available": False,
        "exclusion_policy": "fail-closed schema/provenance validation; no post-validation patient exclusion",
    }


def measurement_support_rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    times = frame["ICULOS"]
    strata = (
        ("all_hours", pd.Series(True, index=frame.index)),
        ("ICULOS_1_6h", times.between(1, 6)),
        ("ICULOS_7_12h", times.between(7, 12)),
        ("ICULOS_13_24h", times.between(13, 24)),
        ("ICULOS_25h_plus", times >= 25),
    )
    for signal in HEMODYNAMIC_COLUMNS:
        column = f"{signal}_sampen_effective_n_24h"
        for stratum, mask in strata:
            values = frame.loc[mask, column].to_numpy(dtype=float)
            rows.append({
                "signal": signal, "time_stratum": stratum, "n_rows": int(len(values)),
                "median_effective_n": float(np.median(values)) if len(values) else math.nan,
                "q1_effective_n": float(np.quantile(values, 0.25)) if len(values) else math.nan,
                "q3_effective_n": float(np.quantile(values, 0.75)) if len(values) else math.nan,
                "fraction_meeting_sampen_minimum": float(np.mean(values >= FEATURE_POLICY["sampen"]["min_observations"])) if len(values) else math.nan,
                "model_input": False,
            })
    return rows


def age_subgroup_metrics(frame: pd.DataFrame, probability_column: str) -> list[dict[str, Any]]:
    groups = pd.cut(frame["Age"], [-np.inf, 50, 70, np.inf], labels=["<50", "50_to_<70", ">=70"], right=False).astype(object)
    groups[pd.isna(groups)] = "missing"
    rows: list[dict[str, Any]] = []
    for group in ("<50", "50_to_<70", ">=70", "missing"):
        subset = frame.loc[groups == group]
        y = subset["SepsisLabel"].to_numpy(dtype=int)
        probability = subset[probability_column].to_numpy(dtype=float)
        both_classes = set(y) == {0, 1}
        rows.append({
            "subgroup_schema_version": "age_v1_left_closed_50_70",
            "probability_source": probability_column,
            "subgroup": group,
            "unit": "descriptive row-time performance; no independent-row inference",
            "n_rows": int(len(subset)),
            "n_patients": int(subset["Patient_ID"].nunique()),
            "n_positive_rows": int(y.sum()),
            "auroc": float(roc_auc_score(y, probability)) if both_classes else math.nan,
            "average_precision": float(average_precision_score(y, probability)) if both_classes else math.nan,
            "brier": float(brier_score_loss(y, probability)) if len(y) else math.nan,
        })
    return rows


def temporal_stratified_metrics(frame: pd.DataFrame, probability_column: str) -> list[dict[str, Any]]:
    times = frame["ICULOS"].to_numpy(dtype=float)
    onset = frame["TrueSepsisOnset_ICULOS"].to_numpy(dtype=float)
    relative = times - onset
    strata = (
        ("time_since_icu_admission", "ICULOS_1_6h", (times >= 1) & (times <= 6)),
        ("time_since_icu_admission", "ICULOS_7_12h", (times >= 7) & (times <= 12)),
        ("time_since_icu_admission", "ICULOS_13_24h", (times >= 13) & (times <= 24)),
        ("time_since_icu_admission", "ICULOS_25_48h", (times >= 25) & (times <= 48)),
        ("time_since_icu_admission", "ICULOS_49h_plus", times >= 49),
        ("time_relative_to_true_onset", "remote_pre_onset_before_12h", relative < -12),
        ("time_relative_to_true_onset", "useful_window_onset_minus_12_to_1h", (relative >= -12) & (relative <= -1)),
        ("time_relative_to_true_onset", "post_onset_0h_plus", relative >= 0),
    )
    rows: list[dict[str, Any]] = []
    for axis, stratum, mask in strata:
        subset = frame.loc[mask]
        y = subset["SepsisLabel"].to_numpy(dtype=int)
        probability = subset[probability_column].to_numpy(dtype=float)
        both_classes = set(y) == {0, 1}
        rows.append({
            "axis": axis,
            "stratum": stratum,
            "probability_source": probability_column,
            "estimand": "descriptive row-time performance; patient dependence retained, no independent-row inference",
            "n_rows": int(len(subset)),
            "n_patients": int(subset["Patient_ID"].nunique()),
            "n_positive_rows": int(y.sum()),
            "positive_prevalence": float(y.mean()) if len(y) else math.nan,
            "auroc": float(roc_auc_score(y, probability)) if both_classes else math.nan,
            "average_precision": float(average_precision_score(y, probability)) if both_classes else math.nan,
            "brier": float(brier_score_loss(y, probability)) if len(y) else math.nan,
        })
    return rows


def decision_curve(frame: pd.DataFrame, probability_column: str, thresholds: Iterable[float]) -> list[dict[str, Any]]:
    """DCA for assessment now when true sepsis onset is within six hours."""
    rows: list[dict[str, Any]] = []
    decision_frames = []
    left_censored_septic = 0
    for patient_id, patient in frame.sort_values(["Patient_ID", "ICULOS"], kind="mergesort").groupby("Patient_ID", sort=False):
        onset = patient["TrueSepsisOnset_ICULOS"].dropna().unique()
        septic = bool(patient["SepsisLabel"].max())
        if septic and not len(onset):
            left_censored_septic += 1
            continue
        at_risk = patient["ICULOS"] < float(onset[0]) if septic else pd.Series(True, index=patient.index)
        eligible = patient.loc[at_risk, ["ICULOS", probability_column]].copy()
        eligible["Patient_ID"] = patient_id
        hours_to_onset = float(onset[0]) - eligible["ICULOS"].to_numpy(dtype=float) if septic else np.full(len(eligible), np.inf)
        eligible["outcome_onset_within_6h"] = ((hours_to_onset > 0) & (hours_to_onset <= 6)).astype(int)
        eligible["probability"] = eligible.pop(probability_column).astype(float)
        decision_frames.append(eligible[["Patient_ID", "outcome_onset_within_6h", "probability"]])
    decisions = pd.concat(decision_frames, ignore_index=True) if decision_frames else pd.DataFrame()
    if decisions.empty or set(decisions["outcome_onset_within_6h"]) != {0, 1} or not np.isfinite(decisions["probability"]).all():
        raise PipelineError("Cannot compute six-hour DCA without finite probabilities and both outcome classes")
    truth = decisions["outcome_onset_within_6h"].to_numpy(dtype=int)
    probability = decisions["probability"].to_numpy(dtype=float)
    codes, patients = pd.factorize(decisions["Patient_ID"], sort=True)
    patient_decision_counts = np.bincount(codes, minlength=len(patients))
    for threshold in thresholds:
        if not 0 < float(threshold) < 1:
            raise PipelineError("DCA threshold probabilities must lie strictly between zero and one")
        odds = float(threshold) / (1 - float(threshold))
        action = (probability >= threshold).astype(int)
        model_contribution = action * truth - action * (1 - truth) * odds
        treat_all_contribution = truth - (1 - truth) * odds
        patient_model_sum = np.bincount(codes, weights=model_contribution, minlength=len(patients))
        patient_all_sum = np.bincount(codes, weights=treat_all_contribution, minlength=len(patients))
        rng = np.random.default_rng(SEED + int(round(threshold * 1000)))
        model_samples = []
        all_samples = []
        for _ in range(FEATURE_POLICY["dca_patient_cluster_bootstrap_repeats"]):
            multiplicity = rng.multinomial(len(patients), np.full(len(patients), 1 / len(patients)))
            denominator = float(multiplicity @ patient_decision_counts)
            model_samples.append(float((multiplicity @ patient_model_sum) / denominator))
            all_samples.append(float((multiplicity @ patient_all_sum) / denominator))
        model_ci = tuple(float(value) for value in np.quantile(model_samples, [0.025, 0.975]))
        all_ci = tuple(float(value) for value in np.quantile(all_samples, [0.025, 0.975]))
        rows.append({
            "action": "initiate clinical assessment now for true sepsis onset within the next 6 hours",
            "outcome_estimand": "true reconstructed onset in (decision time, decision time + 6h]; post-onset and left-censored states excluded",
            "probability_source": probability_column,
            "threshold_probability": float(threshold),
            "model_net_benefit": float(model_contribution.mean()),
            "model_net_benefit_ci_95_low": model_ci[0],
            "model_net_benefit_ci_95_high": model_ci[1],
            "treat_all_net_benefit": float(treat_all_contribution.mean()),
            "treat_all_net_benefit_ci_95_low": all_ci[0],
            "treat_all_net_benefit_ci_95_high": all_ci[1],
            "treat_none_net_benefit": 0.0,
            "n_decision_hours": int(len(decisions)),
            "n_patients": int(len(patients)),
            "n_left_censored_septic_patients_excluded": int(left_censored_septic),
            "tp_decision_hours": int((action * truth).sum()),
            "fp_decision_hours": int((action * (1 - truth)).sum()),
            "estimand_unit": "observed pre-onset decision hour",
            "uncertainty_unit": "patient-cluster bootstrap",
        })
    return rows


def outer_oof(
    features: pd.DataFrame, folds: pd.DataFrame, variant: str, output_dir: Path, gpu: dict[str, Any], columns_override: list[str] | None = None, split_seed: int = SEED, artifact_stem: str | None = None, persist_oof: bool = True
) -> tuple[pd.DataFrame, dict[str, Any]]:
    merged = require_fold_context(features, folds)
    artifact_stem = artifact_stem or variant
    columns = model_features(merged, variant) if columns_override is None else columns_override
    records: list[pd.DataFrame] = []
    selection_rows: list[dict[str, Any]] = []
    for outer_fold in sorted(merged["Fold"].unique()):
        outer_train = merged[merged["Fold"] != outer_fold].copy()
        outer_test = merged[merged["Fold"] == outer_fold].copy()
        train_patients = set(outer_train["Patient_ID"])
        test_patients = set(outer_test["Patient_ID"])
        if train_patients & test_patients:
            raise PipelineError("Outer-fold patient overlap")
        candidate, selected_rounds, inner_oof = select_inner_model(outer_train, columns, gpu, int(outer_fold), split_seed)
        calibrator = fitted_platt(inner_oof, split_seed)
        inner_oof["inner_prob_platt"] = platt_probabilities(calibrator, inner_oof["inner_prob_raw"])
        threshold = threshold_from_inner_oof(inner_oof, "inner_prob_platt")
        model = xgb_model(candidate, split_seed + int(outer_fold), gpu, selected_rounds)
        fit_xgb(model, outer_train, columns)
        outer_test["prob_raw"] = model.predict_proba(matrix(outer_test, columns))[:, 1]
        outer_test["prob_platt"] = platt_probabilities(calibrator, outer_test["prob_raw"])
        outer_test["nested_threshold"] = threshold
        records.append(outer_test[OOF_OUTPUT_COLUMNS])
        selection_rows.append({
            "model_variant": variant,
            "outer_fold": int(outer_fold),
            "selected_candidate": candidate["id"],
            "selected_hyperparameters": json.dumps(candidate, sort_keys=True),
            "selected_tree_count_from_inner_only": selected_rounds,
            "inner_split_seed": split_seed + int(outer_fold) + 1,
            "inner_fold_patient_hash": stable_hash(
                inner_oof[["Patient_ID", "InnerFold"]].drop_duplicates().sort_values("Patient_ID").to_dict("records")
            ),
            "final_model_seed": split_seed + int(outer_fold),
            "calibrator": "pre-specified_platt_fit_on_inner_oof_only",
            "nested_threshold_from_inner_oof_only": threshold,
            "outer_train_patient_count": len(train_patients),
            "outer_test_patient_count": len(test_patients),
            "gpu_available": gpu["available"],
        })
    oof = pd.concat(records, ignore_index=True).sort_values(["Patient_ID", "ICULOS"], kind="mergesort").reset_index(drop=True)
    if len(oof) != len(merged) or oof.duplicated(["Patient_ID", "ICULOS"]).any():
        raise PipelineError("Outer OOF output does not contain each row exactly once")
    if persist_oof:
        atomic_csv(oof, output_dir / f"{artifact_stem}_oof_predictions.csv")
    atomic_csv(pd.DataFrame(selection_rows), output_dir / f"{artifact_stem}_nested_selection.csv")
    return oof, {
        "model_variant": variant,
        "feature_count": len(columns),
        "feature_column_hash": stable_hash(columns),
        "oof_artifact": f"{artifact_stem}_oof_predictions.csv" if persist_oof else None,
        "oof_sha256": sha256_file(output_dir / f"{artifact_stem}_oof_predictions.csv") if persist_oof else None,
        "selection_artifact": f"{artifact_stem}_nested_selection.csv",
        "selection_sha256": sha256_file(output_dir / f"{artifact_stem}_nested_selection.csv"),
        "gpu": gpu,
    }


def model_summary(oof: pd.DataFrame, variant: str, output_dir: Path, persist_artifacts: bool = True) -> dict[str, Any]:
    y = oof["SepsisLabel"].to_numpy(dtype=int)
    raw = oof["prob_raw"].to_numpy(dtype=float)
    calibrated = oof["prob_platt"].to_numpy(dtype=float)
    raw_calibration = calibration_metrics_with_patient_uncertainty(oof, "prob_raw")
    platt_calibration = calibration_metrics_with_patient_uncertainty(oof, "prob_platt")
    onset = oof["TrueSepsisOnset_ICULOS"].to_numpy(dtype=float)
    positive = y == 1
    positive_pre_onset = positive & np.isfinite(onset) & (oof["ICULOS"].to_numpy(dtype=float) < onset)
    positive_onset_or_post = positive & np.isfinite(onset) & ~positive_pre_onset
    positive_left_censored = positive & ~np.isfinite(onset)
    metrics = {
        "model_variant": variant,
        "population": "outer-fold held-out rows; patient grouping is retained for inference",
        "outcome_estimand": "PhysioNet/CinC Challenge shifted persistent SepsisLabel",
        "true_onset_definition": "first persistent challenge-positive ICULOS plus 6 hours",
        "n_rows": int(len(oof)),
        "n_patients": int(oof["Patient_ID"].nunique()),
        "n_positive_rows": int(y.sum()),
        "positive_label_composition": {
            "pre_onset_rows": int(positive_pre_onset.sum()),
            "onset_or_post_onset_rows": int(positive_onset_or_post.sum()),
            "left_censored_onset_unidentifiable_rows": int(positive_left_censored.sum()),
            "onset_or_post_onset_fraction_among_exact_onset_rows": float(positive_onset_or_post.sum() / (positive_pre_onset.sum() + positive_onset_or_post.sum())) if (positive_pre_onset.sum() + positive_onset_or_post.sum()) else math.nan,
            "interpretation": "Challenge-positive rows are not equivalent to fixed-horizon early warnings.",
        },
        "prevalence_only_brier_reference": float(y.mean() * (1 - y.mean())),
        "raw": {**discrimination_metrics(y, raw), **raw_calibration},
        "platt_nested": {
            **discrimination_metrics(y, calibrated), **platt_calibration,
            "discrimination_interpretation": "pooled cross-fit scores use fold-specific monotone calibrators; between-fold ranking may change and is not single-deployment-model discrimination",
        },
        "operating_policy_interpretation": "nested fold-specific thresholds are unbiased internal OOF policy evaluation, not a threshold for a final deployable model",
        "challenge_utility": {
            "raw_at_0_5": challenge_utility(oof, "prob_raw", 0.5),
            "platt_at_0_5": challenge_utility(oof, "prob_platt", 0.5),
            "raw_at_nested_fold_threshold": challenge_utility(
                oof.assign(_nested_probability=(oof["prob_raw"] >= oof["nested_threshold"]).astype(float)),
                "_nested_probability", 0.5,
            ),
            "platt_at_nested_fold_threshold": challenge_utility(
                oof.assign(_nested_probability=(oof["prob_platt"] >= oof["nested_threshold"]).astype(float)),
                "_nested_probability", 0.5,
            ),
            "definition": "official PhysioNet/CinC 2019 normalized utility; persistent shifted labels only",
        },
    }
    early = early_warning_metrics(oof, "prob_platt", "nested_threshold")
    metrics["early_warning"] = early["summary"]
    if variant in {"baseline", "enhanced"}:
        temporal_path = output_dir / f"{variant}_temporal_strata.csv"
        atomic_csv(pd.DataFrame(temporal_stratified_metrics(oof, "prob_platt")), temporal_path)
        age_path = output_dir / f"{variant}_age_subgroups.csv"
        atomic_csv(pd.DataFrame(age_subgroup_metrics(oof, "prob_platt")), age_path)
        metrics["age_subgroups"] = {"artifact": age_path.name, "schema": "age_v1_left_closed_50_70"}
        metrics["temporal_strata"] = {
            "artifact": temporal_path.name,
            "probability_kind": "platt_nested",
            "unit": "descriptive row-time strata; no independent-row inference",
        }
    if persist_artifacts:
        atomic_json(output_dir / f"{variant}_metrics.json", metrics)
        atomic_csv(pd.DataFrame(early["patients"]), output_dir / f"{variant}_early_warning_patients.csv")
        reliability = reliability_rows(y, raw, variant, "raw") + reliability_rows(y, calibrated, variant, "platt_nested")
        atomic_csv(pd.DataFrame(reliability), output_dir / f"{variant}_reliability.csv")
        atomic_csv(
            pd.DataFrame(decision_curve(oof, "prob_platt", np.arange(0.05, 0.51, 0.05))),
            output_dir / f"{variant}_dca.csv",
        )
    return metrics


def bootstrap_ci(values: np.ndarray, seed: int = SEED, repeats: int = 300) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    samples = np.array([rng.choice(values, size=len(values), replace=True).mean() for _ in range(repeats)])
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def benjamini_hochberg(p_values: dict[str, float]) -> dict[str, float]:
    """Return canonical step-up BH adjusted p-values for one declared family."""
    ordered = sorted(p_values, key=p_values.get)
    adjusted: dict[str, float] = {}
    running = 1.0
    for rank, name in reversed(list(enumerate(ordered, start=1))):
        running = min(running, p_values[name] * len(ordered) / rank)
        adjusted[name] = min(1.0, running)
    return adjusted


def paired_patient_permutation(
    baseline: pd.DataFrame, enhanced: pd.DataFrame, probability_column: str,
    repeats: int = FEATURE_POLICY["paired_inference_repeats"],
    probability_kind: str = "unspecified",
) -> list[dict[str, Any]]:
    """Cluster-respecting sharp-null permutation; not a bootstrap sign test."""
    key = ["Patient_ID", "ICULOS", "SepsisLabel", "Fold"]
    base = baseline.sort_values(key).reset_index(drop=True)
    enh = enhanced.sort_values(key).reset_index(drop=True)
    if not base[key].equals(enh[key]):
        raise PipelineError("Paired inference requires exact patient/ICULOS/label/fold alignment")
    y = base["SepsisLabel"].to_numpy(dtype=int)
    b = base[probability_column].to_numpy(dtype=float)
    e = enh[probability_column].to_numpy(dtype=float)
    observed = {
        "auroc": roc_auc_score(y, e) - roc_auc_score(y, b),
        "average_precision": average_precision_score(y, e) - average_precision_score(y, b),
        "brier": brier_score_loss(y, e) - brier_score_loss(y, b),
    }
    positions = {patient: np.flatnonzero(base["Patient_ID"].to_numpy() == patient) for patient in base["Patient_ID"].unique()}
    rng = np.random.default_rng(SEED)
    null: dict[str, list[float]] = {name: [] for name in observed}
    for _ in range(repeats):
        take_enhanced = {patient: bool(rng.integers(0, 2)) for patient in positions}
        first = b.copy()
        second = e.copy()
        for patient, index in positions.items():
            if not take_enhanced[patient]:
                first[index], second[index] = e[index], b[index]
        null["auroc"].append(float(roc_auc_score(y, second) - roc_auc_score(y, first)))
        null["average_precision"].append(float(average_precision_score(y, second) - average_precision_score(y, first)))
        null["brier"].append(float(brier_score_loss(y, second) - brier_score_loss(y, first)))
    family = ["auroc", "average_precision", "brier"]
    rows = []
    raw_p: dict[str, float] = {}
    for name in family:
        values = np.asarray(null[name])
        raw_p[name] = float((1 + np.sum(np.abs(values) >= abs(observed[name]))) / (len(values) + 1))
    adjusted = benjamini_hochberg(raw_p)
    codes, patients = pd.factorize(base["Patient_ID"], sort=True)
    bootstrap: dict[str, list[float]] = {name: [] for name in observed}
    bootstrap_rng = np.random.default_rng(SEED + 1)
    for _ in range(repeats):
        weights = bootstrap_rng.multinomial(len(patients), np.full(len(patients), 1 / len(patients)))[codes]
        if set(y[weights > 0]) != {0, 1}:
            raise PipelineError("Paired patient-cluster bootstrap draw lacks an outcome class")
        bootstrap["auroc"].append(float(roc_auc_score(y, e, sample_weight=weights) - roc_auc_score(y, b, sample_weight=weights)))
        bootstrap["average_precision"].append(float(average_precision_score(y, e, sample_weight=weights) - average_precision_score(y, b, sample_weight=weights)))
        bootstrap["brier"].append(float(brier_score_loss(y, e, sample_weight=weights) - brier_score_loss(y, b, sample_weight=weights)))
    for name in family:
        ci_low, ci_high = np.quantile(bootstrap[name], [0.025, 0.975])
        rows.append({
            "metric": name,
            "probability_kind": probability_kind,
            "enhanced_minus_baseline": observed[name],
            "paired_patient_cluster_bootstrap_ci_95_low": float(ci_low),
            "paired_patient_cluster_bootstrap_ci_95_high": float(ci_high),
            "bootstrap_repeats": repeats,
            "test": "paired_patient_cluster_permutation",
            "permutations": repeats,
            "p_value": raw_p[name],
            "p_value_bh_family_auroc_ap_brier": adjusted[name],
            "hypothesis_family": "predeclared: AUROC, Average Precision, Brier",
        })
    return rows


def logistic_representation_robustness(features: pd.DataFrame, folds: pd.DataFrame) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    merged = require_fold_context(features, folds)
    rows = []
    oof: dict[str, pd.DataFrame] = {}
    for variant in ("baseline", "enhanced"):
        columns = model_features(merged, variant)
        probability = np.full(len(merged), np.nan)
        for fold in sorted(merged["Fold"].unique()):
            train = merged["Fold"] != fold
            test = ~train
            model = make_pipeline(
                SimpleImputer(strategy="median"), StandardScaler(),
                SGDClassifier(loss="log_loss", penalty="l2", alpha=1e-4, max_iter=1000, tol=1e-4, random_state=SEED + int(fold)),
            )
            model.fit(matrix(merged.loc[train], columns), merged.loc[train, "SepsisLabel"], sgdclassifier__sample_weight=equal_patient_weights(merged.loc[train]))
            probability[test] = model.predict_proba(matrix(merged.loc[test], columns))[:, 1]
        if not np.isfinite(probability).all():
            raise PipelineError("L2 logistic robustness OOF predictions are incomplete")
        rows.append({
            "classifier": "sklearn_SGDClassifier_log_loss_l2", "model_variant": variant,
            "split_hash": stable_hash(folds.to_dict(orient="records")), "feature_column_hash": stable_hash(columns),
            "brier": float(brier_score_loss(merged["SepsisLabel"], probability)),
            **discrimination_metrics(merged["SepsisLabel"].to_numpy(dtype=int), probability),
        })
        oof[variant] = merged[["Patient_ID", "ICULOS", "SepsisLabel", "Fold"]].assign(probability=probability)
    inference = paired_patient_permutation(oof["baseline"], oof["enhanced"], "probability", probability_kind="uncalibrated_logistic_probability")
    for row in inference:
        row["classifier"] = "sklearn_SGDClassifier_log_loss_l2"
    return rows, inference


def fit_source_transport(features: pd.DataFrame, variant: str, train_source: str, test_source: str, gpu: dict[str, Any]) -> dict[str, Any]:
    columns = model_features(features, variant)
    train = features[features["SourceSet"] == train_source].copy()
    test = features[features["SourceSet"] == test_source].copy()
    if train.empty or test.empty:
        raise PipelineError("SourceSet transport needs both A and B")
    candidate, rounds, inner_oof = select_inner_model(train, columns, gpu, 100 + (0 if train_source == "A" else 1))
    calibrator = fitted_platt(inner_oof)
    inner_oof["inner_prob_platt"] = platt_probabilities(calibrator, inner_oof["inner_prob_raw"])
    threshold = threshold_from_inner_oof(inner_oof, "inner_prob_platt")
    model = xgb_model(candidate, SEED + 500, gpu, rounds)
    fit_xgb(model, train, columns)
    test["prob_platt"] = platt_probabilities(calibrator, model.predict_proba(matrix(test, columns))[:, 1])
    y = test["SepsisLabel"].to_numpy(dtype=int)
    return {
        "experiment": f"train_{train_source}_test_{test_source}",
        "validation_scope": "public SourceSet transport; not independent external validation",
        "model_variant": variant,
        "n_train_patients": int(train["Patient_ID"].nunique()),
        "n_test_patients": int(test["Patient_ID"].nunique()),
        "nested_train_source_threshold": threshold,
        "threshold_provenance": "train-source inner OOF official Utility only",
        "probability_kind": "nested_platt",
        "selected_candidate": candidate["id"],
        "selected_hyperparameters": json.dumps(candidate, sort_keys=True),
        "selected_tree_count_from_inner_only": rounds,
        "inner_split_seed": SEED + 101 + (0 if train_source == "A" else 1),
        "inner_fold_patient_hash": stable_hash(
            inner_oof[["Patient_ID", "InnerFold"]].drop_duplicates().sort_values("Patient_ID").to_dict("records")
        ),
        "final_model_seed": SEED + 500,
        "feature_count": len(columns),
        "feature_column_hash": stable_hash(columns),
        **discrimination_metrics(y, test["prob_platt"].to_numpy()),
        **calibration_metrics_with_patient_uncertainty(test, "prob_platt"),
        "challenge_utility_at_nested_train_source_threshold": challenge_utility(test, "prob_platt", threshold),
    }


def matched_permutation_control(features: pd.DataFrame, folds: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    merged = require_fold_context(features, folds)
    baseline = set(model_features(merged, "baseline"))
    enhanced_only = sorted(set(model_features(merged, "enhanced")).difference(baseline))
    rng = np.random.default_rng(SEED)
    for fold in sorted(merged["Fold"].unique()):
        index = merged.index[merged["Fold"] == fold]
        for column in enhanced_only:
            merged.loc[index, column] = rng.permutation(merged.loc[index, column].to_numpy())
    return merged.drop(columns="Fold"), enhanced_only


def ablation_columns(frame: pd.DataFrame, name: str) -> list[str]:
    columns = model_features(frame, "enhanced")
    explicit_process = tuple(column for column in columns if column.endswith(("_is_missing", "_observation_age_hours"))) + (
        "Unit1", "Unit2", "HospAdmTime", "ICULOS", "Measurement_Count",
    )
    removals = {
        "physiology_measurements_only": explicit_process + ("Age", "Gender"),
        "without_explicit_process": explicit_process,
        "without_iculos": ("ICULOS",),
        "without_hosp_adm_time": ("HospAdmTime",),
        "without_explicit_missingness_indicators": tuple(column for column in columns if column.endswith(("_is_missing", "_observation_age_hours"))) + ("Measurement_Count",),
        "without_cv": tuple(column for column in columns if "_cv_" in column),
        "without_iqr": tuple(column for column in columns if "_iqr_" in column),
        "without_sampen": tuple(column for column in columns if "_sampen_" in column),
    }
    additions = {
        "baseline_plus_cv": tuple(column for column in columns if "_cv_" in column),
        "baseline_plus_iqr": tuple(column for column in columns if "_iqr_" in column),
        "baseline_plus_sampen": tuple(column for column in columns if "_sampen_" in column),
    }
    if name in additions:
        baseline = model_features(frame, "baseline")
        return baseline + [column for column in additions[name] if column not in baseline]
    if name not in removals:
        raise PipelineError(f"Unknown ablation: {name}")
    filtered = [column for column in columns if column not in set(removals[name])]
    if not filtered:
        raise PipelineError(f"Ablation {name} removes every feature")
    return filtered


def artifact_hashes(run_dir: Path) -> dict[str, str]:
    return {
        str(path.relative_to(run_dir)): sha256_file(path)
        for path in sorted(run_dir.rglob("*"))
        if path.is_file() and path.name != "result_manifest.json"
    }


def validate_artifact_hashes(run_dir: Path, expected: dict[str, str]) -> int:
    observed = artifact_hashes(run_dir)
    if observed != expected:
        missing = sorted(set(expected).difference(observed))
        unexpected = sorted(set(observed).difference(expected))
        mismatched = sorted(name for name in set(expected).intersection(observed) if expected[name] != observed[name])
        raise PipelineError(f"Result artifact inventory mismatch: missing={missing}, unexpected={unexpected}, mismatched={mismatched}")
    return len(observed)


def validate_lineage_nodes(run_dir: Path, lineage: dict[str, Any]) -> int:
    nodes: dict[str, dict[str, Any]] = {}

    def collect(value: Any) -> None:
        if isinstance(value, dict) and value.get("kind") == "artifact_lineage":
            artifact = value.get("artifact")
            if not isinstance(artifact, str) or artifact in nodes:
                raise PipelineError("Lineage contains an invalid or duplicate artifact")
            nodes[artifact] = value
        elif isinstance(value, dict):
            for child in value.values():
                collect(child)

    collect(lineage)
    if not nodes:
        raise PipelineError("Lineage contains no artifact nodes")
    required = {"artifact", "sha256", "inputs", "generator", "generator_git_commit", "definition_ids"}
    for artifact, node in nodes.items():
        if missing := required.difference(node):
            raise PipelineError(f"Lineage node {artifact} missing {sorted(missing)}")
        artifact_path = Path(artifact)
        if not artifact_path.is_absolute():
            artifact_path = run_dir / artifact_path
        if not artifact_path.is_file() or sha256_file(artifact_path) != node["sha256"]:
            raise PipelineError(f"Lineage artifact hash mismatch: {artifact}")
        commit = node["generator_git_commit"]
        if not node["generator"] or not isinstance(commit, str) or len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit) or not node["definition_ids"] or not isinstance(node["inputs"], dict):
            raise PipelineError(f"Lineage node lacks semantic provenance: {artifact}")
        for input_artifact, input_sha256 in node["inputs"].items():
            if input_artifact not in nodes or nodes[input_artifact]["sha256"] != input_sha256:
                raise PipelineError(f"Lineage input is missing or mismatched: {artifact} <- {input_artifact}")
    return len(nodes)


def validate_final_manifest(run_dir: Path, allow_pending: bool = False) -> dict[str, Any]:
    manifest_path = run_dir / "result_manifest.json"
    if not manifest_path.is_file():
        raise PipelineError("Final result manifest is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    required = {
        "runtime", "stages", "artifact_sha256", "lineage", "final_validation", "scientific_status",
    }
    if missing := required.difference(manifest):
        raise PipelineError(f"Result manifest missing required fields: {sorted(missing)}")
    expected_status = "PENDING_FINAL_VALIDATION" if allow_pending else "COMPUTATIONAL_RUN_VALIDATED"
    if manifest["scientific_status"] != expected_status:
        raise PipelineError(f"Result manifest has invalid scientific status: {manifest['scientific_status']}")
    if not allow_pending and manifest["final_validation"].get("status") != "PASS":
        raise PipelineError("Final manifest has not passed final validation")
    validate_lineage_nodes(run_dir, manifest["lineage"])
    validate_artifact_hashes(run_dir, manifest["artifact_sha256"])
    runtime = manifest["runtime"]
    if runtime.get("data_policy") != DATA_POLICY or runtime.get("data_policy_hash") != stable_hash(DATA_POLICY) or runtime.get("feature_policy_hash") != stable_hash(FEATURE_POLICY):
        raise PipelineError("Final runtime policy provenance is invalid")
    cohort = json.loads((run_dir / "cohort_flow.json").read_text(encoding="utf-8"))
    if cohort.get("included_patients") != DATA_POLICY["patient_count"] or cohort.get("rows") != DATA_POLICY["row_count"] or cohort.get("excluded_patients") != 0:
        raise PipelineError("Final cohort flow does not match the pinned complete cohort")
    stability = pd.read_csv(run_dir / "split_stability.csv")
    required_stability = {"split_seed", "model_variant", "fold_artifact", "fold_sha256", "feature_column_hash", "auroc", "average_precision"}
    if len(stability) != 2 * len(FEATURE_POLICY["split_stability_seeds"]) or set(stability["split_seed"]) != set(FEATURE_POLICY["split_stability_seeds"]) or not required_stability.issubset(stability.columns):
        raise PipelineError("Repeated grouped split-stability artifact is invalid")
    for artifact, expected_hash in stability[["fold_artifact", "fold_sha256"]].drop_duplicates().itertuples(index=False):
        if sha256_file(run_dir / artifact) != expected_hash:
            raise PipelineError(f"Split-stability fold provenance is invalid: {artifact}")
    feature_identity = pd.read_csv(run_dir / "features.csv", usecols=["Patient_ID", "ICULOS", "SepsisLabel"])
    feature_schema = pd.read_csv(run_dir / "features.csv", nrows=0)
    oof_identities = []
    for variant in ("baseline", "enhanced"):
        oof = pd.read_csv(run_dir / f"{variant}_oof_predictions.csv")
        expected_features = model_features(feature_schema, variant)
        stage = manifest["stages"].get(variant, {})
        if stage.get("feature_count") != len(expected_features) or stage.get("feature_column_hash") != stable_hash(expected_features):
            raise PipelineError(f"Final {variant} model feature provenance is invalid")
        if list(oof.columns) != OOF_OUTPUT_COLUMNS:
            raise PipelineError(f"Final {variant} OOF artifact contains redundant or missing columns")
        if not (oof.groupby("Patient_ID")["Fold"].nunique() == 1).all():
            raise PipelineError("Final OOF fold provenance is invalid")
        if not np.isfinite(oof[["prob_raw", "prob_platt", "nested_threshold"]].to_numpy(dtype=float)).all():
            raise PipelineError("Final OOF contains non-finite scores or nested thresholds")
        identity = oof[["Patient_ID", "ICULOS", "SepsisLabel"]].sort_values(["Patient_ID", "ICULOS"], kind="mergesort").reset_index(drop=True)
        if not identity.equals(feature_identity.sort_values(["Patient_ID", "ICULOS"], kind="mergesort").reset_index(drop=True)):
            raise PipelineError(f"Final {variant} OOF identity does not match the feature cohort")
        oof_identities.append(identity)
        selection = pd.read_csv(run_dir / f"{variant}_nested_selection.csv")
        if len(selection) != 5 or set(selection["outer_fold"]) != set(range(5)) or not {"inner_fold_patient_hash", "calibrator", "nested_threshold_from_inner_oof_only"}.issubset(selection.columns):
            raise PipelineError(f"Final {variant} nested-selection provenance is invalid")
        dca = pd.read_csv(run_dir / f"{variant}_dca.csv")
        if len(dca) != 10 or not np.allclose(dca["threshold_probability"], np.arange(0.05, 0.51, 0.05)) or set(dca["estimand_unit"]) != {"observed pre-onset decision hour"}:
            raise PipelineError(f"Final {variant} DCA artifact is invalid")
        reliability = pd.read_csv(run_dir / f"{variant}_reliability.csv")
        if len(reliability) != 20 or set(reliability["probability_kind"]) != {"raw", "platt_nested"} or set(reliability["bin"]) != set(range(10)):
            raise PipelineError(f"Final {variant} reliability artifact is invalid")
    if not oof_identities[0].equals(oof_identities[1]):
        raise PipelineError("Final baseline/enhanced OOF identities are not paired")
    inference = pd.read_csv(run_dir / "inference.csv")
    if set(inference["metric"]) != {"auroc", "average_precision", "brier"} or set(inference["probability_kind"]) != {"nested_platt"}:
        raise PipelineError("Final paired inference artifact is invalid")
    transport = pd.read_csv(run_dir / "transport.csv")
    expected_transport = {(variant, f"train_{a}_test_{b}") for variant in ("baseline", "enhanced") for a, b in (("A", "B"), ("B", "A"))}
    if set(zip(transport["model_variant"], transport["experiment"])) != expected_transport or set(transport["probability_kind"]) != {"nested_platt"}:
        raise PipelineError("Final A/B transport artifact is invalid")
    ablations = pd.read_csv(run_dir / "ablations.csv")
    if len(ablations) != 12 or ablations["ablation"].nunique() != 12 or ablations[["definition", "feature_column_hash"]].isna().any().any():
        raise PipelineError("Final feature-ablation artifact is invalid")
    return {"status": "PASS", "validated_at_utc": utc_now(), "manifest": str(manifest_path)}


def stage_checkpoint(run_dir: Path, name: str, producer: Any, expected: dict[str, Any]) -> dict[str, Any]:
    artifact = run_dir / f"{name}.csv"
    manifest_path = run_dir / f"{name}_manifest.json"
    if artifact.exists() != manifest_path.exists():
        raise PipelineError(f"Incomplete {name} checkpoint; preserve evidence and remove only this invalid partial before resuming")
    if artifact.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("stage") != name or manifest.get("artifact_sha256") != sha256_file(artifact) or any(manifest.get(key) != value for key, value in expected.items()):
            raise PipelineError(f"Stale or mismatched {name} checkpoint")
        return manifest
    manifest = producer(artifact)
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise PipelineError(f"New {name} checkpoint has unexpected provenance")
    atomic_json(manifest_path, manifest)
    return manifest


def require_clean_resume_boundary(run_dir: Path) -> None:
    reusable = {"runtime_manifest.json", *(f"{name}{suffix}" for name in ("harmonized", "features", "folds") for suffix in (".csv", "_manifest.json"))}
    downstream = sorted(path.name for path in run_dir.iterdir() if path.name not in reusable)
    if downstream:
        raise PipelineError(f"Resume would overwrite downstream partials; preserve logs/evidence and remove only invalid partials first: {downstream}")


def run_scientific_pipeline(root: Path, archive: Path, run_dir: Path, run_id: str) -> dict[str, Any]:
    require_python_hash_seed()
    resume = os.environ.get("RESUME_EXISTING", "false").lower() == "true"
    if run_dir.exists() and not resume:
        raise PipelineError(f"Run directory already exists; refusing to overwrite evidence: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=resume)
    runtime = runtime_manifest(root, run_id, sys.argv, archive)
    existing_runtime = run_dir / "runtime_manifest.json"
    if resume and existing_runtime.is_file():
        previous = json.loads(existing_runtime.read_text(encoding="utf-8"))
        if previous.get("run_id") != runtime.get("run_id") or runtime_resume_context(previous) != runtime_resume_context(runtime):
            raise PipelineError("Resume runtime provenance does not match the existing run")
        runtime = previous
    elif resume and any(run_dir.iterdir()):
        raise PipelineError("Resume directory lacks its runtime provenance checkpoint")
    if runtime["git_dirty"]:
        raise PipelineError("Scientific runs require a clean committed checkout; refuse dirty provenance.")
    gpu = runtime["gpu"]
    if os.environ.get("REQUIRE_GPU", "false").lower() == "true" and not gpu["available"]:
        raise PipelineError("This scheduled run requires a validated GPU: {0}".format(gpu["reason"]))
    atomic_json(run_dir / "runtime_manifest.json", runtime)
    if resume and (run_dir / "result_manifest.json").is_file():
        return validate_final_manifest(run_dir)
    stages: dict[str, Any] = {}
    stages["harmonized"] = stage_checkpoint(run_dir, "harmonized", lambda output: harmonize_archive(archive, output), {"archive_sha256": runtime["data_archive_sha256"]})
    stages["features"] = stage_checkpoint(run_dir, "features", lambda output: build_features(run_dir / "harmonized.csv", output), {"input_sha256": stages["harmonized"]["artifact_sha256"]})
    feature_frame = pd.read_csv(run_dir / "features.csv")
    patient_hash = stable_hash(feature_frame.groupby("Patient_ID", sort=True)["SepsisLabel"].max().astype(int).reset_index().to_dict("records"))
    stages["folds"] = stage_checkpoint(run_dir, "folds", lambda output: write_folds(feature_frame, output), {"patient_inventory_hash": patient_hash, "seed": SEED})
    if resume:
        require_clean_resume_boundary(run_dir)
    features = pd.read_csv(run_dir / "features.csv")
    atomic_json(run_dir / "cohort_flow.json", cohort_flow_summary(features, stages["harmonized"]))
    atomic_csv(pd.DataFrame(measurement_support_rows(features)), run_dir / "measurement_support.csv")
    stages["measurement_support"] = {"artifact": "measurement_support.csv", "artifact_sha256": sha256_file(run_dir / "measurement_support.csv"), "input_sha256": stages["features"]["artifact_sha256"]}
    folds = pd.read_csv(run_dir / "folds.csv")
    summaries: dict[str, Any] = {}
    oofs: dict[str, pd.DataFrame] = {}
    for variant in ("baseline", "enhanced"):
        oofs[variant], stages[variant] = outer_oof(features, folds, variant, run_dir, gpu)
        summaries[variant] = model_summary(oofs[variant], variant, run_dir)
    atomic_json(run_dir / "metrics.json", summaries)
    stability_rows = [
        {"split_seed": SEED, "model_variant": variant, "fold_artifact": "folds.csv", "fold_sha256": stages["folds"]["artifact_sha256"], "feature_column_hash": stages[variant]["feature_column_hash"], **discrimination_metrics(oofs[variant]["SepsisLabel"].to_numpy(dtype=int), oofs[variant]["prob_platt"].to_numpy(dtype=float))}
        for variant in ("baseline", "enhanced")
    ]
    stability_manifests = []
    for split_seed in FEATURE_POLICY["split_stability_seeds"][1:]:
        fold_path = run_dir / f"stability_folds_seed_{split_seed}.csv"
        fold_manifest = write_folds(features, fold_path, split_seed=split_seed)
        stability_manifests.append(fold_manifest)
        stable_folds = pd.read_csv(fold_path)
        for variant in ("baseline", "enhanced"):
            stable_oof, detail = outer_oof(features, stable_folds, variant, run_dir, gpu, split_seed=split_seed, artifact_stem=f"stability_seed_{split_seed}_{variant}", persist_oof=False)
            stability_rows.append({"split_seed": split_seed, "model_variant": variant, "fold_artifact": fold_path.name, "fold_sha256": fold_manifest["artifact_sha256"], "feature_column_hash": detail["feature_column_hash"], **discrimination_metrics(stable_oof["SepsisLabel"].to_numpy(dtype=int), stable_oof["prob_platt"].to_numpy(dtype=float))})
    atomic_csv(pd.DataFrame(stability_rows), run_dir / "split_stability.csv")
    stages["split_stability"] = {"artifact": "split_stability.csv", "artifact_sha256": sha256_file(run_dir / "split_stability.csv"), "seeds": list(FEATURE_POLICY["split_stability_seeds"]), "additional_fold_manifests": stability_manifests}
    if not oofs["baseline"][["Patient_ID", "ICULOS", "SepsisLabel", "Fold"]].equals(
        oofs["enhanced"][["Patient_ID", "ICULOS", "SepsisLabel", "Fold"]]
    ):
        raise PipelineError("Baseline/enhanced OOF identity mismatch")
    inference = paired_patient_permutation(oofs["baseline"], oofs["enhanced"], "prob_platt", probability_kind="nested_platt")
    atomic_csv(pd.DataFrame(inference), run_dir / "inference.csv")
    atomic_json(run_dir / "paired_early_warning.json", paired_early_warning_comparison(oofs["baseline"], oofs["enhanced"]))
    transport = [
        fit_source_transport(features, variant, train_source, test_source, gpu)
        for variant in ("baseline", "enhanced")
        for train_source, test_source in (("A", "B"), ("B", "A"))
    ]
    atomic_csv(pd.DataFrame(transport), run_dir / "transport.csv")
    classifier_rows, classifier_inference = logistic_representation_robustness(features, folds)
    atomic_csv(pd.DataFrame(classifier_rows), run_dir / "classifier_robustness.csv")
    atomic_csv(pd.DataFrame(classifier_inference), run_dir / "classifier_robustness_inference.csv")
    ablations = []
    ablation_definitions = {
        "physiology_measurements_only": "Age, sex, time, unit and explicit process/missingness features removed; native NaN states remain observable",
        "without_explicit_process": "unit, admission/ICU time and explicit measurement-process indicators removed; native NaN states remain observable",
        "without_iculos": "enhanced feature set with ICULOS removed",
        "without_hosp_adm_time": "enhanced feature set with HospAdmTime removed",
        "without_explicit_missingness_indicators": "missingness flags and measurement count removed; native NaN states remain observable",
        "without_cv": "enhanced feature set with all 8 h coefficient-of-variation descriptors removed",
        "without_iqr": "enhanced feature set with all 8 h interquartile-range descriptors removed",
        "without_sampen": "enhanced feature set with all 24 h SampEn values and zero-match indicators removed",
        "baseline_plus_cv": "baseline feature set plus only the six 8 h coefficient-of-variation descriptors",
        "baseline_plus_iqr": "baseline feature set plus only the six 8 h interquartile-range descriptors",
        "baseline_plus_sampen": "baseline feature set plus only the six 24 h SampEn values and six zero-match indicators",
    }
    for name in (
        "physiology_measurements_only", "without_explicit_process", "without_iculos", "without_hosp_adm_time", "without_explicit_missingness_indicators",
        "without_cv", "without_iqr", "without_sampen",
        "baseline_plus_cv", "baseline_plus_iqr", "baseline_plus_sampen",
    ):
        oof, detail = outer_oof(features, folds, f"ablation_{name}", run_dir, gpu, ablation_columns(features, name), persist_oof=False)
        summary = model_summary(oof, f"ablation_{name}", run_dir, persist_artifacts=False)
        ablations.append({
            "ablation": name,
            "definition": ablation_definitions.get(name, f"pre-specified removal: {name}"),
            "feature_count": detail["feature_count"],
            "feature_column_hash": detail["feature_column_hash"],
            "average_precision_platt": summary["platt_nested"]["average_precision"],
            "auroc_platt": summary["platt_nested"]["auroc"],
            "brier_platt": summary["platt_nested"]["brier"],
            "utility_at_nested_threshold": summary["challenge_utility"]["platt_at_nested_fold_threshold"],
        })
    control, permuted_columns = matched_permutation_control(features, folds)
    control_oof, control_detail = outer_oof(control, folds, "ablation_permuted_enhanced", run_dir, gpu, model_features(control, "enhanced"), persist_oof=False)
    control_summary = model_summary(control_oof, "ablation_permuted_enhanced", run_dir, persist_artifacts=False)
    ablations.append({
        "ablation": "permuted_enhanced_matched_count",
        "definition": "enhanced-only columns independently permuted within outer folds without outcomes; predictor count unchanged",
        "feature_count": control_detail["feature_count"], "permuted_feature_count": len(permuted_columns),
        "feature_column_hash": control_detail["feature_column_hash"],
        "average_precision_platt": control_summary["platt_nested"]["average_precision"],
        "auroc_platt": control_summary["platt_nested"]["auroc"],
        "brier_platt": control_summary["platt_nested"]["brier"],
        "utility_at_nested_threshold": control_summary["challenge_utility"]["platt_at_nested_fold_threshold"],
    })
    atomic_csv(pd.DataFrame(ablations), run_dir / "ablations.csv")
    probast = {
        "instrument": "PROBAST+AI",
        "status": "NOT_LOW_RISK_UNTIL_INDEPENDENT_REVIEW",
        "technical_evidence": "nested grouped validation, provenance, and fail-closed gates are machine-checked; editorial appraisal remains author/reviewer work.",
    }
    atomic_json(run_dir / "probast_ai_status.json", probast)
    def node(artifact: str, inputs: dict[str, str], generator: str, definition_ids: tuple[str, ...]) -> dict[str, Any]:
        artifact_path = Path(artifact)
        if not artifact_path.is_absolute():
            artifact_path = run_dir / artifact_path
        return {
            "kind": "artifact_lineage",
            "artifact": artifact,
            "sha256": sha256_file(artifact_path),
            "inputs": inputs,
            "generator": generator,
            "generator_git_commit": runtime["git_commit"],
            "definition_ids": list(definition_ids),
        }

    raw_artifact = str(archive.resolve())
    raw_node = node(raw_artifact, {}, "external_input", ("physionet_cinc_2019_local_repackaging",))
    harmonized_node = node("harmonized.csv", {raw_artifact: raw_node["sha256"]}, "src.scientific_pipeline:harmonize_archive", ("official_40_predictor_schema_v1", "challenge_shifted_persistent_label"))
    features_node = node("features.csv", {"harmonized.csv": harmonized_node["sha256"]}, "src.scientific_pipeline:build_features", ("causal_feature_policy_v1",))
    cohort_node = node("cohort_flow.json", {"harmonized.csv": harmonized_node["sha256"], "features.csv": features_node["sha256"]}, "src.scientific_pipeline:cohort_flow_summary", ("complete_public_AB_cohort_flow_v1",))
    support_node = node("measurement_support.csv", {"features.csv": features_node["sha256"]}, "src.scientific_pipeline:measurement_support_rows", ("observed_sampen_effective_n_24h_v1",))
    folds_node = node("folds.csv", {"features.csv": features_node["sha256"]}, "src.scientific_pipeline:write_folds", ("stratified_group_kfold_patient_v1",))
    model_nodes = {
        variant: node(
            f"{variant}_oof_predictions.csv",
            {"features.csv": features_node["sha256"], "folds.csv": folds_node["sha256"]},
            "src.scientific_pipeline:outer_oof",
            ("nested_model_selection_v1", "nested_platt_calibration_v1", "nested_utility_threshold_v1"),
        )
        for variant in ("baseline", "enhanced")
    }
    product_definitions = {
        "metrics.json": ("auroc_sklearn", "average_precision_sklearn", "trapezoidal_pr_auc", "official_physionet_2019_utility", "ece_equal_width_10", "brier", "calibration_in_the_large_slope_fixed_1", "calibration_intercept_and_slope"),
        "early_warning_patients.csv": ("onset_anchored_early_warning_12_to_1h", "alarm_refractory_6h"),
        "reliability.csv": ("reliability_equal_width_10",),
        "dca.csv": ("pre_onset_decision_hour_six_hour_net_benefit_patient_cluster_uncertainty_v2",),
        "temporal_strata.csv": ("time_since_icu_and_true_onset_strata_v1",),
        "age_subgroups.csv": ("age_v1_left_closed_50_70",),
    }
    reported_products = {}
    for variant, model_node in model_nodes.items():
        for suffix, definitions in product_definitions.items():
            artifact = f"{variant}_{suffix}"
            reported_products[artifact] = node(artifact, {model_node["artifact"]: model_node["sha256"]}, "src.scientific_pipeline:model_summary", definitions)
    combined_metrics_node = node("metrics.json", {f"{variant}_metrics.json": reported_products[f"{variant}_metrics.json"]["sha256"] for variant in model_nodes}, "src.scientific_pipeline:run_scientific_pipeline", ("combined_current_run_metrics_v1",))
    reported_products["metrics.json"] = combined_metrics_node
    statistics_nodes = {
        "inference": node("inference.csv", {model_nodes[variant]["artifact"]: model_nodes[variant]["sha256"] for variant in model_nodes}, "src.scientific_pipeline:paired_patient_permutation", ("paired_patient_permutation_v1", "bh_primary_family_v1")),
        "paired_early_warning": node("paired_early_warning.json", {model_nodes[variant]["artifact"]: model_nodes[variant]["sha256"] for variant in model_nodes}, "src.scientific_pipeline:paired_early_warning_comparison", ("paired_detection_and_conditional_lead_time_v1",)),
        "transport": node("transport.csv", {"features.csv": features_node["sha256"]}, "src.scientific_pipeline:fit_source_transport", ("train_A_test_B_and_train_B_test_A_v1",)),
        "classifier_robustness": node("classifier_robustness.csv", {"features.csv": features_node["sha256"], "folds.csv": folds_node["sha256"]}, "src.scientific_pipeline:logistic_representation_robustness", ("l2_logistic_sgd_same_grouped_folds_v1",)),
        "classifier_robustness_inference": node("classifier_robustness_inference.csv", {"features.csv": features_node["sha256"], "folds.csv": folds_node["sha256"]}, "src.scientific_pipeline:logistic_representation_robustness", ("l2_logistic_paired_patient_permutation_v1",)),
        "ablations": node("ablations.csv", {"features.csv": features_node["sha256"], "folds.csv": folds_node["sha256"]}, "src.scientific_pipeline:outer_oof+model_summary", ("pre_specified_feature_family_ablations_v1",)),
        "split_stability": node("split_stability.csv", {"features.csv": features_node["sha256"]}, "src.scientific_pipeline:write_folds+outer_oof", ("repeated_grouped_split_seeds_v1",)),
    }
    probast_node = node("probast_ai_status.json", {name["artifact"]: name["sha256"] for name in statistics_nodes.values()}, "src.scientific_pipeline:run_scientific_pipeline", ("probast_ai_not_low_risk_until_independent_review",))
    lineage = {
        "raw_data": raw_node,
        "harmonized": harmonized_node,
        "features": features_node,
        "cohort_flow": cohort_node,
        "measurement_support": support_node,
        "folds": folds_node,
        "models_oof_calibration": model_nodes,
        "reported_products": reported_products,
        "statistics": statistics_nodes,
        "probast_ai": probast_node,
    }
    initial = {
        "runtime": runtime,
        "stages": stages,
        "lineage": lineage,
        "artifact_sha256": artifact_hashes(run_dir),
        "scientific_status": "PENDING_FINAL_VALIDATION",
        "final_validation": {"status": "PENDING"},
    }
    atomic_json(run_dir / "result_manifest.json", initial)
    validation = validate_final_manifest(run_dir, allow_pending=True)
    final = {**initial, "scientific_status": "COMPUTATIONAL_RUN_VALIDATED", "final_validation": validation}
    # The manifest changes after its own hash inventory. It intentionally does
    # not self-hash; every scientific result artifact is covered above.
    atomic_json(run_dir / "result_manifest.json", final)
    return final
