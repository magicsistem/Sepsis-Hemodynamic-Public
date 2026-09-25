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
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    auc,
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold

from src import onset_koopman as onset
from vendor.physionet2019 import evaluate_sepsis_score as official_utility


PIPELINE_VERSION = "scientific-pipeline-v8-bounded-stage-parallelism"
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
    "cv_window_hours": 8,
    "rolling_min_observations": 2,
    "dca_threshold_probabilities": tuple(round(x, 2) for x in np.arange(0.05, 0.51, 0.05)),
    "ece_equal_width_bins": 10,
    "calibration_patient_cluster_bootstrap_repeats": 300,
    "dca_patient_cluster_bootstrap_repeats": 300,
    "paired_inference_repeats": 300,
}
MODEL_CANDIDATES = (
    {"id": "depth3", "max_depth": 3, "learning_rate": 0.05, "min_child_weight": 1, "subsample": 0.8, "colsample_bytree": 0.8},
    {"id": "depth5", "max_depth": 5, "learning_rate": 0.05, "min_child_weight": 1, "subsample": 0.8, "colsample_bytree": 0.8},
)
MODEL_POLICY = {
    "outer_folds": 5,
    "inner_folds": 3,
    "selection_metric": "sklearn_average_precision_equal_total_weight_per_patient_on_inner_held_out_decision_hours",
    "candidate_tie_break": "lower max_depth after exactly equal mean inner-fold Average Precision",
    "tree_count_aggregation": "median inner best iteration then Python round, minimum one",
    "inner_oof_prediction_source": "held-out early-stopped winner predictions; no redundant refit",
    "candidates": MODEL_CANDIDATES,
    "early_stopping_max_estimators": 600,
    "early_stopping_rounds": 30,
    "xgboost_threads": "SEPSIS_FIT_THREADS_or_SLURM_CPUS_PER_TASK_capped_at_32",
    "maximum_fit_threads": 32,
    "maximum_total_cpu_threads": 64,
    "maximum_parallel_workers": 5,
    "maximum_parallel_candidates": 2,
    "planned_full_xgboost_fits": 278,
    "planned_full_koopman_fits": 49,
    "xgboost_objective": "binary:logistic",
    "xgboost_eval_metric": "logloss",
}
ROBUSTNESS_POLICY = {
    "representations": ("C0", "C3"),
    "training_seed_bases": (SEED, SEED + 101, SEED + 202),
    "balance_policies": (
        "equal_patient",
        "equal_row",
        "equal_patient_then_row_class",
    ),
    "estimand": "raw-probability ranking stability; no configuration is selected",
    "evaluation_weighting": "equal total weight per patient",
}
STAGE_POLICY = {
    "prepare_workers": 16,
    "finalize_workers": 4,
    "prepare_memory_gb": 32,
    "finalize_memory_gb": 32,
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
    def json_value(value: Any) -> Any:
        if isinstance(value, np.generic):
            return json_value(value.item())
        if isinstance(value, float) and not math.isfinite(value):
            return None
        if isinstance(value, dict):
            return {key: json_value(child) for key, child in value.items()}
        if isinstance(value, (list, tuple)):
            return [json_value(child) for child in value]
        return value

    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(json_value(payload), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        frame.to_csv(handle, index=False)
        temporary = Path(handle.name)
    temporary.replace(path)


def read_alarm_event_artifact(path: Path) -> pd.DataFrame:
    """Preserve serialized alarm-time text across the CSV round trip."""
    text_columns = (
        "Patient_ID", "alarm_episode_times_iculos",
        "useful_alarm_episode_times_iculos",
    )
    return pd.read_csv(path, dtype={column: str for column in text_columns})


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
        except Exception as exc:
            versions[optional] = f"unavailable:{type(exc).__name__}"
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
        "source_inventory_sha256": os.environ.get("SOURCE_INVENTORY_SHA256", "not-scheduled"),
        "command_line": command,
        "run_sh_sha256": sha256_file(root / "run.sh") if (root / "run.sh").is_file() else "not-created-yet",
        "data_archive_path": str(archive.resolve()),
        "data_archive_sha256": archive_hash,
        "data_policy": DATA_POLICY,
        "data_policy_hash": stable_hash(DATA_POLICY),
        "schema_version": "PhysioNet-CinC-2019-v1.0.0-40-predictors",
        "feature_policy": FEATURE_POLICY,
        "feature_policy_hash": stable_hash(FEATURE_POLICY),
        "model_policy": MODEL_POLICY,
        "model_policy_hash": stable_hash(MODEL_POLICY),
        "robustness_policy": ROBUSTNESS_POLICY,
        "robustness_policy_hash": stable_hash(ROBUSTNESS_POLICY),
        "stage_policy": STAGE_POLICY,
        "stage_policy_hash": stable_hash(STAGE_POLICY),
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
        "primary_target_policy": onset.TARGET_POLICY,
        "primary_target_policy_hash": stable_hash(onset.TARGET_POLICY),
        "koopman_policy": onset.KOOPMAN_POLICY,
        "koopman_policy_hash": stable_hash(onset.KOOPMAN_POLICY),
        "direct_onset_calibration_policy": onset.CALIBRATION_POLICY,
        "direct_onset_calibration_policy_hash": stable_hash(onset.CALIBRATION_POLICY),
        "direct_onset_alarm_policy": onset.ALARM_POLICY,
        "direct_onset_alarm_policy_hash": stable_hash(onset.ALARM_POLICY),
        "primary_representations": list(onset.REPRESENTATIONS),
    }


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
    label_array = binary_array(list(labels), "Onset reconstruction")
    time_array = np.asarray(list(times), dtype=float)
    if len(time_array) != len(label_array) or not np.isfinite(time_array).all() or (np.diff(time_array) <= 0).any():
        raise PipelineError("Onset reconstruction requires aligned strictly increasing finite times")
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
    identities = [source_and_patient(info.filename) for info in psv_members]
    patients = [patient for _, patient in identities]
    if len(patients) != len(set(patients)):
        duplicates = sorted(patient for patient, count in Counter(patients).items() if count > 1)
        raise PipelineError(f"Duplicate source-qualified patient id: {duplicates}")
    workers = stage_worker_count("prepare", len(psv_members))
    batches = [psv_members[index::workers] for index in range(workers)]

    def read_batch(batch: list[zipfile.ZipInfo]) -> list[pd.DataFrame]:
        frames = []
        with zipfile.ZipFile(archive) as zf:
            for info in batch:
                source, patient = source_and_patient(info.filename)
                with zf.open(info) as handle:
                    frame = pd.read_csv(handle, sep="|", dtype=str)
                frame.columns = canonical_headers(frame.columns, info.filename)
                frame = frame.loc[:, list(CHALLENGE_COLUMNS)].copy()
                frame = validate_patient_frame(frame, info.filename)
                frame.insert(0, "Patient_ID", patient)
                frame.insert(1, "SourceSet", source)
                onset_time, onset_status = reconstruct_true_onset(
                    frame["SepsisLabel"], frame["ICULOS"]
                )
                frame["TrueSepsisOnset_ICULOS"] = onset_time
                frame["OnsetReconstructionStatus"] = onset_status
                frames.append(frame)
        return frames

    frames = [
        frame
        for batch in ordered_parallel_map(read_batch, batches, workers)
        for frame in batch
    ]
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


def feature_patient(patient: pd.DataFrame, include_hemodynamics: bool) -> pd.DataFrame:
    patient = patient.sort_values("ICULOS", kind="mergesort").copy()
    times = patient["ICULOS"]
    identity = patient[["Patient_ID", "SourceSet", "ICULOS", "SepsisLabel", "TrueSepsisOnset_ICULOS", "OnsetReconstructionStatus"]].copy()
    engineered: dict[str, Any] = {column: patient[column].to_numpy() for column in STATIC_COLUMNS}
    engineered["Measurement_Count"] = patient.loc[:, DYNAMIC_COLUMNS].notna().sum(axis=1).to_numpy(dtype="int16")
    for column in DYNAMIC_COLUMNS:
        raw = pd.to_numeric(patient[column], errors="coerce")
        engineered[onset.raw_column(column)] = raw.to_numpy()
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
            variability_hours = FEATURE_POLICY["cv_window_hours"]
            mean_8h = rolling_feature(raw, variability_hours, "mean")
            std_8h = rolling_feature(raw, variability_hours, "std")
            engineered[f"{column}_cv_8h"] = (std_8h / mean_8h.abs()).replace([np.inf, -np.inf], np.nan).to_numpy()
    features = pd.concat([identity, pd.DataFrame(engineered, index=patient.index)], axis=1)
    try:
        return onset.add_primary_target(features)
    except onset.OnsetKoopmanError as exc:
        raise PipelineError(str(exc)) from exc


def build_features(harmonized: Path, output: Path) -> dict[str, Any]:
    frame = pd.read_csv(harmonized)
    required = {"Patient_ID", "SourceSet", *CHALLENGE_COLUMNS, "TrueSepsisOnset_ICULOS", "OnsetReconstructionStatus"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise PipelineError(f"Harmonized artifact is invalid; missing {missing}")
    frame[list(CHALLENGE_COLUMNS)] = numeric_columns(frame, CHALLENGE_COLUMNS, "harmonized artifact")
    frame[["TrueSepsisOnset_ICULOS"]] = numeric_columns(frame, ["TrueSepsisOnset_ICULOS"], "harmonized artifact")
    frame = frame.sort_values(["SourceSet", "Patient_ID", "ICULOS"], kind="mergesort").reset_index(drop=True)
    starts = np.flatnonzero(
        np.r_[True, frame["Patient_ID"].to_numpy()[1:] != frame["Patient_ID"].to_numpy()[:-1]]
    )
    workers = stage_worker_count("prepare", len(starts))
    patient_batches = [batch for batch in np.array_split(starts, workers) if len(batch)]
    bounds = [
        (int(batch[0]), int(starts[np.searchsorted(starts, batch[-1]) + 1]) if batch[-1] != starts[-1] else len(frame))
        for batch in patient_batches
    ]

    def feature_batch(bound: tuple[int, int]) -> pd.DataFrame:
        start, end = bound
        return pd.concat(
            [
                feature_patient(group, include_hemodynamics=True)
                for _, group in frame.iloc[start:end].groupby("Patient_ID", sort=False)
            ],
            ignore_index=True,
        )

    features = pd.concat(
        ordered_parallel_map(feature_batch, bounds, workers), ignore_index=True
    )
    if len(features) != len(frame) or features.duplicated(["Patient_ID", "ICULOS"]).any():
        raise PipelineError("Feature construction changed row identity")
    feature_columns = list(features.columns)
    base_model_columns = model_features(features, "baseline")
    c0_model_columns = model_features(features, "enhanced")
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
        "base_model_columns": base_model_columns,
        "base_model_column_hash": stable_hash(base_model_columns),
        "C0_model_columns": c0_model_columns,
        "C0_model_column_hash": stable_hash(c0_model_columns),
    }


def model_features(frame: pd.DataFrame, variant: str) -> list[str]:
    # ICULOS is an observed official predictor, not an identifier. SourceSet is
    # administrative provenance and is never a model input.
    if frame.columns.duplicated().any():
        raise PipelineError("Feature schema contains duplicate columns")
    excluded = {
        "Patient_ID", "SourceSet", "SepsisLabel", "TrueSepsisOnset_ICULOS",
        "OnsetReconstructionStatus", "Fold", onset.TARGET_COLUMN,
        onset.ELIGIBLE_COLUMN, onset.HOURS_TO_ONSET_COLUMN,
    }
    columns = [
        column for column in frame.columns
        if column not in excluded
        and not column.startswith("raw__")
        and not column.startswith(("causal_delta__", "causal_slope__", "koopman_innovation__"))
        and column not in {
            "koopman_energy", "koopman_energy_mean_8h", "koopman_energy_max_8h",
            "koopman_innovation_count",
        }
    ]
    enhanced_only = [column for column in columns if column.endswith("_cv_8h")]
    if variant == "baseline":
        columns = [column for column in columns if column not in enhanced_only]
    elif variant != "enhanced":
        raise PipelineError(f"Unknown model variant: {variant}")
    if not columns or "Hct_last_obs" not in columns:
        raise PipelineError("Feature policy failed: canonical Hct is not a model feature")
    return columns


def primary_model_features(frame: pd.DataFrame, representation: str) -> list[str]:
    """Return the fixed C0--C3 schema; SourceSet and target fields never enter."""
    if representation not in onset.REPRESENTATIONS:
        raise PipelineError(f"Unknown primary representation: {representation}")
    state = model_features(frame, "baseline")
    if representation == "C0":
        cv = [column for column in model_features(frame, "enhanced") if column.endswith("_cv_8h")]
        return state + [column for column in cv if column not in state]
    if representation == "C1":
        return state
    if representation == "C2":
        derived = [name for signal in DYNAMIC_COLUMNS for name in (onset.delta_column(signal), onset.slope_column(signal))]
    else:
        derived = [onset.innovation_column(signal) for signal in DYNAMIC_COLUMNS] + [
            "koopman_energy", "koopman_energy_mean_8h", "koopman_energy_max_8h",
            "koopman_innovation_count",
        ]
    missing = sorted(set(derived).difference(frame.columns))
    if missing:
        raise PipelineError(f"{representation} transformed schema is missing {missing}")
    return state + derived


def write_folds(features: pd.DataFrame, output: Path, n_splits: int = MODEL_POLICY["outer_folds"], split_seed: int = SEED) -> dict[str, Any]:
    labels = pd.Series(binary_array(features["SepsisLabel"], "Fold construction"), index=features.index)
    patient = labels.groupby(features["Patient_ID"], sort=True).max().rename("SepsisLabel").reset_index()
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
    folds = folds.copy()
    feature_labels = pd.Series(binary_array(features["SepsisLabel"], "Fold context"), index=features.index)
    expected_labels = feature_labels.groupby(features["Patient_ID"], sort=True).max()
    observed_labels = pd.to_numeric(folds.set_index("Patient_ID")["SepsisLabel"], errors="coerce").sort_index()
    fold_values = pd.to_numeric(folds["Fold"], errors="coerce")
    if observed_labels.isna().any() or not observed_labels.isin([0, 1]).all() or not observed_labels.astype(int).equals(expected_labels):
        raise PipelineError("Fold outcome provenance does not match the feature cohort")
    if not np.isfinite(fold_values).all() or not np.equal(fold_values, np.floor(fold_values)).all() or set(fold_values.astype(int)) != set(range(MODEL_POLICY["outer_folds"])):
        raise PipelineError("Fold assignments must be integers covering the exact outer-fold policy")
    folds["Fold"] = fold_values.astype(int)
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


def allocated_total_cpu_count() -> int:
    raw = os.environ.get("SLURM_CPUS_PER_TASK")
    maximum = int(MODEL_POLICY["maximum_total_cpu_threads"])
    if raw is None:
        return min(int(os.cpu_count() or 1), maximum)
    try:
        requested = int(raw)
    except ValueError as exc:
        raise PipelineError("SLURM_CPUS_PER_TASK must be an integer") from exc
    if requested < 1 or requested > maximum:
        raise PipelineError(f"Total allocated CPU count must be between 1 and {maximum}; got {requested}")
    return requested


def allocated_cpu_count() -> int:
    total = allocated_total_cpu_count()
    raw = os.environ.get("SEPSIS_FIT_THREADS")
    maximum = int(MODEL_POLICY["maximum_fit_threads"])
    if raw is None:
        return min(total, maximum)
    try:
        requested = int(raw)
    except ValueError as exc:
        raise PipelineError("SEPSIS_FIT_THREADS must be an integer") from exc
    if requested < 1 or requested > min(total, maximum):
        raise PipelineError(
            f"Per-fit CPU count must be between 1 and {min(total, maximum)}; got {requested}"
        )
    return requested


def parallel_model_workers(gpu: dict[str, Any], task_count: int) -> int:
    if task_count < 1:
        raise PipelineError("Parallel model scheduling requires at least one task")
    raw = os.environ.get("SEPSIS_PARALLEL_WORKERS", "1")
    try:
        requested = int(raw)
    except ValueError as exc:
        raise PipelineError("SEPSIS_PARALLEL_WORKERS must be an integer") from exc
    maximum = min(
        int(MODEL_POLICY["maximum_parallel_workers"]),
        task_count,
        allocated_total_cpu_count()
        // (allocated_cpu_count() * parallel_candidate_workers(gpu, len(MODEL_CANDIDATES))),
    )
    if gpu.get("available"):
        maximum = min(maximum, 1)
    if requested < 1 or requested > maximum:
        raise PipelineError(
            f"Parallel model workers must be between 1 and {maximum}; got {requested}"
        )
    return requested


def parallel_candidate_workers(gpu: dict[str, Any], task_count: int) -> int:
    if task_count < 1:
        raise PipelineError("Parallel candidate scheduling requires at least one task")
    raw = os.environ.get("SEPSIS_PARALLEL_CANDIDATES", "1")
    try:
        requested = int(raw)
    except ValueError as exc:
        raise PipelineError("SEPSIS_PARALLEL_CANDIDATES must be an integer") from exc
    maximum = min(
        int(MODEL_POLICY["maximum_parallel_candidates"]),
        task_count,
        allocated_total_cpu_count() // allocated_cpu_count(),
    )
    if gpu.get("available"):
        maximum = min(maximum, 1)
    if requested < 1 or requested > maximum:
        raise PipelineError(
            f"Parallel candidate workers must be between 1 and {maximum}; got {requested}"
        )
    return requested


def stage_worker_count(stage: str, task_count: int) -> int:
    if stage not in ("prepare", "finalize") or task_count < 1:
        raise PipelineError("Stage parallel scheduling requires prepare/finalize tasks")
    raw = os.environ.get("SEPSIS_STAGE_WORKERS", "1")
    try:
        requested = int(raw)
    except ValueError as exc:
        raise PipelineError("SEPSIS_STAGE_WORKERS must be an integer") from exc
    maximum = min(
        int(STAGE_POLICY[f"{stage}_workers"]),
        allocated_total_cpu_count(),
        task_count,
    )
    if requested < 1 or requested > maximum:
        raise PipelineError(
            f"{stage} workers must be between 1 and {maximum}; got {requested}"
        )
    return requested


def ordered_parallel_map(function: Any, items: Iterable[Any], workers: int) -> list[Any]:
    values = list(items)
    if workers == 1:
        return [function(value) for value in values]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(function, values))


def xgb_model(params: dict[str, Any], seed: int, gpu: dict[str, Any], n_estimators: int, early_stopping: bool = False):
    import xgboost as xgb

    kwargs: dict[str, Any] = {
        "n_estimators": int(n_estimators),
        "max_depth": params["max_depth"],
        "learning_rate": params["learning_rate"],
        "min_child_weight": params["min_child_weight"],
        "subsample": params["subsample"],
        "colsample_bytree": params["colsample_bytree"],
        "objective": MODEL_POLICY["xgboost_objective"],
        "eval_metric": MODEL_POLICY["xgboost_eval_metric"],  # XGBoost aucpr is not sklearn Average Precision.
        "random_state": int(seed),
        "n_jobs": allocated_cpu_count(),
        **xgb_backend(gpu),
    }
    if early_stopping:
        kwargs["early_stopping_rounds"] = MODEL_POLICY["early_stopping_rounds"]
    return xgb.XGBClassifier(**kwargs)


def fit_xgb(
    model: Any,
    train: pd.DataFrame,
    columns: list[str],
    validation: pd.DataFrame | None = None,
    target_column: str = "SepsisLabel",
    weight_policy: str = "equal_patient",
    train_matrix: np.ndarray | None = None,
    validation_matrix: np.ndarray | None = None,
) -> Any:
    train_matrix = matrix(train, columns) if train_matrix is None else train_matrix
    kwargs: dict[str, Any] = {
        "sample_weight": model_training_weights(train, target_column, weight_policy),
        "verbose": False,
    }
    if validation is not None:
        validation_matrix = (
            matrix(validation, columns)
            if validation_matrix is None
            else validation_matrix
        )
        kwargs.update({
            "eval_set": [(validation_matrix, validation[target_column])],
            "sample_weight_eval_set": [
                model_training_weights(validation, target_column, weight_policy)
            ],
        })
    elif validation_matrix is not None:
        raise PipelineError("A validation matrix requires validation rows")
    return model.fit(train_matrix, train[target_column], **kwargs)


def model_training_weights(
    frame: pd.DataFrame,
    target_column: str,
    policy: str,
) -> np.ndarray:
    """Predeclared training weights; evaluation always remains patient-balanced."""
    target = binary_array(frame[target_column], f"Training weights {policy}")
    if policy == "equal_row":
        return np.ones(len(frame), dtype=float)
    weights = equal_patient_weights(frame)
    if policy == "equal_patient":
        return weights
    if policy != "equal_patient_then_row_class":
        raise PipelineError(f"Unknown training-weight policy: {policy}")
    class_mass = np.bincount(target, weights=weights, minlength=2)
    if len(class_mass) != 2 or (class_mass <= 0).any():
        raise PipelineError("Class-balanced training weights require both outcome classes")
    return weights / (2.0 * class_mass[target])


PRIMARY_OOF_COLUMNS = [
    "Patient_ID", "SourceSet", "ICULOS", "Age", "SepsisLabel",
    "TrueSepsisOnset_ICULOS", "OnsetReconstructionStatus",
    onset.TARGET_COLUMN, onset.ELIGIBLE_COLUMN, onset.HOURS_TO_ONSET_COLUMN,
    "Fold", "prob_raw", "prob_calibrated", "nested_alarm_threshold",
]


def attach_prediction_columns(
    frame: pd.DataFrame,
    probabilities: dict[str, Iterable[float]],
    threshold_column: str,
    threshold: float,
) -> pd.DataFrame:
    """Append prediction columns in one block so wide feature frames stay usable."""
    if not probabilities or threshold_column in probabilities:
        raise PipelineError("Prediction columns require probabilities and one distinct threshold")
    columns = [*probabilities, threshold_column]
    if len(columns) != len(set(columns)) or any(column in frame.columns for column in columns):
        raise PipelineError("Prediction columns already exist or are duplicated")
    validated = {
        name: probability_array(values, f"Prediction column {name}")
        for name, values in probabilities.items()
    }
    if any(len(values) != len(frame) for values in validated.values()):
        raise PipelineError("Prediction length does not match the feature frame")
    threshold = float(threshold)
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise PipelineError("Alarm threshold must be finite and in [0,1]")
    predictions = pd.DataFrame(
        {**validated, threshold_column: threshold}, index=frame.index
    )
    return pd.concat([frame, predictions], axis=1)


def patient_mask(frame: pd.DataFrame, patients: Iterable[str]) -> np.ndarray:
    return frame["Patient_ID"].isin(set(patients)).to_numpy()


def primary_patient_splits(
    frame: pd.DataFrame,
    n_splits: int = MODEL_POLICY["inner_folds"],
    seed_offset: int = 0,
    split_seed: int = SEED,
) -> tuple[pd.DataFrame, list[tuple[np.ndarray, np.ndarray]]]:
    decisions = onset.primary_decisions(frame)
    patient = decisions.groupby("Patient_ID", sort=True)[onset.TARGET_COLUMN].max().astype(int).reset_index()
    if patient[onset.TARGET_COLUMN].value_counts().min() < n_splits:
        raise PipelineError("Insufficient patient class count for direct-onset inner folds")
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=split_seed + seed_offset)
    return patient, list(splitter.split(patient, patient[onset.TARGET_COLUMN], groups=patient["Patient_ID"]))


def fit_primary_representation(train: pd.DataFrame, representation: str, lift: str = "identity") -> dict[str, Any]:
    if representation not in onset.REPRESENTATIONS:
        raise PipelineError(f"Unknown primary representation: {representation}")
    if representation in {"C0", "C1"}:
        return {"representation": representation, "lift": "not_applicable", "selected_signals": (), "koopman": None}
    primary_population = train.loc[
        train["OnsetReconstructionStatus"] != "septic_onset_left_censored"
    ].copy()
    if primary_population.empty:
        raise PipelineError("Primary representation has no non-left-censored training patients")
    try:
        selected = onset.select_dynamic_signals(
            primary_population,
            DYNAMIC_COLUMNS,
            onset.KOOPMAN_POLICY["minimum_observed_row_fraction"],
            onset.KOOPMAN_POLICY["minimum_patients_with_two_observations"],
            onset.KOOPMAN_POLICY["maximum_signals"],
        )
        if not selected:
            raise onset.OnsetKoopmanError("No official dynamic signal satisfies fold-local support")
        fitted = None
        if representation == "C3":
            fitted = onset.fit_koopman(
                primary_population, DYNAMIC_COLUMNS, lift, selected_signals=selected
            )
    except onset.OnsetKoopmanError as exc:
        raise PipelineError(str(exc)) from exc
    return {"representation": representation, "lift": lift if representation == "C3" else "not_applicable", "selected_signals": selected, "koopman": fitted}


def transform_primary_representation(frame: pd.DataFrame, fitted: dict[str, Any]) -> pd.DataFrame:
    representation = fitted["representation"]
    if representation in {"C0", "C1"}:
        return frame.copy()
    try:
        derived = (
            onset.transform_deltas(frame, DYNAMIC_COLUMNS, fitted["selected_signals"])
            if representation == "C2"
            else onset.transform_koopman(frame, fitted["koopman"])
        )
    except onset.OnsetKoopmanError as exc:
        raise PipelineError(str(exc)) from exc
    return pd.concat([frame, derived], axis=1)


def select_inner_primary_model(
    train: pd.DataFrame,
    representation: str,
    gpu: dict[str, Any],
    outer_fold: int,
    split_seed: int = SEED,
) -> tuple[dict[str, Any], str, int, pd.DataFrame, list[dict[str, Any]]]:
    """Select representation/XGBoost and emit inner OOF using inner-train fits only."""
    try:
        patient, splits = primary_patient_splits(train, seed_offset=outer_fold + 1, split_seed=split_seed)
    except onset.OnsetKoopmanError as exc:
        raise PipelineError(str(exc)) from exc
    lifts = onset.KOOPMAN_POLICY["lifts"] if representation == "C3" else ("identity",)
    candidates = [(lift, candidate) for lift in lifts for candidate in MODEL_CANDIDATES]
    scores: dict[tuple[str, str], list[float]] = {(lift, candidate["id"]): [] for lift, candidate in candidates}
    rounds: dict[tuple[str, str], list[int]] = {(lift, candidate["id"]): [] for lift, candidate in candidates}
    validation_rows: dict[tuple[str, int], pd.DataFrame] = {}
    candidate_probabilities: dict[tuple[str, str], list[np.ndarray]] = {
        (lift, candidate["id"]): [] for lift, candidate in candidates
    }
    selection_detail: list[dict[str, Any]] = []
    for lift_index, lift in enumerate(lifts):
        for inner_fold, (fit_idx, valid_idx) in enumerate(splits):
            fit_patients = patient.iloc[fit_idx]["Patient_ID"]
            valid_patients = patient.iloc[valid_idx]["Patient_ID"]
            fit_full = train.loc[patient_mask(train, fit_patients)].copy()
            valid_full = train.loc[patient_mask(train, valid_patients)].copy()
            representation_fit = fit_primary_representation(fit_full, representation, lift)
            fit_transformed = transform_primary_representation(fit_full, representation_fit)
            valid_transformed = transform_primary_representation(valid_full, representation_fit)
            fit = onset.primary_decisions(fit_transformed)
            valid = onset.primary_decisions(valid_transformed)
            validation_rows[(lift, inner_fold)] = valid[[
                "Patient_ID", "SourceSet", "ICULOS", "SepsisLabel",
                "TrueSepsisOnset_ICULOS", "OnsetReconstructionStatus",
                onset.TARGET_COLUMN, onset.ELIGIBLE_COLUMN,
            ]].copy()
            columns = primary_model_features(fit_transformed, representation)
            fit_matrix = matrix(fit, columns)
            valid_matrix = matrix(valid, columns)
            koopman_fit = representation_fit["koopman"]
            transition_counts = (
                koopman_fit.training_transition_counts if koopman_fit is not None else {}
            )
            def fit_candidate(item: tuple[int, dict[str, Any]]):
                candidate_index, candidate = item
                model = xgb_model(
                    candidate,
                    split_seed + outer_fold * 1000 + lift_index * 100 + candidate_index * 10 + inner_fold,
                    gpu,
                    MODEL_POLICY["early_stopping_max_estimators"],
                    early_stopping=True,
                )
                fit_xgb(
                    model,
                    fit,
                    columns,
                    valid,
                    target_column=onset.TARGET_COLUMN,
                    train_matrix=fit_matrix,
                    validation_matrix=valid_matrix,
                )
                valid_scored = valid.copy()
                probability = model.predict_proba(valid_matrix)[:, 1]
                valid_scored["probability"] = probability
                score = onset.patient_balanced_average_precision(valid_scored, "probability")
                best_round = int(getattr(model, "best_iteration", model.n_estimators - 1)) + 1
                return candidate, probability, score, best_round

            candidate_tasks = list(enumerate(MODEL_CANDIDATES))
            for candidate, probability, score, best_round in ordered_parallel_map(
                fit_candidate,
                candidate_tasks,
                parallel_candidate_workers(gpu, len(candidate_tasks)),
            ):
                scores[(lift, candidate["id"])].append(score)
                rounds[(lift, candidate["id"])].append(best_round)
                candidate_probabilities[(lift, candidate["id"])].append(probability)
                selection_detail.append({
                    "lift": lift,
                    "candidate": candidate["id"],
                    "inner_fold": inner_fold,
                    "patient_balanced_average_precision": score,
                    "best_round": best_round,
                    "selected_signals": json.dumps(list(representation_fit["selected_signals"])),
                    "koopman_training_transition_counts": json.dumps(transition_counts, sort_keys=True),
                    "fit_patient_hash": stable_hash(sorted(fit_patients.astype(str))),
                    "valid_patient_hash": stable_hash(sorted(valid_patients.astype(str))),
                })
    winner_lift, winner_id = max(
        scores,
        key=lambda key: (
            float(np.mean(scores[key])),
            -next(candidate["max_depth"] for candidate in MODEL_CANDIDATES if candidate["id"] == key[1]),
            key[0] == "identity",
        ),
    )
    winner = next(candidate for candidate in MODEL_CANDIDATES if candidate["id"] == winner_id)
    selected_rounds = max(1, int(round(float(np.median(rounds[(winner_lift, winner_id)])))))
    inner_rows: list[pd.DataFrame] = []
    for inner_fold, probability in enumerate(
        candidate_probabilities[(winner_lift, winner_id)]
    ):
        valid = validation_rows[(winner_lift, inner_fold)].copy()
        valid["prob_raw"] = probability
        valid["InnerFold"] = inner_fold
        inner_rows.append(valid[[
            "Patient_ID", "SourceSet", "ICULOS", "SepsisLabel",
            "TrueSepsisOnset_ICULOS", "OnsetReconstructionStatus",
            onset.TARGET_COLUMN, onset.ELIGIBLE_COLUMN, "prob_raw", "InnerFold",
        ]])
    inner_oof = pd.concat(inner_rows, ignore_index=True).sort_values(["Patient_ID", "ICULOS"], kind="mergesort")
    expected = onset.primary_decisions(train)
    if len(inner_oof) != len(expected) or inner_oof.duplicated(["Patient_ID", "ICULOS"]).any():
        raise PipelineError("Direct-onset inner OOF does not cover eligible outer-train rows exactly once")
    return winner, winner_lift, selected_rounds, inner_oof, selection_detail


def challenge_utility(frame: pd.DataFrame, probability_column: str, threshold: float) -> float:
    """Use the official scorer's sole utility definition over source-qualified patients."""
    required = {"Patient_ID", "ICULOS", "SepsisLabel", probability_column}
    if missing := required.difference(frame.columns):
        raise PipelineError(f"Utility requires {sorted(missing)}")
    try:
        threshold = float(threshold)
    except (TypeError, ValueError) as exc:
        raise PipelineError("Utility threshold must be a finite scalar in [0,1]") from exc
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise PipelineError("Utility threshold must be a finite scalar in [0,1]")
    probability_array(frame[probability_column], "Utility")
    observed_total = 0.0
    best_total = 0.0
    inaction_total = 0.0
    for _, patient in frame.sort_values(["Patient_ID", "ICULOS"], kind="mergesort").groupby("Patient_ID", sort=False):
        labels, _ = longitudinal_patient_arrays(patient, "Utility")
        predictions = (patient[probability_column].to_numpy(dtype=float) >= threshold).astype(int)
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


def probability_array(values: Any, context: str) -> np.ndarray:
    try:
        probability = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise PipelineError(f"{context} probabilities must be numeric") from exc
    if probability.ndim != 1 or probability.size == 0 or not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise PipelineError(f"{context} probabilities must be a finite one-dimensional array in [0,1]")
    return probability


def binary_array(values: Any, context: str) -> np.ndarray:
    try:
        raw = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise PipelineError(f"{context} outcomes must be numeric") from exc
    if raw.ndim != 1 or raw.size == 0 or not np.isfinite(raw).all() or not np.isin(raw, [0, 1]).all():
        raise PipelineError(f"{context} outcomes must be a finite nonempty binary array")
    return raw.astype(int)


def longitudinal_patient_arrays(patient: pd.DataFrame, context: str) -> tuple[np.ndarray, np.ndarray]:
    labels = binary_array(patient["SepsisLabel"], context)
    try:
        times = np.asarray(patient["ICULOS"], dtype=float)
    except (TypeError, ValueError) as exc:
        raise PipelineError(f"{context} times must be numeric") from exc
    if not np.isfinite(times).all() or (np.diff(times) <= 0).any() or (np.diff(labels) < 0).any():
        raise PipelineError(f"{context} requires strictly increasing time and persistent Challenge labels")
    return labels, times


def calibration_metrics(y: np.ndarray, probability: np.ndarray, sample_weight: np.ndarray | None = None) -> dict[str, float]:
    probability = probability_array(probability, "Calibration")
    y = binary_array(y, "Calibration")
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
    logit_probability = np.clip(probability, 1e-6, 1 - 1e-6)
    logit = np.log(logit_probability / (1 - logit_probability))
    logit_mean = float(np.average(logit, weights=weights))
    logit_scale = float(np.sqrt(np.average((logit - logit_mean) ** 2, weights=weights)))
    if not np.isfinite(logit_scale) or logit_scale <= np.finfo(float).eps:
        raise PipelineError("Calibration slope is not identifiable from a constant logit")
    standardized_logit = ((logit - logit_mean) / logit_scale).reshape(-1, 1)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            model = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000).fit(
                standardized_logit, y, sample_weight=weights
            )
    except ConvergenceWarning as exc:
        raise PipelineError("Unpenalized calibration regression did not converge after weighted logit standardization") from exc
    calibration_slope = float(model.coef_[0, 0] / logit_scale)
    calibration_intercept = float(model.intercept_[0] - model.coef_[0, 0] * logit_mean / logit_scale)
    calibration_in_the_large = 0.0
    for _ in range(30):
        fitted = 1 / (1 + np.exp(-np.clip(logit + calibration_in_the_large, -50, 50)))
        information = float(np.average(fitted * (1 - fitted), weights=weights))
        if information <= np.finfo(float).eps:
            raise PipelineError("Calibration-in-the-large intercept is not identifiable")
        step = float(np.average(y - fitted, weights=weights) / information)
        calibration_in_the_large += step
        if abs(step) < 1e-10:
            break
    else:
        raise PipelineError("Calibration-in-the-large intercept did not converge")
    return {
        "brier": float(brier_score_loss(y, probability, sample_weight=weights)),
        "ece_fixed_10_bins": float(ece),
        "calibration_in_the_large_intercept_slope_fixed_1": calibration_in_the_large,
        "calibration_intercept_with_slope": calibration_intercept,
        "calibration_slope": calibration_slope,
    }


def calibration_metrics_with_patient_uncertainty(
    frame: pd.DataFrame,
    probability_column: str,
    repeats: int | None = None,
    target_column: str = "SepsisLabel",
    patient_balanced: bool = False,
) -> dict[str, Any]:
    y = binary_array(frame[target_column], "Calibration uncertainty")
    probability = probability_array(frame[probability_column], "Calibration uncertainty")
    base_weight = equal_patient_weights(frame) if patient_balanced else np.ones(len(frame), dtype=float)
    point = calibration_metrics(y, probability, sample_weight=base_weight)
    repeats = FEATURE_POLICY["calibration_patient_cluster_bootstrap_repeats"] if repeats is None else repeats
    codes, patients = pd.factorize(frame["Patient_ID"], sort=True)
    if repeats < 1 or (codes < 0).any() or len(patients) < 2:
        raise PipelineError("Invalid patient-cluster calibration bootstrap context")
    rng = np.random.default_rng(SEED)
    samples = {name: [] for name in point}
    for _ in range(repeats):
        weights = base_weight * rng.multinomial(len(patients), np.full(len(patients), 1 / len(patients)))[codes].astype(float)
        if set(y[weights > 0]) != {0, 1}:
            raise PipelineError("Patient-cluster calibration bootstrap draw lacks an outcome class")
        for name, value in calibration_metrics(y, probability, sample_weight=weights).items():
            samples[name].append(value)
    result = {**point, "calibration_estimand_unit": "observed row-time", "point_estimate_weighting": "equal total weight per patient" if patient_balanced else "each observed row-time equally weighted", "uncertainty_method": "patient-cluster bootstrap percentile 95% CI", "uncertainty_unit": "patient; stated point weighting preserved within resampled patients", "uncertainty_repeats": int(repeats), "uncertainty_confidence_level": 0.95}
    for name, values in samples.items():
        result[f"{name}_ci_95_low"] = float(np.quantile(values, 0.025))
        result[f"{name}_ci_95_high"] = float(np.quantile(values, 0.975))
    return result


def discrimination_metrics(y: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    probability = probability_array(probability, "Discrimination")
    y = binary_array(y, "Discrimination")
    if set(y) != {0, 1}:
        raise PipelineError("Discrimination metrics require both classes")
    precision, recall, _ = precision_recall_curve(y, probability)
    return {
        "auroc": float(roc_auc_score(y, probability)),
        "average_precision": float(average_precision_score(y, probability)),
        "trapezoidal_pr_auc": float(auc(recall[::-1], precision[::-1])),
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


def primary_outer_oof(
    features: pd.DataFrame,
    folds: pd.DataFrame,
    representation: str,
    output_dir: Path,
    gpu: dict[str, Any],
    split_seed: int = SEED,
    persist_oof: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fully nested C0--C3 OOF predictions for true onset in 1--6 hours."""
    merged = require_fold_context(features, folds).sort_values(["Patient_ID", "ICULOS"], kind="mergesort").reset_index(drop=True)
    records: list[pd.DataFrame] = []
    robustness_records: list[pd.DataFrame] = []
    selection_rows: list[dict[str, Any]] = []
    inner_detail_rows: list[dict[str, Any]] = []

    def fit_outer_fold(outer_fold: int) -> tuple[
        pd.DataFrame, pd.DataFrame | None, dict[str, Any], list[dict[str, Any]]
    ]:
        outer_train = merged.loc[merged["Fold"] != outer_fold].copy()
        outer_test = merged.loc[merged["Fold"] == outer_fold].copy()
        if set(outer_train["Patient_ID"]) & set(outer_test["Patient_ID"]):
            raise PipelineError("Direct-onset outer-fold patient overlap")
        candidate, lift, rounds, inner_oof, detail = select_inner_primary_model(
            outer_train, representation, gpu, int(outer_fold), split_seed
        )
        calibration = onset.fit_calibration_policy(inner_oof, "prob_raw")
        inner_oof["prob_calibrated"] = onset.apply_calibration(calibration, inner_oof["prob_raw"])
        threshold, inner_alarm = onset.select_alarm_threshold(inner_oof, "prob_calibrated")
        representation_fit = fit_primary_representation(outer_train, representation, lift)
        koopman_fit = representation_fit["koopman"]
        transformed_train = transform_primary_representation(outer_train, representation_fit)
        transformed_test = transform_primary_representation(outer_test, representation_fit)
        train_decisions = onset.primary_decisions(transformed_train)
        columns = primary_model_features(transformed_train, representation)
        train_matrix = matrix(train_decisions, columns)
        test_matrix = matrix(transformed_test, columns)
        model = xgb_model(candidate, split_seed + int(outer_fold), gpu, rounds)
        fit_xgb(
            model,
            train_decisions,
            columns,
            target_column=onset.TARGET_COLUMN,
            train_matrix=train_matrix,
        )
        raw_probability = model.predict_proba(test_matrix)[:, 1]
        calibrated_probability = onset.apply_calibration(calibration, raw_probability)
        transformed_test = attach_prediction_columns(
            transformed_test,
            {"prob_raw": raw_probability, "prob_calibrated": calibrated_probability},
            "nested_alarm_threshold",
            threshold,
        )
        robustness = None
        if representation in ROBUSTNESS_POLICY["representations"]:
            test_decisions = onset.primary_decisions(transformed_test).sort_values(
                ["Patient_ID", "ICULOS"], kind="mergesort"
            ).reset_index(drop=True)
            robustness_matrix = matrix(test_decisions, columns)
            robustness = test_decisions[ROBUSTNESS_IDENTITY_COLUMNS].copy()
            primary_seed = ROBUSTNESS_POLICY["training_seed_bases"][0]
            configurations = robustness_configurations()

            def fit_robustness(configuration: tuple[int, str, str]):
                seed_base, balance_policy, _ = configuration
                if seed_base == primary_seed and balance_policy == "equal_patient":
                    probability = probability_array(
                        test_decisions["prob_raw"], "Primary robustness OOF"
                    )
                else:
                    sensitivity_model = xgb_model(
                        candidate, seed_base + int(outer_fold), gpu, rounds
                    )
                    fit_xgb(
                        sensitivity_model,
                        train_decisions,
                        columns,
                        target_column=onset.TARGET_COLUMN,
                        weight_policy=balance_policy,
                        train_matrix=train_matrix,
                    )
                    probability = sensitivity_model.predict_proba(robustness_matrix)[:, 1]
                return seed_base, balance_policy, probability

            for seed_base, balance_policy, probability in ordered_parallel_map(
                fit_robustness,
                configurations,
                parallel_candidate_workers(gpu, len(configurations)),
            ):
                robustness[robustness_probability_column(
                    representation, seed_base, balance_policy
                )] = probability_array(
                    probability, "Robustness raw probability"
                ).astype(np.float32)
        selection = {
            "representation": representation,
            "outer_fold": int(outer_fold),
            "selected_candidate": candidate["id"],
            "selected_hyperparameters": json.dumps(candidate, sort_keys=True),
            "selected_lift": lift if representation == "C3" else "not_applicable",
            "selected_tree_count_from_inner_only": rounds,
            "selected_signals_outer_train": json.dumps(list(representation_fit["selected_signals"])),
            "selected_signals_hash": stable_hash(list(representation_fit["selected_signals"])),
            "koopman_training_transition_counts": json.dumps(
                koopman_fit.training_transition_counts if koopman_fit is not None else {},
                sort_keys=True,
            ),
            "feature_count": len(columns),
            "feature_column_hash": stable_hash(columns),
            "calibrator_selected_by_inner_oof_brier": calibration.method,
            "inner_oof_identity_brier": calibration.identity_brier,
            "inner_oof_logistic_brier": calibration.logistic_brier,
            "nested_alarm_threshold_from_inner_oof_only": threshold,
            "inner_oof_alarm_budget": inner_alarm["false_alarm_episodes_per_patient_day"],
            "inner_oof_useful_sensitivity": inner_alarm["useful_sensitivity"],
            "outer_train_patient_hash": stable_hash(sorted(outer_train["Patient_ID"].unique())),
            "outer_test_patient_hash": stable_hash(sorted(outer_test["Patient_ID"].unique())),
            "outer_train_patient_count": int(outer_train["Patient_ID"].nunique()),
            "outer_test_patient_count": int(outer_test["Patient_ID"].nunique()),
        }
        inner_detail = [
            {"representation": representation, "outer_fold": int(outer_fold), **row}
            for row in detail
        ]
        return transformed_test[PRIMARY_OOF_COLUMNS], robustness, selection, inner_detail

    outer_folds = [int(value) for value in sorted(merged["Fold"].unique())]
    workers = parallel_model_workers(gpu, len(outer_folds))
    for record, robustness, selection, inner_detail in ordered_parallel_map(
        fit_outer_fold, outer_folds, workers
    ):
        records.append(record)
        if robustness is not None:
            robustness_records.append(robustness)
        selection_rows.append(selection)
        inner_detail_rows.extend(inner_detail)
    oof = pd.concat(records, ignore_index=True).sort_values(["Patient_ID", "ICULOS"], kind="mergesort").reset_index(drop=True)
    if len(oof) != len(merged) or oof.duplicated(["Patient_ID", "ICULOS"]).any():
        raise PipelineError("Direct-onset outer OOF does not contain every source row exactly once")
    selection_path = output_dir / f"{representation}_nested_selection.csv"
    inner_detail_path = output_dir / f"{representation}_inner_selection.csv"
    atomic_csv(pd.DataFrame(selection_rows), selection_path)
    atomic_csv(pd.DataFrame(inner_detail_rows), inner_detail_path)
    oof_path = output_dir / f"{representation}_oof_predictions.csv"
    if persist_oof:
        atomic_csv(oof, oof_path)
        oof = pd.read_csv(oof_path)
    robustness_oof = None
    if representation in ROBUSTNESS_POLICY["representations"]:
        robustness_oof = pd.concat(robustness_records, ignore_index=True).sort_values(
            ["Patient_ID", "ICULOS"], kind="mergesort"
        ).reset_index(drop=True)
        expected_columns = ROBUSTNESS_IDENTITY_COLUMNS + [
            robustness_probability_column(representation, seed, balance)
            for seed, balance, _ in robustness_configurations()
        ]
        if (
            len(robustness_oof) != len(onset.primary_decisions(merged))
            or list(robustness_oof.columns) != expected_columns
            or robustness_oof.duplicated(["Patient_ID", "ICULOS"]).any()
        ):
            raise PipelineError("Robustness OOF is incomplete or duplicated")
    return oof, {
        "representation": representation,
        "outcome_estimand": "true reconstructed onset in 1--6 hours",
        "oof_artifact": oof_path.name if persist_oof else None,
        "oof_sha256": sha256_file(oof_path) if persist_oof else None,
        "selection_artifact": selection_path.name,
        "selection_sha256": sha256_file(selection_path),
        "inner_selection_artifact": inner_detail_path.name,
        "inner_selection_sha256": sha256_file(inner_detail_path),
        "robustness_oof": robustness_oof,
    }


def robustness_configurations() -> tuple[tuple[int, str, str], ...]:
    """Prespecified sensitivity configurations; none may replace the primary result."""
    primary_seed = ROBUSTNESS_POLICY["training_seed_bases"][0]
    return tuple(
        (seed, "equal_patient", "seed_stability")
        for seed in ROBUSTNESS_POLICY["training_seed_bases"]
    ) + tuple(
        (primary_seed, policy, "balance_sensitivity")
        for policy in ROBUSTNESS_POLICY["balance_policies"]
        if policy != "equal_patient"
    )


ROBUSTNESS_IDENTITY_COLUMNS = [
    "Patient_ID", "SourceSet", "ICULOS", "Fold", onset.TARGET_COLUMN,
]


def robustness_probability_column(
    representation: str,
    seed_base: int,
    balance_policy: str,
) -> str:
    return (
        f"prob_raw__{representation}__seed_{seed_base}__{balance_policy}"
    )


def combine_robustness_oof(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Combine paired C0/C3 probabilities without repeating row identities."""
    if set(frames) != set(ROBUSTNESS_POLICY["representations"]):
        raise PipelineError("Robustness OOF representations are incomplete")
    base = frames[ROBUSTNESS_POLICY["representations"][0]].copy()
    for representation in ROBUSTNESS_POLICY["representations"][1:]:
        frame = frames[representation]
        try:
            pd.testing.assert_frame_equal(
                base[ROBUSTNESS_IDENTITY_COLUMNS],
                frame[ROBUSTNESS_IDENTITY_COLUMNS],
                check_dtype=False,
            )
        except AssertionError as exc:
            raise PipelineError("Robustness representation identities are not paired") from exc
        for seed, balance, _ in robustness_configurations():
            column = robustness_probability_column(representation, seed, balance)
            base[column] = frame[column].to_numpy(dtype=np.float32)
    return base


def robustness_summary(robustness_oof: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize every prespecified seed/balance result without winner selection."""
    rows = []
    keys = ["representation", "training_seed_base", "balance_policy", "analysis_family"]
    target = binary_array(robustness_oof[onset.TARGET_COLUMN], "Robustness summary")
    weights = onset.equal_patient_weights(robustness_oof)
    for representation in ROBUSTNESS_POLICY["representations"]:
        for seed, balance, family in robustness_configurations():
            probability = probability_array(
                robustness_oof[
                    robustness_probability_column(representation, seed, balance)
                ],
                "Robustness summary",
            )
            rows.append({
                **dict(zip(keys, (representation, seed, balance, family))),
                "n_decision_hours": int(len(robustness_oof)),
                "n_patients": int(robustness_oof["Patient_ID"].nunique()),
                "patient_balanced_average_precision": float(
                    average_precision_score(target, probability, sample_weight=weights)
                ),
                "patient_balanced_auroc": float(
                    roc_auc_score(target, probability, sample_weight=weights)
                ),
                "patient_balanced_brier_uncalibrated": float(
                    brier_score_loss(target, probability, sample_weight=weights)
                ),
                "configuration_selected": False,
            })
    summary = pd.DataFrame(rows).sort_values(keys, kind="mergesort").reset_index(drop=True)
    comparator = summary.loc[summary["representation"] == "C0"].drop(
        columns="representation"
    )
    candidate = summary.loc[summary["representation"] == "C3"].drop(
        columns="representation"
    )
    paired = comparator.merge(
        candidate,
        on=["training_seed_base", "balance_policy", "analysis_family"],
        suffixes=("_C0", "_C3"),
        validate="one_to_one",
    )
    for metric in (
        "patient_balanced_average_precision",
        "patient_balanced_auroc",
        "patient_balanced_brier_uncalibrated",
    ):
        paired[f"C3_minus_C0_{metric}"] = paired[f"{metric}_C3"] - paired[f"{metric}_C0"]
    paired["configuration_selected"] = False
    return summary, paired


def representation_ablation_summary(
    summaries: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    """Data-driven C0--C3 ablation table; it never selects a representation."""
    definitions = {
        "C0": "baseline plus causal CV",
        "C1": "40-predictor causal state",
        "C2": "C1 plus deltas and slopes",
        "C3": "C1 plus Koopman innovations",
    }
    rows = []
    for representation in onset.REPRESENTATIONS:
        summary = summaries[representation]
        alarm = summary["alarm_policy"]
        rows.append({
            "representation": representation,
            "ablation_definition": definitions[representation],
            "patient_balanced_average_precision": summary[
                "patient_balanced_average_precision"
            ],
            "patient_balanced_brier": summary["patient_balanced_brier"],
            "useful_sensitivity": alarm["useful_sensitivity"],
            "false_alarm_episodes_per_patient_day": alarm[
                "false_alarm_episodes_per_patient_day"
            ],
            "delta_average_precision_vs_C0": summary[
                "patient_balanced_average_precision"
            ] - summaries["C0"]["patient_balanced_average_precision"],
            "configuration_selected": False,
        })
    return pd.DataFrame(rows)


def master_result_table(
    summaries: dict[str, dict[str, Any]],
    transport: pd.DataFrame,
    robustness: pd.DataFrame,
) -> pd.DataFrame:
    """One data-driven inventory of every executed current-run model result."""
    rows = []
    for representation in onset.REPRESENTATIONS:
        summary = summaries[representation]
        alarm = summary["alarm_policy"]
        rows.append({
            "analysis_family": "primary_nested_internal",
            "experiment": "mixed_A_B_patient_grouped_outer_CV",
            "representation": representation,
            "probability_state": "nested_calibrated",
            "training_seed_base": SEED,
            "balance_policy": "equal_patient",
            "patient_balanced_average_precision": summary[
                "patient_balanced_average_precision"
            ],
            "patient_balanced_auroc": summary["patient_balanced_auroc"],
            "patient_balanced_brier": summary["patient_balanced_brier"],
            "useful_sensitivity": alarm["useful_sensitivity"],
            "false_alarm_episodes_per_patient_day": alarm[
                "false_alarm_episodes_per_patient_day"
            ],
            "median_lead_time_hours": alarm["median_lead_time_hours"],
            "configuration_selected": False,
        })
    for result in transport.to_dict("records"):
        rows.append({
            "analysis_family": "source_transport",
            "experiment": result["experiment"],
            "representation": result["representation"],
            "probability_state": "train_source_nested_calibrated",
            "training_seed_base": SEED,
            "balance_policy": "equal_patient",
            "patient_balanced_average_precision": result[
                "patient_balanced_average_precision"
            ],
            "patient_balanced_auroc": result["patient_balanced_auroc"],
            "patient_balanced_brier": result["patient_balanced_brier"],
            "useful_sensitivity": result["useful_sensitivity"],
            "false_alarm_episodes_per_patient_day": result[
                "false_alarm_episodes_per_patient_day"
            ],
            "median_lead_time_hours": result["median_lead_time_hours"],
            "configuration_selected": False,
        })
    for result in robustness.to_dict("records"):
        rows.append({
            "analysis_family": result["analysis_family"],
            "experiment": (
                f"seed_{result['training_seed_base']}_balance_{result['balance_policy']}"
            ),
            "representation": result["representation"],
            "probability_state": "raw_uncalibrated_sensitivity_only",
            "training_seed_base": result["training_seed_base"],
            "balance_policy": result["balance_policy"],
            "patient_balanced_average_precision": result[
                "patient_balanced_average_precision"
            ],
            "patient_balanced_auroc": result["patient_balanced_auroc"],
            "patient_balanced_brier": result[
                "patient_balanced_brier_uncalibrated"
            ],
            "useful_sensitivity": math.nan,
            "false_alarm_episodes_per_patient_day": math.nan,
            "median_lead_time_hours": math.nan,
            "configuration_selected": False,
        })
    return pd.DataFrame(rows)


def primary_model_summary(oof: pd.DataFrame, representation: str, output_dir: Path) -> dict[str, Any]:
    decisions = onset.primary_decisions(oof)
    target = binary_array(decisions[onset.TARGET_COLUMN], f"{representation} direct-onset summary")
    probability = probability_array(decisions["prob_calibrated"], f"{representation} direct-onset summary")
    weights = onset.equal_patient_weights(decisions)
    alarm = onset.alarm_metrics(oof, "prob_calibrated", "nested_alarm_threshold")
    reliability_name = f"{representation}_reliability.csv"
    alarm_events_name = f"{representation}_alarm_events.csv"
    atomic_csv(
        pd.DataFrame(onset.primary_reliability_rows(
            oof, "prob_calibrated", representation, FEATURE_POLICY["ece_equal_width_bins"]
        )),
        output_dir / reliability_name,
    )
    atomic_csv(
        pd.DataFrame(onset.alarm_event_rows(
            oof, "prob_calibrated", "nested_alarm_threshold"
        )),
        output_dir / alarm_events_name,
    )
    summary = {
        "representation": representation,
        "outcome_estimand": "Y(i,t)=1 iff 1 <= true_onset(i)-t <= 6",
        "eligibility": onset.TARGET_POLICY,
        "n_rows_source": int(len(oof)),
        "n_primary_decision_hours": int(len(decisions)),
        "n_patients": int(decisions["Patient_ID"].nunique()),
        "n_positive_decision_hours": int(target.sum()),
        "patient_balanced_average_precision": float(average_precision_score(target, probability, sample_weight=weights)),
        "patient_balanced_auroc": float(roc_auc_score(target, probability, sample_weight=weights)),
        "patient_balanced_brier": float(brier_score_loss(target, probability, sample_weight=weights)),
        "reliability_artifact": reliability_name,
        "calibration": calibration_metrics_with_patient_uncertainty(
            decisions,
            "prob_calibrated",
            target_column=onset.TARGET_COLUMN,
            patient_balanced=True,
        ),
        "alarm_policy": {
            "window": "onset-6h through onset-1h",
            "refractory_hours": onset.ALARM_POLICY["refractory_hours"],
            "threshold_provenance": "fold-specific outer-train inner OOF only",
            "event_artifact": alarm_events_name,
            "n_patients_with_predictions": alarm["n_patients_with_predictions"],
            "n_monitored_patients": alarm["n_monitored_patients"],
            "n_alarm_episodes": alarm["n_alarm_episodes"],
            "n_onset_eligible_septic_patients": alarm["n_onset_eligible_septic_patients"],
            "tp_patients": alarm["tp_patients"],
            "fn_patients": alarm["fn_patients"],
            "false_alarm_episodes": alarm["false_alarm_episodes"],
            "false_alarm_episodes_per_patient_day": alarm["false_alarm_episodes_per_patient_day"],
            "repeated_alarm_episodes": alarm["repeated_alarm_episodes"],
            "late_pre_onset_alarm_episodes": alarm["late_pre_onset_alarm_episodes"],
            "post_onset_alarm_episodes": alarm["post_onset_alarm_episodes"],
            "right_censored_alarm_episodes": alarm["right_censored_alarm_episodes"],
            "left_censored_unclassified_alarm_episodes": alarm[
                "left_censored_unclassified_alarm_episodes"
            ],
            "useful_sensitivity": alarm["useful_sensitivity"],
            "median_lead_time_hours": alarm["median_lead_time_hours"],
            "alarm_episode_policy": alarm["alarm_episode_policy"],
        },
        "challenge_label_secondary": {
            "outcome_estimand": "PhysioNet/CinC 2019 shifted persistent label",
            "utility_at_nested_onset_alarm_policy": challenge_utility(
                oof.assign(_policy=(oof["prob_calibrated"] >= oof["nested_alarm_threshold"]).astype(float)),
                "_policy", 0.5,
            ),
            "interpretation": "secondary compatibility analysis; it is not the fixed-horizon target",
        },
    }
    atomic_json(output_dir / f"{representation}_metrics.json", summary)
    atomic_csv(
        pd.DataFrame(onset.decision_curve(
            oof,
            "prob_calibrated",
            FEATURE_POLICY["dca_threshold_probabilities"],
            repeats=FEATURE_POLICY["dca_patient_cluster_bootstrap_repeats"],
            seed=SEED,
        )),
        output_dir / f"{representation}_dca.csv",
    )
    return summary


def validate_primary_nested_provenance(
    selection: pd.DataFrame,
    inner: pd.DataFrame,
    representation: str,
) -> None:
    """Reject incomplete or policy-inconsistent nested selection evidence."""
    candidates = {candidate["id"]: candidate for candidate in MODEL_CANDIDATES}
    lifts = set(onset.KOOPMAN_POLICY["lifts"] if representation == "C3" else ("identity",))
    selection_required = {
        "outer_fold", "selected_candidate", "selected_hyperparameters", "selected_lift",
        "selected_tree_count_from_inner_only", "selected_signals_outer_train",
        "selected_signals_hash", "koopman_training_transition_counts", "feature_count",
        "feature_column_hash", "calibrator_selected_by_inner_oof_brier",
        "nested_alarm_threshold_from_inner_oof_only", "inner_oof_alarm_budget",
        "inner_oof_useful_sensitivity", "outer_train_patient_hash",
        "outer_test_patient_hash", "outer_train_patient_count", "outer_test_patient_count",
    }
    inner_required = {
        "representation", "outer_fold", "lift", "candidate", "inner_fold",
        "patient_balanced_average_precision", "best_round", "selected_signals",
        "koopman_training_transition_counts", "fit_patient_hash", "valid_patient_hash",
    }
    if not selection_required.issubset(selection) or not inner_required.issubset(inner):
        raise PipelineError(f"{representation} nested provenance is missing required fields")
    expected_outer = set(range(MODEL_POLICY["outer_folds"]))
    expected_lift_values = onset.KOOPMAN_POLICY["lifts"] if representation == "C3" else ("not_applicable",)
    expected_inner_rows = (
        MODEL_POLICY["outer_folds"] * MODEL_POLICY["inner_folds"]
        * len(candidates) * len(lifts)
    )
    try:
        parameters_valid = all(
            json.loads(row.selected_hyperparameters) == candidates[row.selected_candidate]
            for row in selection.itertuples()
        )
        selection_signals = [json.loads(value) for value in selection["selected_signals_outer_train"]]
        selection_counts = [json.loads(value) for value in selection["koopman_training_transition_counts"]]
        inner_signals = [json.loads(value) for value in inner["selected_signals"]]
        inner_counts = [json.loads(value) for value in inner["koopman_training_transition_counts"]]
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise PipelineError(f"{representation} nested provenance contains invalid JSON") from exc
    hashes = pd.concat([
        selection["selected_signals_hash"], selection["feature_column_hash"],
        selection["outer_train_patient_hash"], selection["outer_test_patient_hash"],
        inner["fit_patient_hash"], inner["valid_patient_hash"],
    ]).astype(str)
    rounds = pd.to_numeric(selection["selected_tree_count_from_inner_only"], errors="coerce")
    inner_rounds = pd.to_numeric(inner["best_round"], errors="coerce")
    counts = selection[["outer_train_patient_count", "outer_test_patient_count"]].apply(
        pd.to_numeric, errors="coerce"
    )
    thresholds = pd.to_numeric(
        selection["nested_alarm_threshold_from_inner_oof_only"], errors="coerce"
    )
    budgets = pd.to_numeric(selection["inner_oof_alarm_budget"], errors="coerce")
    sensitivity = pd.to_numeric(selection["inner_oof_useful_sensitivity"], errors="coerce")
    scores = pd.to_numeric(inner["patient_balanced_average_precision"], errors="coerce")
    feature_counts = pd.to_numeric(selection["feature_count"], errors="coerce")
    if (
        len(selection) != MODEL_POLICY["outer_folds"]
        or set(selection["outer_fold"]) != expected_outer
        or len(inner) != expected_inner_rows
        or set(inner["outer_fold"]) != expected_outer
        or set(inner["inner_fold"]) != set(range(MODEL_POLICY["inner_folds"]))
        or set(inner["candidate"]) != set(candidates)
        or set(inner["lift"]) != lifts
        or set(inner["representation"]) != {representation}
        or inner.duplicated(["outer_fold", "lift", "candidate", "inner_fold"]).any()
        or not set(selection["selected_candidate"]).issubset(candidates)
        or not set(selection["selected_lift"]).issubset(expected_lift_values)
        or not parameters_valid
        or not hashes.str.fullmatch(r"[0-9a-f]{64}").all()
        or not np.isfinite(rounds).all()
        or not rounds.between(1, MODEL_POLICY["early_stopping_max_estimators"]).all()
        or not np.equal(rounds, np.floor(rounds)).all()
        or not np.isfinite(inner_rounds).all()
        or not inner_rounds.between(1, MODEL_POLICY["early_stopping_max_estimators"]).all()
        or not np.equal(inner_rounds, np.floor(inner_rounds)).all()
        or not np.isfinite(counts).all().all()
        or (counts <= 0).any().any()
        or not counts.sum(axis=1).eq(DATA_POLICY["patient_count"]).all()
        or not np.isfinite(thresholds).all()
        or not thresholds.isin(onset.ALARM_POLICY["threshold_grid"]).all()
        or not np.isfinite(budgets).all()
        or not budgets.between(
            0, onset.ALARM_POLICY["maximum_false_alarm_episodes_per_patient_day"]
        ).all()
        or not np.isfinite(sensitivity).all()
        or not sensitivity.between(0, 1).all()
        or not np.isfinite(scores).all()
        or not scores.between(0, 1).all()
        or not np.isfinite(feature_counts).all()
        or not np.equal(feature_counts, np.floor(feature_counts)).all()
        or (feature_counts <= 0).any()
        or not all(
            observed == stable_hash(signals)
            for observed, signals in zip(selection["selected_signals_hash"], selection_signals)
        )
        or (inner["fit_patient_hash"] == inner["valid_patient_hash"]).any()
        or not set(selection["calibrator_selected_by_inner_oof_brier"]).issubset(
            set(onset.CALIBRATION_POLICY["candidates"])
        )
    ):
        raise PipelineError(f"{representation} nested provenance is invalid")

    maximum_signals = onset.KOOPMAN_POLICY["maximum_signals"]
    maximum_transitions = onset.KOOPMAN_POLICY["maximum_training_transitions_per_signal"]
    for signals, transition_counts in zip(
        selection_signals + inner_signals,
        selection_counts + inner_counts,
    ):
        if (
            not isinstance(signals, list)
            or len(signals) > maximum_signals
            or len(signals) != len(set(signals))
            or not set(signals).issubset(DYNAMIC_COLUMNS)
            or not isinstance(transition_counts, dict)
            or any(
                signal not in signals
                or isinstance(count, bool)
                or not isinstance(count, int)
                or not 0 <= count <= maximum_transitions
                for signal, count in transition_counts.items()
            )
            or (representation == "C3" and set(transition_counts) != set(signals))
            or (representation != "C3" and transition_counts)
        ):
            raise PipelineError(f"{representation} fold-local representation provenance is invalid")


def fit_primary_source_transport(
    features: pd.DataFrame,
    representation: str,
    train_source: str,
    test_source: str,
    gpu: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Fit every parameter in one SourceSet and evaluate once in the other."""
    train = features.loc[features["SourceSet"] == train_source].copy()
    test = features.loc[features["SourceSet"] == test_source].copy()
    if train.empty or test.empty or train_source == test_source:
        raise PipelineError("Direct-onset transport requires distinct nonempty A/B sources")
    candidate, lift, rounds, inner_oof, detail = select_inner_primary_model(
        train, representation, gpu, 100 + (0 if train_source == "A" else 1)
    )
    calibration = onset.fit_calibration_policy(inner_oof, "prob_raw")
    inner_oof["prob_calibrated"] = onset.apply_calibration(calibration, inner_oof["prob_raw"])
    threshold, _ = onset.select_alarm_threshold(inner_oof, "prob_calibrated")
    representation_fit = fit_primary_representation(train, representation, lift)
    koopman_fit = representation_fit["koopman"]
    transformed_train = transform_primary_representation(train, representation_fit)
    transformed_test = transform_primary_representation(test, representation_fit)
    columns = primary_model_features(transformed_train, representation)
    model = xgb_model(candidate, SEED + 500 + (0 if train_source == "A" else 1), gpu, rounds)
    fit_xgb(
        model, onset.primary_decisions(transformed_train), columns,
        target_column=onset.TARGET_COLUMN,
    )
    calibrated_probability = onset.apply_calibration(
        calibration, model.predict_proba(matrix(transformed_test, columns))[:, 1]
    )
    transformed_test = attach_prediction_columns(
        transformed_test,
        {"prob_calibrated": calibrated_probability},
        "transport_threshold",
        threshold,
    )
    performance = onset.primary_performance(transformed_test, "prob_calibrated")
    alarm = onset.alarm_metrics(transformed_test, "prob_calibrated", "transport_threshold")
    experiment = f"train_{train_source}_test_{test_source}"
    evidence = [
        {"experiment": experiment, "representation": representation, **row}
        for row in detail
    ]
    return {
        "experiment": experiment,
        "validation_scope": "public SourceSet transport; not independent external validation",
        "representation": representation,
        "outcome_estimand": "true reconstructed onset in 1--6 hours",
        "source_set_is_predictor": False,
        "destination_labels_used_for_fitting": False,
        "n_train_patients": int(train["Patient_ID"].nunique()),
        "n_test_patients": int(test["Patient_ID"].nunique()),
        "selected_candidate": candidate["id"],
        "selected_lift": lift if representation == "C3" else "not_applicable",
        "selected_tree_count_from_train_source_inner_only": rounds,
        "selected_signals_train_source": json.dumps(list(representation_fit["selected_signals"])),
        "koopman_training_transition_counts_train_source": json.dumps(
            koopman_fit.training_transition_counts if koopman_fit is not None else {},
            sort_keys=True,
        ),
        "calibrator_train_source_inner_only": calibration.method,
        "threshold_train_source_inner_only": threshold,
        "train_source_inner_fold_hash": stable_hash(evidence),
        "feature_column_hash": stable_hash(columns),
        **performance,
        "useful_sensitivity": alarm["useful_sensitivity"],
        "false_alarm_episodes_per_patient_day": alarm["false_alarm_episodes_per_patient_day"],
        "median_lead_time_hours": alarm["median_lead_time_hours"],
    }, evidence


def artifact_hashes(run_dir: Path) -> dict[str, str]:
    paths = [
        path for path in sorted(run_dir.rglob("*"))
        if path.is_file() and path.name != "result_manifest.json"
    ]
    if any(path.is_symlink() for path in paths):
        raise PipelineError("Result artifact inventory refuses symbolic links")
    return {str(path.relative_to(run_dir)): sha256_file(path) for path in paths}


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
            if ".." in artifact_path.parts:
                raise PipelineError(f"Lineage contains an unsafe artifact path: {artifact}")
            artifact_path = run_dir / artifact_path
        if artifact_path.is_symlink() or not artifact_path.is_file() or sha256_file(artifact_path) != node["sha256"]:
            raise PipelineError(f"Lineage artifact hash mismatch: {artifact}")
        commit = node["generator_git_commit"]
        if not node["generator"] or not isinstance(commit, str) or len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit) or not node["definition_ids"] or not isinstance(node["inputs"], dict):
            raise PipelineError(f"Lineage node lacks semantic provenance: {artifact}")
        for input_artifact, input_sha256 in node["inputs"].items():
            if input_artifact not in nodes or nodes[input_artifact]["sha256"] != input_sha256:
                raise PipelineError(f"Lineage input is missing or mismatched: {artifact} <- {input_artifact}")
    return len(nodes)


DIRECT_ONSET_STATUSES = {
    "INTERNAL_IMPROVEMENT_CONFIRMED",
    "TRANSPORT_ROBUSTNESS_CONFIRMED",
    "PROMISING_BUT_NOT_TRANSPORTABLE",
    "NO_VERIFIED_IMPROVEMENT",
}


def _require_stage(run_dir: Path, stage: str) -> dict[str, Any]:
    path = run_dir / f"{stage}_stage_manifest.json"
    if not path.is_file():
        raise PipelineError(f"Required {stage} stage manifest is missing")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("stage") != stage or manifest.get("status") != "PASS":
        raise PipelineError(f"Required {stage} stage did not pass")
    for artifact, expected in manifest.get("artifacts", {}).items():
        relative = Path(artifact)
        if relative.is_absolute() or ".." in relative.parts:
            raise PipelineError(f"{stage} stage contains an unsafe artifact path")
        target = run_dir / relative
        if target.is_symlink() or not target.is_file() or sha256_file(target) != expected:
            raise PipelineError(f"{stage} stage artifact is missing or changed: {artifact}")
    return manifest


def _write_stage(run_dir: Path, stage: str, artifacts: Iterable[str], detail: dict[str, Any]) -> dict[str, Any]:
    artifact_paths = [Path(artifact) for artifact in artifacts]
    if any(
        path.is_absolute() or ".." in path.parts or (run_dir / path).is_symlink()
        for path in artifact_paths
    ):
        raise PipelineError(f"{stage} stage refuses an unsafe artifact path")
    payload = {
        "stage": stage,
        "status": "PASS",
        "completed_at_utc": utc_now(),
        "artifacts": {str(path): sha256_file(run_dir / path) for path in artifact_paths},
        **detail,
    }
    atomic_json(run_dir / f"{stage}_stage_manifest.json", payload)
    return payload


def direct_onset_lineage(
    run_dir: Path,
    runtime: dict[str, Any],
    *,
    include_resources: bool = False,
) -> dict[str, Any]:
    """Build the hash-linked raw-to-report chain from current-run artifacts."""
    commit = runtime["git_commit"]

    def node(
        artifact: str,
        inputs: dict[str, dict[str, Any]],
        generator: str,
        definitions: tuple[str, ...],
    ) -> dict[str, Any]:
        path = Path(artifact)
        if not path.is_absolute():
            path = run_dir / path
        return {
            "kind": "artifact_lineage",
            "artifact": artifact,
            "sha256": sha256_file(path),
            "inputs": {name: value["sha256"] for name, value in inputs.items()},
            "generator": generator,
            "generator_git_commit": commit,
            "definition_ids": list(definitions),
        }

    raw_name = runtime["data_archive_path"]
    raw = node(raw_name, {}, "external_input", ("physionet_cinc_2019_public_AB_archive",))
    runtime_node = node(
        "runtime_manifest.json", {raw_name: raw},
        "src.scientific_pipeline:runtime_manifest",
        ("source_environment_configuration", "dataset_and_policy_hashes"),
    )
    harmonized = node(
        "harmonized.csv", {raw_name: raw},
        "src.scientific_pipeline:harmonize_archive",
        ("official_40_predictor_schema", "true_onset_reconstruction"),
    )
    features = node(
        "features.csv", {"harmonized.csv": harmonized},
        "src.scientific_pipeline:build_features",
        ("causal_features", "direct_onset_1_to_6h_target"),
    )
    folds = node(
        "folds.csv", {"features.csv": features},
        "src.scientific_pipeline:write_folds",
        ("patient_grouped_outer_folds",),
    )
    model_runtime = node(
        "model_runtime_manifest.json", {"runtime_manifest.json": runtime_node},
        "src.scientific_pipeline:model_direct_onset_stage",
        ("model_allocation", "validated_gpu_runtime", "xgboost_backend"),
    )
    selections = {}
    inner_selections = {}
    oofs = {}
    reports = {}
    dca = {}
    reliability = {}
    alarm_events = {}
    for representation in onset.REPRESENTATIONS:
        selection_name = f"{representation}_nested_selection.csv"
        inner_name = f"{representation}_inner_selection.csv"
        oof_name = f"{representation}_oof_predictions.csv"
        metric_name = f"{representation}_metrics.json"
        dca_name = f"{representation}_dca.csv"
        reliability_name = f"{representation}_reliability.csv"
        alarm_events_name = f"{representation}_alarm_events.csv"
        model_inputs = {
            "features.csv": features, "folds.csv": folds,
            "model_runtime_manifest.json": model_runtime,
        }
        selections[representation] = node(
            selection_name, model_inputs,
            "src.scientific_pipeline:primary_outer_oof",
            ("inner_only_model_calibration_threshold_selection",),
        )
        inner_selections[representation] = node(
            inner_name, model_inputs,
            "src.scientific_pipeline:select_inner_primary_model",
            ("fold_local_representation_and_model_selection",),
        )
        oofs[representation] = node(
            oof_name,
            {
                "features.csv": features,
                "folds.csv": folds,
                selection_name: selections[representation],
                inner_name: inner_selections[representation],
            },
            "src.scientific_pipeline:primary_outer_oof",
            ("outer_held_out_predictions", "nested_calibration", "nested_alarm_threshold"),
        )
        reports[representation] = node(
            metric_name, {oof_name: oofs[representation]},
            "src.scientific_pipeline:primary_model_summary",
            ("patient_balanced_metrics", "official_utility_secondary", "alarm_burden"),
        )
        dca[representation] = node(
            dca_name, {oof_name: oofs[representation]},
            "src.onset_koopman:decision_curve",
            ("fixed_horizon_dca", "patient_cluster_uncertainty"),
        )
        reliability[representation] = node(
            reliability_name, {oof_name: oofs[representation]},
            "src.onset_koopman:primary_reliability_rows",
            ("fixed_equal_width_reliability", "patient_balanced_calibration"),
        )
        alarm_events[representation] = node(
            alarm_events_name, {oof_name: oofs[representation]},
            "src.onset_koopman:alarm_event_rows",
            ("first_eligible_alert", "refractory_alarm_episodes", "alarm_burden"),
        )
    robustness_oof = node(
        "robustness_oof.csv",
        {
            "features.csv": features,
            "folds.csv": folds,
            selections["C0"]["artifact"]: selections["C0"],
            selections["C3"]["artifact"]: selections["C3"],
            oofs["C0"]["artifact"]: oofs["C0"],
            oofs["C3"]["artifact"]: oofs["C3"],
        },
        "src.scientific_pipeline:primary_outer_oof",
        ("prespecified_seed_stability", "prespecified_balance_sensitivity"),
    )
    robustness_report = node(
        "robustness_summary.csv", {"robustness_oof.csv": robustness_oof},
        "src.scientific_pipeline:robustness_summary",
        ("all_configurations_reported_without_winner_selection",),
    )
    robustness_differences = node(
        "robustness_differences.csv", {"robustness_oof.csv": robustness_oof},
        "src.scientific_pipeline:robustness_summary",
        ("C3_minus_C0_seed_and_balance_sensitivity",),
    )
    transport_selection = node(
        "transport_inner_selection.csv",
        {"features.csv": features, "model_runtime_manifest.json": model_runtime},
        "src.scientific_pipeline:select_inner_primary_model",
        ("source_only_nested_representation_and_model_selection",),
    )
    transport = node(
        "transport.csv", {
            "features.csv": features,
            "model_runtime_manifest.json": model_runtime,
            "transport_inner_selection.csv": transport_selection,
        },
        "src.scientific_pipeline:fit_primary_source_transport",
        ("train_A_test_B", "train_B_test_A", "no_destination_label_fitting"),
    )
    inference = node(
        "inference.csv",
        {name["artifact"]: name for name in (oofs["C0"], oofs["C3"])},
        "src.onset_koopman:paired_patient_bootstrap",
        ("paired_patient_cluster_bootstrap_C3_minus_C0",),
    )
    dca_inference = node(
        "dca_inference.csv",
        {name["artifact"]: name for name in (oofs["C0"], oofs["C3"])},
        "src.onset_koopman:paired_decision_curve_difference",
        ("paired_patient_cluster_DCA_C3_minus_C0",),
    )
    metrics = node(
        "metrics.json", {name["artifact"]: name for name in reports.values()},
        "src.scientific_pipeline:finalize_direct_onset_stage",
        ("current_run_C0_to_C3_metrics",),
    )
    ablation = node(
        "ablation_summary.csv",
        {"metrics.json": metrics},
        "src.scientific_pipeline:representation_ablation_summary",
        ("C0_C1_C2_C3_representation_ablation",),
    )
    master_results = node(
        "master_results.csv",
        {
            "metrics.json": metrics,
            "transport.csv": transport,
            "robustness_summary.csv": robustness_report,
        },
        "src.scientific_pipeline:master_result_table",
        ("all_executed_current_run_results", "no_historical_values"),
    )
    gate = node(
        "scientific_gate_status.json",
        {
            "metrics.json": metrics,
            "inference.csv": inference,
            "dca_inference.csv": dca_inference,
            "transport.csv": transport,
            dca["C0"]["artifact"]: dca["C0"],
            dca["C3"]["artifact"]: dca["C3"],
        },
        "src.scientific_pipeline:_primary_gate_status",
        ("predefined_internal_and_transport_gates",),
    )
    probast = node(
        "probast_ai_status.json", {"scientific_gate_status.json": gate},
        "src.scientific_pipeline:finalize_direct_onset_stage",
        ("not_low_risk_until_independent_review",),
    )
    lineage = {
        "raw_data": raw,
        "runtime": runtime_node,
        "harmonized": harmonized,
        "features_and_target": features,
        "folds": folds,
        "model_runtime": model_runtime,
        "inner_selection": inner_selections,
        "models_oof_calibration_thresholds": oofs,
        "model_selection": selections,
        "metrics": reports,
        "dca": dca,
        "reliability": reliability,
        "alarm_events": alarm_events,
        "robustness_oof": robustness_oof,
        "robustness_summary": robustness_report,
        "robustness_differences": robustness_differences,
        "representation_ablation": ablation,
        "master_results": master_results,
        "transport_inner_selection": transport_selection,
        "transport": transport,
        "paired_inference": inference,
        "paired_dca_inference": dca_inference,
        "combined_metrics": metrics,
        "scientific_gate": gate,
        "probast_ai": probast,
    }
    if include_resources:
        lineage["resource_provenance"] = node(
            "resource_manifest.json", {},
            "scripts.resource_provenance:aggregate",
            ("measured_resource_profile_and_stage_usage",),
        )
    return lineage


def prepare_direct_onset_stage(root: Path, archive: Path, run_dir: Path, run_id: str) -> dict[str, Any]:
    """Validate raw data and publish only harmonized/features/folds checkpoints."""
    require_python_hash_seed()
    if run_dir.exists():
        job_id = os.environ.get("SLURM_JOB_ID", "")
        allowed_directories = {Path("resources"), Path("profiles")}
        allowed_files = {
            Path("resources") / f"prepare-{job_id}.gpu.csv",
            Path("resources") / f"prepare-{job_id}.time",
        } if job_id else set()
        unsafe = []
        for path in run_dir.rglob("*"):
            relative = path.relative_to(run_dir)
            if (
                path.is_symlink()
                or (path.is_dir() and relative not in allowed_directories)
                or (not path.is_dir() and relative not in allowed_files)
            ):
                unsafe.append(str(relative))
        if unsafe:
            raise PipelineError(
                f"Prepare refuses unsafe existing run content: {sorted(unsafe)}"
            )
    run_dir.mkdir(parents=True, exist_ok=True)
    runtime = runtime_manifest(root, run_id, sys.argv, archive)
    if runtime["git_dirty"]:
        raise PipelineError("Scientific runs require a clean committed checkout")
    atomic_json(run_dir / "runtime_manifest.json", runtime)
    harmonized = harmonize_archive(archive, run_dir / "harmonized.csv")
    atomic_json(run_dir / "harmonized_manifest.json", harmonized)
    features = build_features(run_dir / "harmonized.csv", run_dir / "features.csv")
    atomic_json(run_dir / "features_manifest.json", features)
    # Folds and cohort accounting need only these columns. Reloading the full
    # wide matrix here exceeded the 10 GB prepare cgroup after feature export.
    summary_columns = [
        "Patient_ID", "SourceSet", "SepsisLabel", "OnsetReconstructionStatus",
        onset.ELIGIBLE_COLUMN,
    ]
    feature_frame = pd.read_csv(run_dir / "features.csv", usecols=summary_columns)
    patient_hash = stable_hash(
        feature_frame.groupby("Patient_ID", sort=True)["SepsisLabel"].max().astype(int).reset_index().to_dict("records")
    )
    folds = write_folds(feature_frame, run_dir / "folds.csv")
    if folds["patient_inventory_hash"] != patient_hash:
        raise PipelineError("Fold patient inventory does not match prepared features")
    atomic_json(run_dir / "folds_manifest.json", folds)
    cohort = cohort_flow_summary(feature_frame, harmonized)
    cohort.update({
        "primary_decision_hours": int((feature_frame[onset.ELIGIBLE_COLUMN] == 1).sum()),
        "left_censored_patients_excluded_from_primary": int(
            feature_frame.loc[
                feature_frame["OnsetReconstructionStatus"] == "septic_onset_left_censored", "Patient_ID"
            ].nunique()
        ),
        "primary_target_policy": onset.TARGET_POLICY,
    })
    atomic_json(run_dir / "cohort_flow.json", cohort)
    return _write_stage(
        run_dir,
        "prepare",
        (
            "runtime_manifest.json", "harmonized.csv", "harmonized_manifest.json",
            "features.csv", "features_manifest.json", "folds.csv",
            "folds_manifest.json", "cohort_flow.json",
        ),
        {
            "run_id": run_id,
            "pipeline_version": PIPELINE_VERSION,
            "stage_workers": stage_worker_count("prepare", DATA_POLICY["patient_count"]),
        },
    )


def model_direct_onset_stage(root: Path, run_dir: Path, run_id: str) -> dict[str, Any]:
    """Fit C0--C3 and A/B transport after a hash-verified prepare stage."""
    require_python_hash_seed()
    _require_stage(run_dir, "prepare")
    if (run_dir / "model_stage_manifest.json").exists():
        raise PipelineError("Model stage refuses to overwrite an existing model-stage manifest")
    runtime = json.loads((run_dir / "runtime_manifest.json").read_text(encoding="utf-8"))
    if runtime.get("run_id") != run_id or runtime.get("pipeline_version") != PIPELINE_VERSION:
        raise PipelineError("Model stage runtime context mismatch")
    gpu = gpu_runtime()
    if os.environ.get("REQUIRE_GPU", "false").lower() == "true" and not gpu["available"]:
        raise PipelineError(f"Model stage requires a validated GPU: {gpu['reason']}")
    model_runtime = {
        "timestamp_utc": utc_now(),
        "hostname": platform.node(),
        "gpu": gpu,
        "xgboost_backend": xgb_backend(gpu),
        "slurm_cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK", "unset"),
        "slurm_mem_per_node": os.environ.get("SLURM_MEM_PER_NODE", "unset"),
        "fit_threads": allocated_cpu_count(),
        "parallel_workers": parallel_model_workers(gpu, MODEL_POLICY["outer_folds"]),
        "parallel_candidates": parallel_candidate_workers(gpu, len(MODEL_CANDIDATES)),
    }
    atomic_json(run_dir / "model_runtime_manifest.json", model_runtime)
    features = pd.read_csv(run_dir / "features.csv").sort_values(["Patient_ID", "ICULOS"], kind="mergesort").reset_index(drop=True)
    folds = pd.read_csv(run_dir / "folds.csv")
    artifacts = ["model_runtime_manifest.json"]
    robustness_frames = {}
    for representation in onset.REPRESENTATIONS:
        oof, detail = primary_outer_oof(features, folds, representation, run_dir, gpu)
        if representation in ROBUSTNESS_POLICY["representations"]:
            robustness_frames[representation] = detail["robustness_oof"]
        artifacts.extend([
            detail["oof_artifact"], detail["selection_artifact"], detail["inner_selection_artifact"],
        ])
    robustness = combine_robustness_oof(robustness_frames)
    atomic_csv(robustness, run_dir / "robustness_oof.csv")
    artifacts.append("robustness_oof.csv")
    transport_tasks = [
        (representation, train_source, test_source)
        for representation in onset.REPRESENTATIONS
        for train_source, test_source in (("A", "B"), ("B", "A"))
    ]

    def fit_transport(task: tuple[str, str, str]):
        representation, train_source, test_source = task
        return fit_primary_source_transport(
            features, representation, train_source, test_source, gpu
        )

    transport_runs = ordered_parallel_map(
        fit_transport,
        transport_tasks,
        parallel_model_workers(gpu, len(transport_tasks)),
    )
    atomic_csv(pd.DataFrame([summary for summary, _ in transport_runs]), run_dir / "transport.csv")
    atomic_csv(
        pd.DataFrame([row for _, evidence in transport_runs for row in evidence]),
        run_dir / "transport_inner_selection.csv",
    )
    artifacts.extend(["transport.csv", "transport_inner_selection.csv"])
    return _write_stage(
        run_dir,
        "model",
        artifacts,
        {
            "run_id": run_id,
            "representations": list(onset.REPRESENTATIONS),
            "robustness_policy": ROBUSTNESS_POLICY,
        },
    )


def _primary_gate_status(
    summaries: dict[str, dict[str, Any]],
    inference: list[dict[str, Any]],
    transport: pd.DataFrame,
    c0_dca: pd.DataFrame,
    c3_dca: pd.DataFrame,
    dca_inference: pd.DataFrame,
) -> dict[str, Any]:
    def finite(value: Any) -> bool:
        return value is not None and bool(np.isfinite(value))

    inference_by_metric = {row["metric"]: row for row in inference}
    ap = inference_by_metric["patient_balanced_average_precision"]
    brier = inference_by_metric["patient_balanced_brier"]
    c0_alarm = summaries["C0"]["alarm_policy"]
    c3_alarm = summaries["C3"]["alarm_policy"]
    dca = c0_dca[["threshold_probability", "model_net_benefit"]].merge(
        c3_dca[[
            "threshold_probability", "model_net_benefit", "model_net_benefit_ci_95_low",
            "treat_all_net_benefit", "treat_none_net_benefit",
        ]],
        on="threshold_probability",
        suffixes=("_C0", "_C3"),
        validate="one_to_one",
    ).merge(dca_inference, on="threshold_probability", validate="one_to_one")
    dca_favorable = bool((
        (dca["C3_minus_C0_net_benefit_ci_95_low"] > 0)
        & (dca["model_net_benefit_ci_95_low"] >= dca["treat_all_net_benefit"])
        & (dca["model_net_benefit_ci_95_low"] >= dca["treat_none_net_benefit"])
    ).any())
    gates = {
        "ap_ci_lower_above_zero": ap["paired_patient_bootstrap_ci_95_low"] > 0,
        "useful_sensitivity_superior_to_C0": c3_alarm["useful_sensitivity"] > c0_alarm["useful_sensitivity"],
        "false_alarm_budget_at_most_0_25": (
            finite(c3_alarm["false_alarm_episodes_per_patient_day"])
            and c3_alarm["false_alarm_episodes_per_patient_day"]
            <= onset.ALARM_POLICY["maximum_false_alarm_episodes_per_patient_day"]
        ),
        "brier_no_statistically_supported_deterioration": brier["paired_patient_bootstrap_ci_95_low"] <= 0,
        "median_lead_time_noninferior": (
            finite(c3_alarm["median_lead_time_hours"])
            and finite(c0_alarm["median_lead_time_hours"])
            and c3_alarm["median_lead_time_hours"] >= c0_alarm["median_lead_time_hours"]
        ),
        "dca_favorable_to_C0_treat_all_treat_none": dca_favorable,
    }
    internal = all(gates.values())
    transport_pivot = transport.pivot(index="experiment", columns="representation", values="patient_balanced_average_precision")
    transport_directions = {
        direction: bool(transport_pivot.loc[direction, "C3"] > transport_pivot.loc[direction, "C0"])
        for direction in ("train_A_test_B", "train_B_test_A")
    }
    transport_confirmed = internal and all(transport_directions.values())
    if transport_confirmed:
        status = "TRANSPORT_ROBUSTNESS_CONFIRMED"
    elif internal:
        status = "PROMISING_BUT_NOT_TRANSPORTABLE"
    else:
        status = "NO_VERIFIED_IMPROVEMENT"
    return {
        "scientific_status": status,
        "internal_gates": gates,
        "transport_C3_superior_to_C0": transport_directions,
        "external_validation": "BLOCKED_EXTERNAL_DATA",
        "interpretation": "Status is determined by current-run artifacts; successful execution alone cannot produce a scientific PASS.",
    }


def finalize_direct_onset_stage(root: Path, run_dir: Path, run_id: str) -> dict[str, Any]:
    """Create data-driven reports and a pending manifest; promotion is separate."""
    require_python_hash_seed()
    _require_stage(run_dir, "prepare")
    _require_stage(run_dir, "model")
    if (run_dir / "finalize_stage_manifest.json").exists():
        raise PipelineError("Finalize stage refuses to overwrite existing evidence")
    workers = stage_worker_count("finalize", len(onset.REPRESENTATIONS))
    summaries = {}
    oofs = {}
    artifacts = []

    def summarize_representation(representation: str):
        oof = pd.read_csv(run_dir / f"{representation}_oof_predictions.csv")
        return representation, oof, primary_model_summary(
            oof, representation, run_dir
        )

    for representation, oof, summary in ordered_parallel_map(
        summarize_representation, onset.REPRESENTATIONS, workers
    ):
        oofs[representation] = oof
        summaries[representation] = summary
        artifacts.extend([
            f"{representation}_metrics.json", f"{representation}_dca.csv",
            f"{representation}_reliability.csv", f"{representation}_alarm_events.csv",
        ])
    atomic_json(run_dir / "metrics.json", summaries)
    artifacts.append("metrics.json")
    atomic_csv(
        representation_ablation_summary(summaries),
        run_dir / "ablation_summary.csv",
    )
    robustness, robustness_differences = robustness_summary(
        pd.read_csv(run_dir / "robustness_oof.csv")
    )
    atomic_csv(robustness, run_dir / "robustness_summary.csv")
    atomic_csv(
        robustness_differences, run_dir / "robustness_differences.csv"
    )
    artifacts.extend([
        "ablation_summary.csv",
        "robustness_summary.csv",
        "robustness_differences.csv",
    ])
    inference = onset.paired_patient_bootstrap(
        oofs["C0"], oofs["C3"], repeats=FEATURE_POLICY["paired_inference_repeats"], seed=SEED
    )
    atomic_csv(pd.DataFrame(inference), run_dir / "inference.csv")
    artifacts.append("inference.csv")
    dca_inference = pd.DataFrame(onset.paired_decision_curve_difference(
        oofs["C0"], oofs["C3"], FEATURE_POLICY["dca_threshold_probabilities"],
        repeats=FEATURE_POLICY["dca_patient_cluster_bootstrap_repeats"], seed=SEED,
    ))
    atomic_csv(dca_inference, run_dir / "dca_inference.csv")
    artifacts.append("dca_inference.csv")
    transport = pd.read_csv(run_dir / "transport.csv")
    atomic_csv(
        master_result_table(summaries, transport, robustness),
        run_dir / "master_results.csv",
    )
    artifacts.append("master_results.csv")
    gate_status = _primary_gate_status(
        summaries,
        inference,
        transport,
        pd.read_csv(run_dir / "C0_dca.csv"),
        pd.read_csv(run_dir / "C3_dca.csv"),
        dca_inference,
    )
    atomic_json(run_dir / "scientific_gate_status.json", gate_status)
    artifacts.append("scientific_gate_status.json")
    probast = {
        "instrument": "PROBAST+AI",
        "status": "NOT_LOW_RISK_UNTIL_INDEPENDENT_REVIEW",
        "assessment_completion": "AUTHOR_ACTION_REQUIRED",
        "external_validation": "BLOCKED_EXTERNAL_DATA",
        "domains": {
            domain: {
                "status": "NOT_ASSESSED_AUTHOR_ACTION_REQUIRED",
                "computational_evidence": evidence,
            }
            for domain, evidence in {
                "participants_and_data_sources": ["cohort_flow.json", "harmonized_manifest.json"],
                "predictors": ["features_manifest.json", "C0_nested_selection.csv", "C3_nested_selection.csv"],
                "outcome": ["runtime_manifest.json", "cohort_flow.json"],
                "analysis": [
                    "metrics.json", "inference.csv", "ablation_summary.csv",
                    "robustness_summary.csv", "master_results.csv",
                    "scientific_gate_status.json",
                ],
                "ai_specific_considerations": ["model_runtime_manifest.json", "resource_manifest.json"],
            }.items()
        },
        "interpretation": "Computational evidence is prepared; human domain-level signalling-question assessment remains required.",
    }
    atomic_json(run_dir / "probast_ai_status.json", probast)
    artifacts.append("probast_ai_status.json")
    stage = _write_stage(
        run_dir,
        "finalize",
        artifacts,
        {
            "run_id": run_id,
            "scientific_status": gate_status["scientific_status"],
            "stage_workers": workers,
        },
    )
    initial = {
        "runtime": json.loads((run_dir / "runtime_manifest.json").read_text(encoding="utf-8")),
        "stage_manifests": {
            name: json.loads((run_dir / f"{name}_stage_manifest.json").read_text(encoding="utf-8"))
            for name in ("prepare", "model", "finalize")
        },
        "lineage": direct_onset_lineage(
            run_dir,
            json.loads((run_dir / "runtime_manifest.json").read_text(encoding="utf-8")),
        ),
        "artifact_sha256": artifact_hashes(run_dir),
        "scientific_status": gate_status["scientific_status"],
        "computational_status": "PENDING_RESOURCE_AND_FINAL_VALIDATION",
        "final_validation": {"status": "PENDING"},
    }
    atomic_json(run_dir / "result_manifest.json", initial)
    return stage


def _validate_representation_artifacts(
    run_dir: Path,
    representation: str,
    expected_identity: pd.DataFrame,
    metrics: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Recompute one representation independently; safe to run concurrently."""
    oof = pd.read_csv(run_dir / f"{representation}_oof_predictions.csv")
    if list(oof.columns) != PRIMARY_OOF_COLUMNS:
        raise PipelineError(f"{representation} OOF schema is invalid")
    identity = oof[PRIMARY_OOF_COLUMNS[:11]].sort_values(
        ["Patient_ID", "ICULOS"], kind="mergesort"
    ).reset_index(drop=True)
    if not identity.equals(expected_identity):
        raise PipelineError(f"{representation} OOF identity does not match features/folds")
    recomputed = onset.primary_performance(oof, "prob_calibrated")
    if any(metrics[representation].get(name) != value for name, value in recomputed.items()):
        raise PipelineError(f"{representation} primary metrics do not reproduce from OOF")
    reported_summary = json.loads(
        (run_dir / f"{representation}_metrics.json").read_text(encoding="utf-8")
    )
    if stable_hash(reported_summary) != stable_hash(metrics[representation]):
        raise PipelineError(f"{representation} metric products disagree")
    decisions = onset.primary_decisions(oof)
    calibration = calibration_metrics(
        binary_array(decisions[onset.TARGET_COLUMN], f"{representation} calibration validation"),
        probability_array(decisions["prob_calibrated"], f"{representation} calibration validation"),
        onset.equal_patient_weights(decisions),
    )
    reported_calibration = reported_summary.get("calibration", {})
    if (
        any(
            not np.isclose(reported_calibration.get(name, math.nan), value)
            for name, value in calibration.items()
        )
        or reported_calibration.get("uncertainty_repeats")
        != FEATURE_POLICY["calibration_patient_cluster_bootstrap_repeats"]
        or any(
            not np.isfinite(reported_calibration.get(f"{name}_ci_95_low", math.nan))
            or not np.isfinite(reported_calibration.get(f"{name}_ci_95_high", math.nan))
            or reported_calibration[f"{name}_ci_95_low"]
            > reported_calibration[f"{name}_ci_95_high"]
            for name in calibration
        )
    ):
        raise PipelineError(f"{representation} calibration report is not traceable to OOF")
    utility = challenge_utility(
        oof.assign(
            _policy=(
                oof["prob_calibrated"] >= oof["nested_alarm_threshold"]
            ).astype(float)
        ),
        "_policy",
        0.5,
    )
    if not np.isclose(
        reported_summary.get("challenge_label_secondary", {}).get(
            "utility_at_nested_onset_alarm_policy", math.nan
        ),
        utility,
    ):
        raise PipelineError(f"{representation} official Utility does not reproduce from OOF")
    expected_dca = pd.DataFrame(onset.decision_curve(
        oof,
        "prob_calibrated",
        FEATURE_POLICY["dca_threshold_probabilities"],
        repeats=FEATURE_POLICY["dca_patient_cluster_bootstrap_repeats"],
        seed=SEED,
    ))
    observed_dca = pd.read_csv(run_dir / f"{representation}_dca.csv")
    try:
        pd.testing.assert_frame_equal(
            observed_dca, expected_dca,
            check_dtype=False, check_exact=False, rtol=1e-12, atol=1e-12,
        )
    except AssertionError as exc:
        raise PipelineError(f"{representation} DCA does not reproduce from OOF") from exc
    selection = pd.read_csv(run_dir / f"{representation}_nested_selection.csv")
    inner_selection = pd.read_csv(run_dir / f"{representation}_inner_selection.csv")
    validate_primary_nested_provenance(selection, inner_selection, representation)
    selected_threshold = selection.set_index("outer_fold")[
        "nested_alarm_threshold_from_inner_oof_only"
    ]
    if not oof["nested_alarm_threshold"].eq(oof["Fold"].map(selected_threshold)).all():
        raise PipelineError(f"{representation} OOF alarm thresholds do not match nested selection")
    expected_reliability = pd.DataFrame(onset.primary_reliability_rows(
        oof, "prob_calibrated", representation, FEATURE_POLICY["ece_equal_width_bins"]
    ))
    observed_reliability = pd.read_csv(run_dir / f"{representation}_reliability.csv")
    try:
        pd.testing.assert_frame_equal(
            observed_reliability, expected_reliability,
            check_dtype=False, check_exact=False, rtol=1e-12, atol=1e-12,
        )
    except AssertionError as exc:
        raise PipelineError(f"{representation} reliability data do not reproduce from OOF") from exc
    expected_events = pd.DataFrame(onset.alarm_event_rows(
        oof, "prob_calibrated", "nested_alarm_threshold"
    ))
    observed_events = read_alarm_event_artifact(
        run_dir / f"{representation}_alarm_events.csv"
    )
    numeric_event_columns = [
        column for column in expected_events.columns
        if column not in {
            "Patient_ID", "alarm_episode_times_iculos", "useful_alarm_episode_times_iculos"
        }
    ]
    if (
        list(observed_events.columns) != list(expected_events.columns)
        or not observed_events["Patient_ID"].astype(str).equals(
            expected_events["Patient_ID"].astype(str)
        )
        or not np.allclose(
            observed_events[numeric_event_columns].to_numpy(dtype=float),
            expected_events[numeric_event_columns].to_numpy(dtype=float),
            equal_nan=True,
        )
        or any(
            not observed_events[column].fillna("").astype(str).equals(
                expected_events[column].fillna("").astype(str)
            )
            for column in (
                "alarm_episode_times_iculos", "useful_alarm_episode_times_iculos"
            )
        )
    ):
        raise PipelineError(f"{representation} alarm-event data do not reproduce from OOF")
    recomputed_alarm = onset.alarm_metrics(
        oof, "prob_calibrated", "nested_alarm_threshold"
    )
    reported_alarm = metrics[representation]["alarm_policy"]
    for name in (
        "n_patients_with_predictions", "n_monitored_patients", "n_alarm_episodes",
        "n_onset_eligible_septic_patients", "tp_patients", "fn_patients",
        "false_alarm_episodes", "false_alarm_episodes_per_patient_day",
        "repeated_alarm_episodes", "late_pre_onset_alarm_episodes",
        "post_onset_alarm_episodes", "right_censored_alarm_episodes",
        "useful_sensitivity", "median_lead_time_hours",
        "left_censored_unclassified_alarm_episodes", "alarm_episode_policy",
    ):
        reported_value = reported_alarm.get(name)
        recomputed_value = recomputed_alarm[name]
        equal = (
            (
                reported_value is None
                if not np.isfinite(recomputed_value)
                else bool(np.isclose(reported_value, recomputed_value))
            )
            if isinstance(recomputed_value, (int, float))
            and not isinstance(recomputed_value, bool)
            else reported_value == recomputed_value
        )
        if not equal:
            raise PipelineError(
                f"{representation} alarm metric {name} is not traceable to OOF"
            )
    return oof, identity


def validate_direct_onset_manifest(run_dir: Path, allow_pending: bool = False) -> dict[str, Any]:
    manifest_path = run_dir / "result_manifest.json"
    if not manifest_path.is_file():
        raise PipelineError("Direct-onset result manifest is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = "PENDING_RESOURCE_AND_FINAL_VALIDATION" if allow_pending else "COMPUTATIONAL_RUN_VALIDATED"
    if manifest.get("computational_status") != expected:
        raise PipelineError("Direct-onset computational status is invalid")
    if manifest.get("scientific_status") not in DIRECT_ONSET_STATUSES:
        raise PipelineError("Direct-onset scientific status is invalid")
    if not allow_pending:
        validation = manifest.get("final_validation", {})
        validator_commit = str(validation.get("validator_git_commit", ""))
        validator_inventory = str(
            validation.get("validator_source_inventory_sha256", "")
        )
        if (
            validation.get("status") != "PASS"
            or len(validator_commit) != 40
            or len(validator_inventory) != 64
            or any(character not in "0123456789abcdef" for character in validator_commit)
            or any(character not in "0123456789abcdef" for character in validator_inventory)
            or validation.get("validator_git_dirty") is not False
            or validation.get("validator_pipeline_version") != PIPELINE_VERSION
        ):
            raise PipelineError("Final validator source provenance is invalid")
    if "lineage" not in manifest:
        raise PipelineError("Direct-onset artifact lineage is missing")
    validate_lineage_nodes(run_dir, manifest["lineage"])
    for stage in ("prepare", "model", "finalize"):
        current = _require_stage(run_dir, stage)
        if (
            stable_hash(current) != stable_hash(manifest.get("stage_manifests", {}).get(stage))
            or current.get("run_id") != manifest.get("runtime", {}).get("run_id")
        ):
            raise PipelineError(f"Manifest does not contain the current {stage} stage")
    if (
        manifest["stage_manifests"]["prepare"].get("stage_workers")
        != STAGE_POLICY["prepare_workers"]
        or manifest["stage_manifests"]["finalize"].get("stage_workers")
        != STAGE_POLICY["finalize_workers"]
    ):
        raise PipelineError("Prepare/finalize worker provenance is invalid")
    if stable_hash(_require_stage(run_dir, "model").get("robustness_policy")) != stable_hash(
        ROBUSTNESS_POLICY
    ):
        raise PipelineError("Model stage robustness policy is not current")
    validate_artifact_hashes(run_dir, manifest["artifact_sha256"])
    runtime = manifest["runtime"]
    runtime_artifact = json.loads(
        (run_dir / "runtime_manifest.json").read_text(encoding="utf-8")
    )
    if (
        stable_hash(runtime) != stable_hash(runtime_artifact)
        or runtime.get("pipeline_version") != PIPELINE_VERSION
        or runtime.get("git_dirty") is not False
        or runtime.get("feature_policy_hash") != stable_hash(FEATURE_POLICY)
        or runtime.get("model_policy_hash") != stable_hash(MODEL_POLICY)
        or runtime.get("robustness_policy_hash") != stable_hash(ROBUSTNESS_POLICY)
        or runtime.get("stage_policy_hash") != stable_hash(STAGE_POLICY)
        or runtime.get("primary_target_policy_hash") != stable_hash(onset.TARGET_POLICY)
        or runtime.get("koopman_policy_hash") != stable_hash(onset.KOOPMAN_POLICY)
        or runtime.get("direct_onset_calibration_policy_hash") != stable_hash(onset.CALIBRATION_POLICY)
        or runtime.get("direct_onset_alarm_policy_hash") != stable_hash(onset.ALARM_POLICY)
        or runtime.get("primary_representations") != list(onset.REPRESENTATIONS)
    ):
        raise PipelineError("Direct-onset runtime policy provenance is invalid")
    feature_identity = pd.read_csv(
        run_dir / "features.csv",
        usecols=[
            "Patient_ID", "SourceSet", "ICULOS", "Age", "SepsisLabel",
            "TrueSepsisOnset_ICULOS", "OnsetReconstructionStatus",
            onset.TARGET_COLUMN, onset.ELIGIBLE_COLUMN, onset.HOURS_TO_ONSET_COLUMN,
        ],
    ).loc[:, PRIMARY_OOF_COLUMNS[:10]]
    folds = pd.read_csv(run_dir / "folds.csv")
    harmonized_manifest = json.loads(
        (run_dir / "harmonized_manifest.json").read_text(encoding="utf-8")
    )
    features_manifest = json.loads(
        (run_dir / "features_manifest.json").read_text(encoding="utf-8")
    )
    folds_manifest = json.loads(
        (run_dir / "folds_manifest.json").read_text(encoding="utf-8")
    )
    feature_columns = pd.read_csv(run_dir / "features.csv", nrows=0).columns.tolist()
    empty_features = pd.DataFrame(columns=feature_columns)
    source_counts = {
        key: int(value)
        for key, value in feature_identity.groupby("SourceSet")["Patient_ID"].nunique().to_dict().items()
    }
    if (
        harmonized_manifest.get("artifact_sha256") != sha256_file(run_dir / "harmonized.csv")
        or harmonized_manifest.get("archive_sha256") != runtime.get("data_archive_sha256")
        or harmonized_manifest.get("row_count") != len(feature_identity)
        or harmonized_manifest.get("patient_count") != feature_identity["Patient_ID"].nunique()
        or harmonized_manifest.get("source_patient_counts") != source_counts
        or features_manifest.get("artifact_sha256") != sha256_file(run_dir / "features.csv")
        or features_manifest.get("input_sha256") != sha256_file(run_dir / "harmonized.csv")
        or features_manifest.get("row_count") != len(feature_identity)
        or features_manifest.get("feature_columns") != feature_columns
        or features_manifest.get("feature_column_hash") != stable_hash(feature_columns)
        or features_manifest.get("base_model_columns") != model_features(empty_features, "baseline")
        or features_manifest.get("base_model_column_hash") != stable_hash(model_features(empty_features, "baseline"))
        or features_manifest.get("C0_model_columns") != model_features(empty_features, "enhanced")
        or features_manifest.get("C0_model_column_hash") != stable_hash(model_features(empty_features, "enhanced"))
        or folds_manifest.get("artifact_sha256") != sha256_file(run_dir / "folds.csv")
        or folds_manifest.get("patient_count") != len(folds)
        or folds_manifest.get("fold_count") != MODEL_POLICY["outer_folds"]
        or folds_manifest.get("seed") != SEED
        or folds_manifest.get("patient_inventory_hash") != stable_hash(
            folds[["Patient_ID", "SepsisLabel"]].to_dict("records")
        )
    ):
        raise PipelineError("Prepared data, feature, or fold provenance is invalid")
    expected_cohort = cohort_flow_summary(feature_identity, harmonized_manifest)
    expected_cohort.update({
        "primary_decision_hours": int((feature_identity[onset.ELIGIBLE_COLUMN] == 1).sum()),
        "left_censored_patients_excluded_from_primary": int(
            feature_identity.loc[
                feature_identity["OnsetReconstructionStatus"] == "septic_onset_left_censored",
                "Patient_ID",
            ].nunique()
        ),
        "primary_target_policy": onset.TARGET_POLICY,
    })
    if stable_hash(expected_cohort) != stable_hash(
        json.loads((run_dir / "cohort_flow.json").read_text(encoding="utf-8"))
    ):
        raise PipelineError("Cohort flow is not traceable to prepared features")
    expected_identity = feature_identity.merge(
        folds[["Patient_ID", "Fold"]], on="Patient_ID", validate="many_to_one"
    ).sort_values(["Patient_ID", "ICULOS"], kind="mergesort").reset_index(drop=True)
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    if set(metrics) != set(onset.REPRESENTATIONS):
        raise PipelineError("Combined metrics do not contain exactly C0--C3")
    validation = ordered_parallel_map(
        lambda representation: (
            representation,
            *_validate_representation_artifacts(
                run_dir, representation, expected_identity, metrics
            ),
        ),
        onset.REPRESENTATIONS,
        stage_worker_count("finalize", len(onset.REPRESENTATIONS)),
    )
    validated_oofs = {
        representation: oof for representation, oof, _ in validation
    }
    identities = [identity for _, _, identity in validation]
    if not all(identities[0].equals(identity) for identity in identities[1:]):
        raise PipelineError("C0--C3 OOF identities are not paired")
    expected_ablation = representation_ablation_summary(metrics)
    observed_ablation = pd.read_csv(run_dir / "ablation_summary.csv")
    try:
        pd.testing.assert_frame_equal(
            observed_ablation, expected_ablation,
            check_dtype=False, check_exact=False, rtol=1e-12, atol=1e-12,
        )
    except AssertionError as exc:
        raise PipelineError("Representation ablation table does not reproduce") from exc
    robustness_oof = pd.read_csv(run_dir / "robustness_oof.csv")
    required_robustness = ROBUSTNESS_IDENTITY_COLUMNS + [
        robustness_probability_column(representation, seed, balance)
        for representation in ROBUSTNESS_POLICY["representations"]
        for seed, balance, _ in robustness_configurations()
    ]
    if (
        list(robustness_oof.columns) != required_robustness
        or robustness_oof.duplicated(["Patient_ID", "ICULOS"]).any()
        or not np.isfinite(robustness_oof[required_robustness[5:]].to_numpy(
            dtype=float
        )).all()
    ):
        raise PipelineError("Robustness OOF schema or policy coverage is invalid")
    expected_decisions = onset.primary_decisions(
        validated_oofs[ROBUSTNESS_POLICY["representations"][0]]
    )[ROBUSTNESS_IDENTITY_COLUMNS].sort_values(
        ["Patient_ID", "ICULOS"], kind="mergesort"
    ).reset_index(drop=True)
    try:
        pd.testing.assert_frame_equal(
            robustness_oof[ROBUSTNESS_IDENTITY_COLUMNS],
            expected_decisions,
            check_dtype=False,
        )
    except AssertionError as exc:
        raise PipelineError("Robustness OOF identity differs from primary OOF") from exc
    primary_seed = ROBUSTNESS_POLICY["training_seed_bases"][0]
    for representation in ROBUSTNESS_POLICY["representations"]:
        primary_probability = onset.primary_decisions(
            validated_oofs[representation]
        ).sort_values(["Patient_ID", "ICULOS"], kind="mergesort")["prob_raw"]
        observed_probability = robustness_oof[
            robustness_probability_column(
                representation, primary_seed, "equal_patient"
            )
        ]
        if not np.allclose(
            observed_probability.to_numpy(dtype=float),
            primary_probability.to_numpy(dtype=float),
            rtol=1e-7,
            atol=1e-8,
        ):
            raise PipelineError("Primary robustness probabilities differ from primary OOF")
    expected_robustness, expected_robustness_differences = robustness_summary(
        robustness_oof
    )
    for name, expected_frame in (
        ("robustness_summary.csv", expected_robustness),
        ("robustness_differences.csv", expected_robustness_differences),
    ):
        observed = pd.read_csv(run_dir / name)
        try:
            pd.testing.assert_frame_equal(
                observed, expected_frame,
                check_dtype=False, check_exact=False, rtol=1e-12, atol=1e-12,
            )
        except AssertionError as exc:
            raise PipelineError(f"{name} does not reproduce from robustness OOF") from exc
        if not observed["configuration_selected"].eq(False).all():
            raise PipelineError(f"{name} improperly selects a sensitivity configuration")
    transport = pd.read_csv(run_dir / "transport.csv")
    expected_transport = {
        (representation, f"train_{source}_test_{target}")
        for representation in onset.REPRESENTATIONS
        for source, target in (("A", "B"), ("B", "A"))
    }
    if (
        set(zip(transport["representation"], transport["experiment"])) != expected_transport
        or not transport["source_set_is_predictor"].eq(False).all()
        or not transport["destination_labels_used_for_fitting"].eq(False).all()
    ):
        raise PipelineError("Direct-onset transport provenance is invalid")
    transport_evidence = pd.read_csv(run_dir / "transport_inner_selection.csv")
    evidence_required = {
        "experiment", "representation", "lift", "candidate", "inner_fold",
        "patient_balanced_average_precision", "best_round", "selected_signals",
        "koopman_training_transition_counts", "fit_patient_hash", "valid_patient_hash",
    }
    if not evidence_required.issubset(transport_evidence):
        raise PipelineError("Transport inner-selection evidence is incomplete")
    for summary in transport.itertuples(index=False):
        evidence = transport_evidence.loc[
            (transport_evidence["experiment"] == summary.experiment)
            & (transport_evidence["representation"] == summary.representation)
        ].reset_index(drop=True)
        expected_lifts = set(
            onset.KOOPMAN_POLICY["lifts"]
            if summary.representation == "C3" else ("identity",)
        )
        expected_rows = (
            MODEL_POLICY["inner_folds"] * len(MODEL_CANDIDATES) * len(expected_lifts)
        )
        scores = pd.to_numeric(
            evidence["patient_balanced_average_precision"], errors="coerce"
        )
        rounds = pd.to_numeric(evidence["best_round"], errors="coerce")
        hashes = pd.concat([
            evidence["fit_patient_hash"], evidence["valid_patient_hash"]
        ]).astype(str)
        if (
            len(evidence) != expected_rows
            or evidence.duplicated(["lift", "candidate", "inner_fold"]).any()
            or set(evidence["lift"]) != expected_lifts
            or set(evidence["candidate"]) != {
                candidate["id"] for candidate in MODEL_CANDIDATES
            }
            or set(evidence["inner_fold"]) != set(range(MODEL_POLICY["inner_folds"]))
            or not np.isfinite(scores).all()
            or not scores.between(0, 1).all()
            or not np.isfinite(rounds).all()
            or not rounds.between(1, MODEL_POLICY["early_stopping_max_estimators"]).all()
            or not np.equal(rounds, np.floor(rounds)).all()
            or not hashes.str.fullmatch(r"[0-9a-f]{64}").all()
            or (evidence["fit_patient_hash"] == evidence["valid_patient_hash"]).any()
            or stable_hash(evidence.to_dict("records"))
            != summary.train_source_inner_fold_hash
        ):
            raise PipelineError(
                f"Transport evidence is invalid for {summary.representation} {summary.experiment}"
            )
    if (
        not set(transport["selected_candidate"]).issubset(
            {candidate["id"] for candidate in MODEL_CANDIDATES}
        )
        or not set(transport["calibrator_train_source_inner_only"]).issubset(
            set(onset.CALIBRATION_POLICY["candidates"])
        )
        or not pd.to_numeric(
            transport["threshold_train_source_inner_only"], errors="coerce"
        ).isin(onset.ALARM_POLICY["threshold_grid"]).all()
        or not transport["feature_column_hash"].astype(str).str.fullmatch(r"[0-9a-f]{64}").all()
    ):
        raise PipelineError("Transport model-selection summary is invalid")
    try:
        transport_counts = [json.loads(value) for value in transport.loc[
            transport["representation"] == "C3",
            "koopman_training_transition_counts_train_source",
        ]]
        transport_signals = [json.loads(value) for value in transport.loc[
            transport["representation"] == "C3", "selected_signals_train_source",
        ]]
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise PipelineError("Transport transition-count provenance is invalid") from exc
    if len(transport_counts) != 2 or any(
        not isinstance(counts, dict)
        or not isinstance(signals, list)
        or set(counts) != set(signals)
        or not set(signals).issubset(DYNAMIC_COLUMNS)
        or any(
            isinstance(count, bool)
            or not isinstance(count, int)
            or not 0 <= count <= onset.KOOPMAN_POLICY["maximum_training_transitions_per_signal"]
            for count in counts.values()
        )
        for signals, counts in zip(transport_signals, transport_counts)
    ):
        raise PipelineError("Transport transition-count provenance is invalid")
    expected_master = master_result_table(metrics, transport, expected_robustness)
    observed_master = pd.read_csv(run_dir / "master_results.csv")
    try:
        pd.testing.assert_frame_equal(
            observed_master, expected_master,
            check_dtype=False, check_exact=False, rtol=1e-12, atol=1e-12,
        )
    except AssertionError as exc:
        raise PipelineError("Master result table does not reproduce") from exc
    if not observed_master["configuration_selected"].eq(False).all():
        raise PipelineError("Master result table improperly selects a sensitivity result")
    inference = pd.read_csv(run_dir / "inference.csv")
    expected_inference_metrics = {
        "patient_balanced_average_precision",
        "patient_balanced_brier",
    }
    if (
        set(inference["metric"]) != expected_inference_metrics
        or not inference["inference_unit"].eq("patient").all()
        or not pd.to_numeric(inference["bootstrap_repeats"], errors="coerce").eq(
            FEATURE_POLICY["paired_inference_repeats"]
        ).all()
    ):
        raise PipelineError("Paired inference provenance is invalid")
    expected_differences = {
        "patient_balanced_average_precision": (
            metrics["C3"]["patient_balanced_average_precision"]
            - metrics["C0"]["patient_balanced_average_precision"]
        ),
        "patient_balanced_brier": (
            metrics["C3"]["patient_balanced_brier"]
            - metrics["C0"]["patient_balanced_brier"]
        ),
    }
    for row in inference.itertuples(index=False):
        if (
            not np.isclose(
                row.candidate_minus_comparator, expected_differences[row.metric]
            )
            or not np.isfinite(row.paired_patient_bootstrap_ci_95_low)
            or not np.isfinite(row.paired_patient_bootstrap_ci_95_high)
            or row.paired_patient_bootstrap_ci_95_low
            > row.paired_patient_bootstrap_ci_95_high
        ):
            raise PipelineError("Paired inference is not traceable to C0/C3 OOF metrics")
    expected_dca_inference = pd.DataFrame(onset.paired_decision_curve_difference(
        pd.read_csv(run_dir / "C0_oof_predictions.csv"),
        pd.read_csv(run_dir / "C3_oof_predictions.csv"),
        FEATURE_POLICY["dca_threshold_probabilities"],
        repeats=FEATURE_POLICY["dca_patient_cluster_bootstrap_repeats"],
        seed=SEED,
    ))
    observed_dca_inference = pd.read_csv(run_dir / "dca_inference.csv")
    try:
        pd.testing.assert_frame_equal(
            observed_dca_inference, expected_dca_inference,
            check_dtype=False, check_exact=False, rtol=1e-12, atol=1e-12,
        )
    except AssertionError as exc:
        raise PipelineError("Paired DCA inference does not reproduce from C0/C3 OOF") from exc
    gate_status = json.loads(
        (run_dir / "scientific_gate_status.json").read_text(encoding="utf-8")
    )
    recomputed_gate = _primary_gate_status(
        metrics,
        inference.to_dict("records"),
        transport,
        pd.read_csv(run_dir / "C0_dca.csv"),
        pd.read_csv(run_dir / "C3_dca.csv"),
        pd.read_csv(run_dir / "dca_inference.csv"),
    )
    if (
        stable_hash(gate_status) != stable_hash(recomputed_gate)
        or manifest.get("scientific_status") != gate_status.get("scientific_status")
        or manifest.get("stage_manifests", {}).get("finalize", {}).get(
            "scientific_status"
        ) != gate_status.get("scientific_status")
    ):
        raise PipelineError("Scientific gate status is not traceable to current results")
    probast = json.loads((run_dir / "probast_ai_status.json").read_text(encoding="utf-8"))
    expected_probast_domains = {
        "participants_and_data_sources", "predictors", "outcome", "analysis",
        "ai_specific_considerations",
    }
    if (
        probast.get("status") != "NOT_LOW_RISK_UNTIL_INDEPENDENT_REVIEW"
        or probast.get("assessment_completion") != "AUTHOR_ACTION_REQUIRED"
        or probast.get("external_validation") != "BLOCKED_EXTERNAL_DATA"
        or set(probast.get("domains", {})) != expected_probast_domains
        or any(
            domain.get("status") != "NOT_ASSESSED_AUTHOR_ACTION_REQUIRED"
            or not domain.get("computational_evidence")
            for domain in probast.get("domains", {}).values()
        )
    ):
        raise PipelineError("PROBAST+AI support artifact overstates or lacks review status")
    if not allow_pending:
        resources = run_dir / "resource_manifest.json"
        if resources.is_symlink() or not resources.is_file():
            raise PipelineError("Final resource manifest is missing")
        resource_manifest = json.loads(resources.read_text(encoding="utf-8"))
        stages = resource_manifest.get("stages", {})
        profile_selection = resource_manifest.get("profile_selection", {})
        selected = profile_selection.get("selected", {})
        fit_profile = profile_selection.get("fit_profile", {})
        parallel_workers = profile_selection.get("parallel_workers", 0)
        parallel_candidates = profile_selection.get("parallel_candidates", 0)
        selected_validation = resource_manifest.get("selected_model_validation", {})
        model_runtime = json.loads((run_dir / "model_runtime_manifest.json").read_text(encoding="utf-8"))
        resource_files = {
            stage: json.loads(
                (run_dir / "resources" / f"{stage}.json").read_text(encoding="utf-8")
            )
            for stage in ("prepare", "model", "finalize")
        }
        profile_file = json.loads(
            (run_dir / "resource_profile_selection.json").read_text(encoding="utf-8")
        )
        selected_validation_file = json.loads(
            (run_dir / "profiles" / "selected-model.json").read_text(encoding="utf-8")
        )
        selected_benchmark = json.loads(
            (run_dir / "profiles" / "selected-model-benchmark.json").read_text(
                encoding="utf-8"
            )
        )
        try:
            runtime_cpus = int(model_runtime.get("slurm_cpus_per_task", 0))
            runtime_gpus = int(model_runtime.get("gpu", {}).get("n_gpus_used", 0))
            expected_fit_memory = max(
                2,
                int(math.ceil((profile_selection.get("memory_basis_peak_gb", 0) * 1.20) / 2) * 2),
            )
        except (TypeError, ValueError) as exc:
            raise PipelineError("Model runtime resource values are invalid") from exc
        if (
            resource_manifest.get("status") != "PASS"
            or set(stages) != {"prepare", "model", "finalize"}
            or any(stable_hash(stages[name]) != stable_hash(resource_files[name]) for name in stages)
            or stable_hash(profile_selection) != stable_hash(profile_file)
            or stable_hash(selected_validation) != stable_hash(selected_validation_file)
            or selected_validation.get("benchmark") != selected_benchmark
            or selected_validation.get("requested") != selected
            or selected_benchmark.get("fit_threads") != fit_profile.get("cpus")
            or selected_benchmark.get("parallel_workers") != parallel_workers
            or selected_benchmark.get("parallel_candidates") != parallel_candidates
            or selected_benchmark.get("concurrent_xgboost_fits")
            != parallel_workers * parallel_candidates
            or resource_manifest.get("caps") != {
                "maximum_cpus": 64, "maximum_memory_gb": 64, "maximum_gpus": 1,
            }
            or resource_manifest.get("no_artificial_memory_fill") is not True
            or not 1 <= selected.get("cpus", 0) <= 64
            or not 2 <= selected.get("memory_gb", 0) <= 64
            or selected.get("gpus") not in (0, 1)
            or fit_profile.get("memory_gb") != expected_fit_memory
            or fit_profile.get("gpus") != selected.get("gpus")
            or not 1 <= fit_profile.get("cpus", 0) <= 32
            or not 1 <= parallel_workers <= MODEL_POLICY["maximum_parallel_workers"]
            or not 1 <= parallel_candidates <= MODEL_POLICY["maximum_parallel_candidates"]
            or selected.get("cpus")
            != fit_profile.get("cpus") * parallel_workers * parallel_candidates
            or selected.get("memory_gb") != fit_profile.get("memory_gb") * parallel_workers
            or len(profile_selection.get("profiles", [])) != 6
            or (
                selected.get("gpus") == 0
                and (
                    profile_selection.get(
                        "selected_profile_active_cpu_efficiency", 0
                    ) <= 0.50
                    or selected_benchmark.get(
                        "xgboost_active_cpu_efficiency", 0
                    ) <= 0.50
                )
            )
            or (
                selected.get("gpus") == 1
                and (
                    profile_selection.get("gpu_speedup_over_best_cpu", 0) <= 1.05
                    or profile_selection.get(
                        "selected_profile_active_gpu_utilization_percent", 0
                    ) <= 50
                    or profile_selection.get(
                        "selected_profile_active_gpu_samples", 0
                    ) < 3
                )
            )
            or profile_selection.get("run_id") != runtime.get("run_id")
            or profile_selection.get("source_git_commit") != runtime.get("git_commit")
            or profile_selection.get("source_inventory_sha256")
            != runtime.get("source_inventory_sha256")
            or any(
                stage.get("status") != "PASS" or stage.get("hostname") != "compute-0-2"
                for stage in stages.values()
            )
            or stages["model"].get("requested") != selected
            or stages["prepare"].get("requested") != {
                "cpus": STAGE_POLICY["prepare_workers"],
                "memory_gb": STAGE_POLICY["prepare_memory_gb"],
                "gpus": 0,
            }
            or stages["finalize"].get("requested") != {
                "cpus": STAGE_POLICY["finalize_workers"],
                "memory_gb": STAGE_POLICY["finalize_memory_gb"],
                "gpus": 0,
            }
            or any(stage.get("run_id") != runtime.get("run_id") for stage in stages.values())
            or any(stage.get("source_git_commit") != runtime.get("git_commit") for stage in stages.values())
            or any(
                stage.get("source_inventory_sha256") != runtime.get("source_inventory_sha256")
                for stage in stages.values()
            )
            or runtime_cpus != selected.get("cpus")
            or selected_validation.get("measured", {}).get("max_rss_gb", math.inf)
            > selected.get("memory_gb", 0)
            or model_runtime.get("fit_threads") != fit_profile.get("cpus")
            or model_runtime.get("parallel_workers") != parallel_workers
            or model_runtime.get("parallel_candidates") != parallel_candidates
            or bool(model_runtime.get("gpu", {}).get("available")) != bool(selected.get("gpus"))
            or runtime_gpus != selected.get("gpus")
            or any(
                stage.get("measured", {}).get("exit_status") != 0
                or stage.get("measured", {}).get("max_rss_gb", math.inf)
                > stage.get("requested", {}).get("memory_gb", 0)
                for stage in stages.values()
            )
            or (
                selected.get("gpus") == 1
                and (
                    "A100" not in str(model_runtime.get("gpu", {}).get("gpu_model"))
                    or not 39000
                    <= stages["model"].get("measured", {}).get("gpu", {}).get(
                        "memory_total_mib", 0
                    )
                    <= 42000
                )
            )
        ):
            raise PipelineError("Final resource manifest is invalid")
    return {"status": "PASS", "validated_at_utc": utc_now()}


def promote_direct_onset_manifest(run_dir: Path) -> dict[str, Any]:
    manifest_path = run_dir / "result_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("computational_status") != "PENDING_RESOURCE_AND_FINAL_VALIDATION":
        raise PipelineError("Only a pending direct-onset manifest can be promoted")
    # Resource provenance is intentionally added after finalize-stage timing is known.
    manifest["artifact_sha256"] = artifact_hashes(run_dir)
    atomic_json(manifest_path, manifest)
    validate_direct_onset_manifest(run_dir, allow_pending=True)
    resources = run_dir / "resource_manifest.json"
    if not resources.is_file():
        raise PipelineError("Cannot promote without final resource provenance")
    manifest["lineage"] = direct_onset_lineage(
        run_dir, manifest["runtime"], include_resources=True
    )
    manifest["artifact_sha256"] = artifact_hashes(run_dir)
    manifest["computational_status"] = "COMPUTATIONAL_RUN_VALIDATED"
    validator_commit = os.environ.get("SOURCE_GIT_COMMIT", "")
    validator_inventory = os.environ.get("SOURCE_INVENTORY_SHA256", "")
    validator_dirty = os.environ.get("SOURCE_GIT_DIRTY", "").lower() != "false"
    if (
        len(validator_commit) != 40
        or len(validator_inventory) != 64
        or validator_dirty
    ):
        raise PipelineError("Final validation requires exact validator source provenance")
    manifest["final_validation"] = {
        "status": "PASS",
        "validated_at_utc": utc_now(),
        "validator_git_commit": validator_commit,
        "validator_git_dirty": validator_dirty,
        "validator_source_inventory_sha256": validator_inventory,
        "validator_pipeline_version": PIPELINE_VERSION,
    }
    atomic_json(manifest_path, manifest)
    return validate_direct_onset_manifest(run_dir)
