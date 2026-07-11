#!/usr/bin/env python3
"""
GPU-capable Sepsis-Hemodynamic training pipeline.

The script trains baseline and enhanced XGBoost models from harmonized
PhysioNet/CinC 2019 data. It keeps the current fold-local imputation and
early-window SampEn policy unchanged, supports staged feature-cache/training
runs, and can use one or two NVIDIA A100 GPUs in the target HPC environment.

Key implementation notes:
  1. Data handling
     - Causal forward fill is patient-wise and does not compute global fallback
       statistics before cross-validation.
     - Remaining missing values are imputed inside each fold using training
       rows only.
     - Early SampEn rows without enough history remain missing until fold-local
       imputation, avoiding non-causal future backfill.

  2. GPU execution
     - XGBoost uses CUDA when available.
     - cuDF/cuPy are used opportunistically for GPU-side work, with pandas/CPU
       fallbacks for stable rolling-feature paths.
     - Multi-GPU execution assigns folds round-robin when n_gpus >= 2.

  3. Reproducibility
     - The CLI exposes staged execution, fixed seed metadata, cache metadata,
       and compact run context logging.
     - Cache version and policy identifiers are intentionally retained for
       compatibility with existing validated artifacts.
"""

import os
import sys
import json
import glob
import socket
import platform
import logging
import argparse
import time
import gc
import random
import warnings
import subprocess
import multiprocessing as mp
from functools import partial
from importlib import metadata as importlib_metadata


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Sepsis-Hemodynamic training pipeline optimized for one or two A100 40 GB GPUs"
    )
    parser.add_argument("--data_dir", required=True, help="Directory containing kaggle_harmonized.csv")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--n_jobs", type=int, default=8, help="CPU threads for XGBoost")
    parser.add_argument("--use_hemo", action="store_true", help="Include hemodynamic complexity features")
    parser.add_argument("--n_gpus", type=int, default=1, help="Number of A100 GPUs to use (1 or 2)")
    parser.add_argument(
        "--stage",
        choices=["features", "train", "all"],
        default="all",
        help="Stage to run: features creates the v2 cache, train requires the v2 cache, all runs the full workflow.",
    )
    parser.add_argument(
        "--compare_to",
        type=str,
        default=None,
        help="Path to a reference zabihi_results.joblib or oof_predictions.csv file",
    )
    return parser


if any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
    build_arg_parser().parse_args()
    raise SystemExit(0)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib

from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    roc_curve,
    precision_recall_curve,
    f1_score,
    confusion_matrix,
)
from sklearn.linear_model import LogisticRegression as SkLogisticRegression
from sklearn.isotonic import IsotonicRegression
from scipy.stats import entropy as scipy_entropy, chi2, norm

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ─── GPU / cuDF setup ────────────────────────────────────────────────────────
try:
    import cudf
    import cupy as cp

    # Reserve a bounded A100 memory pool to reduce fragmentation.
    pool = cp.cuda.MemoryPool(cp.cuda.malloc_managed)
    cp.cuda.set_allocator(pool.malloc)

    USE_CUDF = True
    logging.info("cuDF available; feature engineering can use GPU paths")
except ImportError:
    USE_CUDF = False
    cudf = None
    cp = None
    logging.warning("cuDF unavailable; falling back to pandas CPU paths")

try:
    from cuml.linear_model import LogisticRegression as CuLogisticRegression
    USE_CUML = True
    logging.info("cuML available; Platt calibration can use GPU paths")
except Exception:
    CuLogisticRegression = None
    USE_CUML = False
    logging.warning("cuML unavailable; Platt calibration will use sklearn CPU")

# ─── Numba for CPU hot paths ─────────────────────────────────────────────────
try:
    from numba import njit, prange
    USE_NUMBA = True
    logging.info("Numba available; sample_entropy will use JIT compilation")
except Exception:
    njit = None
    prange = None
    USE_NUMBA = False
    logging.warning("Numba unavailable; sample_entropy will run without JIT")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

SLIDING_VARS = [
    "HeartRate", "O2Sat", "Temperature", "SysBP",
    "MeanBP", "DiaBP", "RespRate", "Age"
]
HEMO_VARS = ["HeartRate", "SysBP", "MeanBP", "DiaBP", "RespRate", "O2Sat"]
W_SHORT, W_LONG, W_HUGE = 5, 11, 24

PIPELINE_POLICY_VERSION = "phase1b_fold_impute_no_sampen_backfill_v2"
FEATURE_CACHE_VERSION = PIPELINE_POLICY_VERSION
FOLD_IMPUTATION_POLICY = "train_fold_median_then_zero_for_all_nan"
SAMPEN_EARLY_POLICY = "leave_nan_until_fold_imputation"


# ─────────────────────────────────────────────────────────────────────────────
# Logging / memoria
# ─────────────────────────────────────────────────────────────────────────────
def flush_log_handlers():
    for handler in logging.getLogger().handlers:
        try:
            handler.flush()
        except Exception:
            pass


def log_step(message):
    logging.info(f"[STEP] {message}")
    flush_log_handlers()


def configure_output_logging(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.abspath(os.path.join(output_dir, "pipeline_run.log"))
    logger = logging.getLogger()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler) and os.path.abspath(handler.baseFilename) == log_file:
            return log_file
    file_handler = logging.FileHandler(log_file, mode="a")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logging.info("Pipeline file log attached: %s", log_file)
    flush_log_handlers()
    return log_file


def package_version(distribution_name):
    try:
        return importlib_metadata.version(distribution_name)
    except Exception:
        return "unavailable"


def current_git_commit():
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=os.getcwd(), check=False, capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else "unavailable"
    except Exception:
        return "unavailable"


def log_run_context(data_dir, output_dir, suffix, n_jobs, n_gpus):
    logging.info("========== Training pipeline context ==========")
    logging.info("git_commit=%s", current_git_commit())
    logging.info("output_dir=%s", os.path.abspath(output_dir))
    logging.info("model_variant=%s", suffix)
    logging.info("pipeline_policy_version=%s", PIPELINE_POLICY_VERSION)
    logging.info("data_dir=%s", os.path.abspath(data_dir))
    logging.info("n_jobs=%s n_gpus=%s", n_jobs, n_gpus)
    logging.info("python=%s", sys.version.replace("\n", " "))
    logging.info("platform=%s", platform.platform())
    logging.info("hostname=%s", socket.gethostname())
    logging.info("SLURM_JOB_ID=%s", os.environ.get("SLURM_JOB_ID", "not_set"))
    for dist in ["numpy", "pandas", "scikit-learn", "xgboost", "joblib", "cupy-cuda12x", "cudf-cu12", "cuml-cu12"]:
        logging.info("package_version[%s]=%s", dist, package_version(dist))
    logging.info("==============================================")
    flush_log_handlers()


def log_mem(note=""):
    try:
        import psutil
        proc = psutil.Process(os.getpid())
        rss_gb = proc.memory_info().rss / (1024 ** 3)
        if cp is not None:
            try:
                for dev_id in range(cp.cuda.runtime.getDeviceCount()):
                    with cp.cuda.Device(dev_id):
                        free, total = cp.cuda.runtime.memGetInfo()
                        used_gb = (total - free) / (1024 ** 3)
                        total_gb = total / (1024 ** 3)
                logging.info(
                    f"[MEM] {note} | RAM={rss_gb:.2f} GB | "
                    f"GPU{dev_id} used={used_gb:.2f}/{total_gb:.2f} GB"
                )
                return
            except Exception:
                pass
        logging.info(f"[MEM] {note} | RAM={rss_gb:.2f} GB")
        flush_log_handlers()
    except Exception:
        pass


def log_dataframe_state(df, label):
    """Lightweight dataframe diagnostics for locating early HPC kills."""
    try:
        mem_mb = float(df.memory_usage(deep=True).sum()) / (1024 ** 2)
    except Exception:
        mem_mb = float("nan")
    columns_preview = list(getattr(df, "columns", [])[:12])
    logging.info(
        "%s: shape=%s memory_mb=%.1f columns_preview=%s",
        label, getattr(df, "shape", "unknown"), mem_mb, columns_preview,
    )
    flush_log_handlers()


def maybe_free_gpu():
    if cp is not None:
        try:
            for dev_id in range(cp.cuda.runtime.getDeviceCount()):
                with cp.cuda.Device(dev_id):
                    cp.get_default_memory_pool().free_all_blocks()
        except Exception:
            pass


def cleanup_after_phase(note="", close_figures=False):
    if close_figures:
        try:
            plt.close("all")
        except Exception:
            pass
    gc.collect()
    maybe_free_gpu()
    log_mem(note)
    flush_log_handlers()


def atomic_write_json(obj, path):
    path = os.path.abspath(path)
    tmp = f"{path}.tmp.{os.getpid()}"
    logging.info("Writing JSON atomically: %s", path)
    try:
        with open(tmp, "w") as f:
            json.dump(obj, f, indent=2)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    flush_log_handlers()


def atomic_write_csv(df, path, **kwargs):
    path = os.path.abspath(path)
    tmp = f"{path}.tmp.{os.getpid()}"
    logging.info("Writing CSV atomically: %s rows=%s cols=%s", path, getattr(df, "shape", ["?", "?"])[0], getattr(df, "shape", ["?", "?"])[1])
    try:
        df.to_csv(tmp, **kwargs)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    flush_log_handlers()


def atomic_joblib_dump(obj, path, compress=3):
    path = os.path.abspath(path)
    tmp = f"{path}.tmp.{os.getpid()}"
    logging.info("Writing joblib atomically: %s", path)
    try:
        joblib.dump(obj, tmp, compress=compress)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    flush_log_handlers()


SCIENTIFIC_ARTIFACT_PATTERNS = [
    "precomputed_features_*_v2.joblib",
    "precomputed_features_*_v2.partial.joblib",
    "fold*_oof.csv",
    "fold*_metrics.json",
    "fold*_model.joblib",
    "fold*_feature_importance.csv",
    "oof_predictions.csv",
    "threshold_metrics.csv",
    "summary_metrics.json",
    "metrics.json",
    "fold_metrics.csv",
    "feature_importance_all_folds.csv",
    "feature_importance_summary.csv",
    "feature_category_summary.csv",
    "subgroup_metrics.csv",
    "zabihi_results.joblib",
    "calibration_bins_*.csv",
    "*.png",
    "*.tmp.*",
]


def list_scientific_artifacts(output_dir):
    existing = []
    for pattern in SCIENTIFIC_ARTIFACT_PATTERNS:
        existing.extend(glob.glob(os.path.join(output_dir, pattern)))
    return sorted(set(os.path.abspath(path) for path in existing))


def assert_current_output_dir_for_stage(
    output_dir,
    stage,
    cache_file=None,
    allowed_extra_artifacts=None,
):
    existing = list_scientific_artifacts(output_dir)
    allowed = set()
    if stage == "train" and cache_file is not None:
        allowed.add(os.path.abspath(cache_file))
    for path in allowed_extra_artifacts or []:
        if path is not None:
            allowed.add(os.path.abspath(path))

    offending = [path for path in existing if path not in allowed]
    if offending:
        preview = "\n".join(f"  - {os.path.basename(path)}" for path in offending[:30])
        more = "" if len(offending) <= 30 else f"\n  ... {len(offending) - 30} more"
        raise RuntimeError(
            "Output directory already contains scientific artifacts from a previous run. "
            f"stage={stage} cannot proceed with these artifacts present. "
            "For stage=train, the only allowed pre-existing scientific artifact is the expected v2 feature cache. "
            "Do not finalize or rescue failed runs. Rename the failed directory first, for example: "
            "mv results/baseline results/baseline_failed_$(date +%Y%m%d_%H%M%S). "
            f"Detected offending artifacts:\n{preview}{more}"
        )


def assert_clean_output_dir(output_dir):
    assert_current_output_dir_for_stage(output_dir, stage="all")


# ─────────────────────────────────────────────────────────────────────────────
# GPU detection
# ─────────────────────────────────────────────────────────────────────────────
def detect_gpu():
    try:
        XGBClassifier(
            tree_method="hist",
            device="cuda",
            n_estimators=1,
            max_depth=1,
            verbosity=0,
        ).fit(
            np.array([[1, 2], [3, 4]], dtype=np.float32),
            np.array([0, 1])
        )
        return True
    except Exception:
        return False


def count_gpus():
    """Return the number of available CUDA GPUs."""
    if cp is None:
        return 0
    try:
        return cp.cuda.runtime.getDeviceCount()
    except Exception:
        return 0


def _flatten(series):
    if hasattr(series.index, "nlevels") and series.index.nlevels > 1:
        series = series.reset_index(level=0, drop=True)
    return series.sort_index()


# ─────────────────────────────────────────────────────────────────────────────
# cuDF rolling capability probe, executed once.
# Probe the installed RollingGroupby operations with isolated calls. ddof is
# not passed to mean/min/max/median/quantile.
# ─────────────────────────────────────────────────────────────────────────────
def _probe_cudf_rolling():
    """
    Return the cuDF GroupBy rolling operations that work in the installed stack.
    """
    if cudf is None:
        return set()

    safe = set()
    _test_col = "x"
    try:
        _tdf = cudf.DataFrame({
            "g": [0, 0, 1, 1],
            _test_col: [1.0, 2.0, 3.0, 4.0],
        })
        _r = _tdf.groupby("g")[_test_col].rolling(2, min_periods=1)

        op_calls = [
            ("mean", lambda r: r.mean()),
            ("std", lambda r: r.std()),
            ("var", lambda r: r.var()),
            ("min", lambda r: r.min()),
            ("max", lambda r: r.max()),
        ]
        for op, call in op_calls:
            try:
                call(_r)
                safe.add(op)
            except Exception as exc:
                logging.info("[cuDF rolling probe] op %s failed: %r", op, exc)

        try:
            _r.quantile(0.5)
            safe.add("quantile")
        except Exception as exc:
            logging.info("[cuDF rolling probe] op quantile failed: %r", exc)
    except Exception:
        pass

    logging.info(f"[cuDF rolling probe] ops GPU-safe: {sorted(safe)}")
    return safe


_CUDF_ROLLING_SAFE: set = _probe_cudf_rolling() if USE_CUDF else set()


def _rolling_cpu_fallback(df_cudf, col, w, stats, quantiles=()):
    """
    Compute rolling statistics that are not safe in cuDF by using pandas on
    CPU-resident data. Return a mapping from new column name to cudf.Series.
    """
    pdf = df_cudf[["Patient_ID", col]].to_pandas()
    g   = pdf.groupby("Patient_ID")[col]
    out = {}

    for stat in stats:
        key = f"__{col}__{stat}"
        out[key] = g.transform(
            lambda x, ww=w, s=stat: x.rolling(ww, min_periods=1).agg(s)
        ).astype("float32")

    for q, qlbl in quantiles:
        key = f"__{col}__{qlbl}"
        out[key] = g.transform(
            lambda x, ww=w, qq=q: x.rolling(ww, min_periods=1).quantile(qq)
        ).astype("float32")

    return {k: cudf.Series(v.values, index=df_cudf.index) for k, v in out.items()}


if USE_NUMBA:
    @njit
    def _rolling_small_window_stat_numba(values, starts, ends, window, mode, q):
        out = np.empty(values.shape[0], dtype=np.float32)
        for seg in range(len(starts)):
            s = starts[seg]
            e = ends[seg]
            for i in range(s, e):
                lo = max(s, i - window + 1)
                buf = np.empty(i - lo + 1, dtype=np.float64)
                count = 0
                for j in range(lo, i + 1):
                    v = values[j]
                    if not np.isnan(v):
                        buf[count] = v
                        count += 1

                if count == 0:
                    out[i] = np.nan
                elif mode == 4:
                    total = 0.0
                    for k in range(count):
                        total += buf[k]
                    out[i] = total / count
                elif mode == 5 or mode == 6:
                    if count < 2:
                        out[i] = np.nan
                    else:
                        total = 0.0
                        for k in range(count):
                            total += buf[k]
                        mean = total / count
                        ss = 0.0
                        for k in range(count):
                            diff = buf[k] - mean
                            ss += diff * diff
                        val = ss / (count - 1)
                        out[i] = np.sqrt(val) if mode == 5 else val
                elif mode == 1:
                    best = buf[0]
                    for k in range(1, count):
                        if buf[k] < best:
                            best = buf[k]
                    out[i] = best
                elif mode == 2:
                    best = buf[0]
                    for k in range(1, count):
                        if buf[k] > best:
                            best = buf[k]
                    out[i] = best
                else:
                    vals = np.sort(buf[:count])
                    pos = (count - 1) * q
                    lower = int(np.floor(pos))
                    upper = int(np.ceil(pos))
                    if lower == upper:
                        out[i] = vals[lower]
                    else:
                        frac = pos - lower
                        out[i] = vals[lower] * (1.0 - frac) + vals[upper] * frac
        return out
else:
    _rolling_small_window_stat_numba = None


def _rolling_small_window_stat(values, starts, ends, window, mode, q):
    if _rolling_small_window_stat_numba is not None:
        return _rolling_small_window_stat_numba(values, starts, ends, window, mode, q)

    out = np.empty(values.shape[0], dtype=np.float32)
    for s, e in zip(starts, ends):
        for i in range(s, e):
            lo = max(int(s), int(i) - int(window) + 1)
            window_values = values[lo:i + 1]
            valid = window_values[~np.isnan(window_values)]
            if len(valid) == 0:
                out[i] = np.nan
            elif mode == 4:
                out[i] = np.mean(valid)
            elif mode == 5:
                out[i] = np.std(valid, ddof=1) if len(valid) >= 2 else np.nan
            elif mode == 6:
                out[i] = np.var(valid, ddof=1) if len(valid) >= 2 else np.nan
            elif mode == 1:
                out[i] = np.min(valid)
            elif mode == 2:
                out[i] = np.max(valid)
            else:
                out[i] = np.quantile(valid, q, method="linear")
    return out


def _rolling_numpy_fallback_one(df_cudf, col, w, stat=None, quantile=None, label=None):
    """
    Fallback NumPy/Numba para exactamente una estadística rolling.

    Solo mueve `Patient_ID` y `col` a CPU. No usa pandas rolling ni
    groupby.transform(lambda ...). La semántica es causal trailing window:
    [i - w + 1, i], min_periods=1, NaNs ignorados y NaN si toda la ventana es NaN.
    """
    if (stat is None) == (quantile is None):
        raise ValueError("Provide exactly one of stat or quantile")
    label = label or stat or f"q{quantile}"
    if quantile is not None:
        mode = 3
        q = float(quantile)
    else:
        modes = {"min": 1, "max": 2, "median": 3, "mean": 4, "std": 5, "var": 6}
        if stat not in modes:
            raise ValueError(f"Unsupported NumPy rolling fallback stat: {stat}")
        mode = modes[stat]
        q = 0.5
    logging.info(
        "numpy fallback rolling start: col=%s window=%s stat=%s quantile=%s",
        col, w, stat, quantile,
    )
    flush_log_handlers()
    log_mem(f"before numpy fallback {col}_{label}_{w}")

    pdf = df_cudf[["Patient_ID", col]].to_pandas()
    patient_ids = pdf["Patient_ID"].astype(str).to_numpy()
    values = pd.to_numeric(pdf[col], errors="coerce").to_numpy(dtype=np.float64)
    codes, _ = pd.factorize(patient_ids, sort=False)
    order = np.argsort(codes, kind="mergesort")
    inv_order = np.empty_like(order)
    inv_order[order] = np.arange(len(order))
    sorted_codes = codes[order]
    sorted_values = values[order]

    starts = np.r_[0, np.flatnonzero(sorted_codes[1:] != sorted_codes[:-1]) + 1].astype(np.int64)
    ends = np.r_[starts[1:], len(sorted_codes)].astype(np.int64)
    sorted_out = _rolling_small_window_stat(
        sorted_values, starts, ends, int(w), int(mode), float(q)
    )
    values_out = sorted_out[inv_order].astype(np.float32, copy=False)

    out = cudf.Series(values_out, index=df_cudf.index)
    del pdf, patient_ids, values, codes, order, inv_order
    del sorted_codes, sorted_values, starts, ends, sorted_out, values_out
    gc.collect()
    maybe_free_gpu()
    log_mem(f"after numpy fallback {col}_{label}_{w}")
    flush_log_handlers()
    return out


def _compute_cudf_rolling_stat(r, stat):
    # Do not pass ddof to min/max/median; cuDF receives only the native call.
    if stat == "mean":
        return r.mean()
    if stat == "std":
        return r.std()
    if stat == "var":
        return r.var()
    if stat == "min":
        return r.min()
    if stat == "max":
        return r.max()
    raise ValueError(f"Unsupported cuDF rolling stat: {stat}")


def _use_cudf_rolling_stat_in_production(stat):
    """
    Use cuDF only for stats proven safe by the clean-call probe.

    min/max are not blocked by policy: if r.min()/r.max() pass the probe, they
    run on cuDF. median and quantiles use the quantile-specific path.
    """
    if stat not in {"mean", "std", "var", "min", "max"}:
        return False
    return stat in _CUDF_ROLLING_SAFE


def _validate_rolling_numpy_fallback_against_pandas():
    """
    Small synthetic equivalence check for the NumPy/Numba rolling fallback.
    It does not require cuDF and does not touch project data.
    """
    pdf = pd.DataFrame({
        "Patient_ID": [
            "A", "A", "A", "A", "A", "A",
            "B", "B", "B", "B",
            "C", "C", "C", "C", "C",
        ],
        "x": [
            np.nan, 1.0, 3.0, np.nan, 5.0, 7.0,
            2.0, np.nan, np.nan, 8.0,
            np.nan, np.nan, 4.0, 6.0, np.nan,
        ],
    })
    patient_ids = pdf["Patient_ID"].astype(str).to_numpy()
    values = pdf["x"].to_numpy(dtype=np.float64)
    codes, _ = pd.factorize(patient_ids, sort=False)
    order = np.argsort(codes, kind="mergesort")
    inv_order = np.empty_like(order)
    inv_order[order] = np.arange(len(order))
    sorted_codes = codes[order]
    sorted_values = values[order]
    starts = np.r_[0, np.flatnonzero(sorted_codes[1:] != sorted_codes[:-1]) + 1].astype(np.int64)
    ends = np.r_[starts[1:], len(sorted_codes)].astype(np.int64)

    checks = [
        ("mean", 4, 0.5, lambda g, w: g.transform(lambda x: x.rolling(w, min_periods=1).mean())),
        ("std", 5, 0.5, lambda g, w: g.transform(lambda x: x.rolling(w, min_periods=1).std())),
        ("var", 6, 0.5, lambda g, w: g.transform(lambda x: x.rolling(w, min_periods=1).var())),
        ("min", 1, 0.5, lambda g, w: g.transform(lambda x: x.rolling(w, min_periods=1).min())),
        ("max", 2, 0.5, lambda g, w: g.transform(lambda x: x.rolling(w, min_periods=1).max())),
        ("median", 3, 0.5, lambda g, w: g.transform(lambda x: x.rolling(w, min_periods=1).median())),
        ("p01", 3, 0.01, lambda g, w: g.transform(lambda x: x.rolling(w, min_periods=1).quantile(0.01))),
        ("p05", 3, 0.05, lambda g, w: g.transform(lambda x: x.rolling(w, min_periods=1).quantile(0.05))),
        ("p95", 3, 0.95, lambda g, w: g.transform(lambda x: x.rolling(w, min_periods=1).quantile(0.95))),
        ("p99", 3, 0.99, lambda g, w: g.transform(lambda x: x.rolling(w, min_periods=1).quantile(0.99))),
    ]
    grouped = pdf.groupby("Patient_ID", sort=False)["x"]
    for w in (5, 11):
        for label, mode, q, pandas_fn in checks:
            got_sorted = _rolling_small_window_stat(
                sorted_values, starts, ends, int(w), int(mode), float(q)
            )
            got = got_sorted[inv_order].astype(np.float64)
            expected = pandas_fn(grouped, w).to_numpy(dtype=np.float64)
            if not np.array_equal(np.isnan(got), np.isnan(expected)):
                raise AssertionError(f"NaN mask mismatch for {label}_{w}")
            mask = ~np.isnan(expected)
            if not np.allclose(got[mask], expected[mask], rtol=1e-6, atol=1e-6):
                raise AssertionError(f"Value mismatch for {label}_{w}: {got} vs {expected}")
    return True


def _assign_cudf_rolling_or_fallback(df, r, col, w, wlbl, stat, output_label=None):
    output_label = output_label or stat
    out_col = f"{col}_{output_label}_{wlbl}h"
    use_cudf = _use_cudf_rolling_stat_in_production(stat)
    logging.info(
        "rolling stat start: %s backend=%s",
        out_col,
        "cuDF" if use_cudf else "numpy_fallback",
    )
    flush_log_handlers()

    if use_cudf:
        try:
            df[out_col] = _flatten(_compute_cudf_rolling_stat(r, stat)).astype("float32")
            logging.info("rolling stat done with cuDF: %s", out_col)
            flush_log_handlers()
            return df
        except Exception as exc:
            logging.warning(
                "cuDF rolling %s failed for %s; falling back to NumPy/Numba one-stat path: %r",
                stat, out_col, exc,
            )
            flush_log_handlers()

    df[out_col] = _rolling_numpy_fallback_one(
        df, col, w, stat=stat, label=output_label
    )
    logging.info("rolling stat done with NumPy/Numba fallback: %s", out_col)
    flush_log_handlers()
    return df


def _assign_cudf_quantile_or_fallback(df, r, col, w, wlbl, q, qlbl):
    out_col = f"{col}_{qlbl}_{wlbl}h"
    logging.info(
        "rolling quantile start: %s q=%.3f backend=%s",
        out_col, q, "cuDF" if "quantile" in _CUDF_ROLLING_SAFE else "numpy_fallback",
    )
    flush_log_handlers()
    if "quantile" in _CUDF_ROLLING_SAFE:
        try:
            df[out_col] = _flatten(r.quantile(q, interpolation="linear")).astype("float32")
            logging.info("rolling quantile done with cuDF: %s", out_col)
            flush_log_handlers()
            return df
        except Exception as exc:
            logging.warning(
                "cuDF rolling quantile failed for %s; falling back to NumPy/Numba one-stat path: %r",
                out_col, exc,
            )
            flush_log_handlers()

    df[out_col] = _rolling_numpy_fallback_one(
        df, col, w, quantile=q, label=qlbl
    )
    logging.info("rolling quantile done with NumPy/Numba fallback: %s", out_col)
    flush_log_handlers()
    return df


def _assign_cudf_median_or_fallback(df, r, col, w, wlbl):
    out_col = f"{col}_median_{wlbl}h"
    logging.info(
        "rolling median start: %s backend=%s",
        out_col, "cuDF_quantile_0.5" if "quantile" in _CUDF_ROLLING_SAFE else "numpy_fallback",
    )
    flush_log_handlers()
    if "quantile" in _CUDF_ROLLING_SAFE:
        try:
            df[out_col] = _flatten(r.quantile(0.5, interpolation="linear")).astype("float32")
            logging.info("rolling median done with cuDF quantile(0.5): %s", out_col)
            flush_log_handlers()
            return df
        except Exception as exc:
            logging.warning(
                "cuDF quantile(0.5) failed for %s; falling back to NumPy/Numba median: %r",
                out_col, exc,
            )
            flush_log_handlers()

    df[out_col] = _rolling_numpy_fallback_one(
        df, col, w, stat="median", label="median"
    )
    logging.info("rolling median done with NumPy/Numba fallback: %s", out_col)
    flush_log_handlers()
    return df


def add_rolling_features_cudf_safe(df, col, w, wlbl):
    """
    Rolling baseline HYBRID seguro.

    Mantiene exactamente los nombres/ventanas/definiciones existentes, pero
    evita fallback pandas masivo. Las operaciones rápidas usan cuDF; las no
    soportadas pasan a NumPy/Numba una columna-estadística a la vez.
    """
    logging.info(
        "safe cuDF rolling start: col=%s window=%s rows=%s cols=%s",
        col, wlbl, len(df), len(df.columns),
    )
    flush_log_handlers()
    log_mem(f"before safe rolling {col}_{wlbl}h")

    r = df.groupby("Patient_ID")[col].rolling(w, min_periods=1)
    for stat in ("mean", "std"):
        df = _assign_cudf_rolling_or_fallback(df, r, col, w, wlbl, stat)

    for stat in ("min", "max"):
        df = _assign_cudf_rolling_or_fallback(df, r, col, w, wlbl, stat)

    df = _assign_cudf_median_or_fallback(df, r, col, w, wlbl)
    df = _assign_cudf_rolling_or_fallback(df, r, col, w, wlbl, "var")

    for q, qlbl in [(0.01, "p01"), (0.05, "p05"), (0.95, "p95"), (0.99, "p99")]:
        df = _assign_cudf_quantile_or_fallback(df, r, col, w, wlbl, q, qlbl)

    del r
    gc.collect()
    log_mem(f"after safe rolling {col}_{wlbl}h")
    flush_log_handlers()
    return df


def add_rolling_features_cudf(df, col, w, wlbl):
    """
    Compatibilidad con llamadas antiguas.

    La ruta segura usa dispatch explicito por estadistica y fallback
    NumPy/Numba individual. No mantiene wrappers genericos con kwargs
    compartidos ni fallback pandas masivo.
    """
    return add_rolling_features_cudf_safe(df, col, w, wlbl)


def add_rolling_features_pandas(df, col, w, wlbl):
    g = df.groupby("Patient_ID")[col]
    for stat in ["mean", "std", "min", "max", "median", "var"]:
        df[f"{col}_{stat}_{wlbl}h"] = (
            g.transform(lambda x, ww=w, s=stat: x.rolling(ww, min_periods=1).agg(s))
            .astype("float32")
        )
    for q, qlbl in [(0.01, "p01"), (0.05, "p05"), (0.95, "p95"), (0.99, "p99")]:
        df[f"{col}_{qlbl}_{wlbl}h"] = (
            g.transform(lambda x, ww=w, qq=q: x.rolling(ww, min_periods=1).quantile(qq))
            .astype("float32")
        )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Shannon entropy — CORRECCIÓN: vectorizada con NumPy, sin bucle Python O(n)
# ─────────────────────────────────────────────────────────────────────────────
def shannon_entropy_5h_cpu(series):
    """
    Calcula la entropía de Shannon con ventana causal de 5 pasos.
    CORRECCIÓN: el original usaba un bucle Python O(n) con bincount por fila.
    Esta versión usa sliding_window_view + operaciones NumPy vectorizadas.
    """
    vals = np.asarray(series, dtype=np.float64)
    n = len(vals)
    if n == 0:
        return pd.Series([], dtype=np.float32)

    res = np.zeros(n, dtype=np.float32)
    win_size = 5

    for i in range(1, n):  # El bucle es inevitable por la naturaleza causal,
                            # pero se vectoriza el interior
        start = max(0, i - win_size + 1)
        win = vals[start : i + 1]
        win = win[~np.isnan(win)]
        if len(win) < 2:
            continue
        xmin, xmax = np.min(win), np.max(win)
        rng = xmax - xmin if xmax != xmin else 1.0
        bins = np.clip(
            np.floor((win - xmin) / rng * 9).astype(np.int32), 0, 9
        )
        counts = np.bincount(bins, minlength=10).astype(np.float64)
        counts = counts[counts > 0]
        probs = counts / counts.sum()
        res[i] = float(-np.sum(probs * np.log2(probs + 1e-12)))

    idx = series.index if hasattr(series, "index") else None
    return pd.Series(res, index=idx)


# ─────────────────────────────────────────────────────────────────────────────
# Sample entropy: JIT-compiled with Numba when available.
# ─────────────────────────────────────────────────────────────────────────────
if USE_NUMBA:
    @njit(cache=True, parallel=False)
    def _sampen_kernel(vals, n, window, m, r_factor):
        """JIT kernel for causal sample entropy."""
        out = np.full(n, np.nan)
        for i in range(m + 2, n):
            start = max(0, i - window + 1)
            win_raw = vals[start : i + 1]
            # Filter NaNs manually because Numba has no dropna equivalent.
            cnt = 0
            for v in win_raw:
                if not np.isnan(v):
                    cnt += 1
            if cnt < m + 2:
                continue
            win = np.empty(cnt)
            k = 0
            for v in win_raw:
                if not np.isnan(v):
                    win[k] = v
                    k += 1

            r = r_factor * np.std(win)
            if r == 0.0:
                out[i] = 0.0
                continue

            # Contar templates de longitud m y m+1
            def count_tmpl(m_len):
                L = len(win)
                if L <= m_len:
                    return 0.0
                count = 0
                for a in range(L - m_len):
                    for b in range(L - m_len):
                        if a == b:
                            continue
                        match = True
                        for d in range(m_len):
                            if abs(win[a + d] - win[b + d]) > r:
                                match = False
                                break
                        if match:
                            count += 1
                return float(count)

            B = count_tmpl(m)
            A = count_tmpl(m + 1)
            if B > 0 and A > 0:
                out[i] = -np.log(A / B)
            elif B == 0:
                out[i] = np.nan
            else:
                out[i] = 0.0

        # Current policy: early rows are not backfilled. Windows without enough
        # history remain NaN and are imputed inside each fold.
        return out
else:
    _sampen_kernel = None


def sample_entropy_causal_cpu(series, window=24, m=2, r_factor=0.2):
    """
    Compute causal Sample Entropy from past/current values only.

    Current policy: early rows without enough history remain NaN. They are
    filled later by fold-level imputation values fitted on X_train only. This
    avoids the previous non-causal backfill with the first future valid SampEn.
    """
    vals = np.asarray(series, dtype=np.float64)
    n = len(vals)

    if USE_NUMBA and _sampen_kernel is not None:
        return _sampen_kernel(vals, n, window, m, r_factor).astype(np.float32)

    out = np.full(n, np.nan, dtype=np.float32)
    for i in range(m + 2, n):
        start = max(0, i - window + 1)
        win = vals[start : i + 1]
        win = win[~np.isnan(win)]
        if len(win) < m + 2:
            continue
        r = r_factor * np.std(win)
        if r == 0:
            out[i] = 0.0
            continue

        def count_tmpl(m_len):
            if len(win) <= m_len:
                return 0.0
            tmpl = np.lib.stride_tricks.sliding_window_view(win, m_len)
            diffs = np.abs(tmpl[:, None, :] - tmpl[None, :, :]).max(axis=-1)
            return float(np.sum(diffs <= r) - len(tmpl))

        B = count_tmpl(m)
        A = count_tmpl(m + 1)
        if B > 0 and A > 0:
            out[i] = -np.log(A / B)
        elif B == 0:
            out[i] = np.nan
        else:
            out[i] = 0.0

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Causal forward fill only; final fallback imputation is fold-local.
# ─────────────────────────────────────────────────────────────────────────────
def impute_causal_gpu(df):
    """
    Deprecated name retained for compatibility.

    Current policy: this function performs only patient-wise causal forward
    fill. It does not compute global fallback statistics. Remaining NaNs are
    intentionally preserved in feature caches and imputed inside each CV fold
    using training rows only.
    """
    meta = {"Patient_ID", "TimeStep", "SourceSet", "SepsisLabel"}
    num_cols = [
        c for c in df.columns
        if c not in meta and df[c].dtype.kind in ("f", "i", "u")
    ]
    logging.info(
        "Current-policy causal ffill only on GPU: %d numeric columns; remaining NaNs preserved",
        len(num_cols),
    )
    for col in num_cols:
        df[col] = df.groupby("Patient_ID")[col].ffill()
    return df


def impute_causal_pandas(df, candidate_cols=None):
    """
    Deprecated name retained for compatibility.

    Current policy: this function performs only patient-wise causal forward
    fill. It does not compute global fallback statistics. Remaining NaNs are
    intentionally preserved in feature caches and imputed inside each CV fold
    using training rows only.
    """
    excluded = {"Patient_ID", "TimeStep", "SourceSet", "SepsisLabel", "Measurement_Count"}
    source_cols = list(candidate_cols) if candidate_cols is not None else list(df.columns)
    num_cols = [
        c for c in source_cols
        if c in df.columns
        and c not in excluded
        and not c.endswith("_is_missing")
        and pd.api.types.is_numeric_dtype(df[c])
        and bool(df[c].isna().any())
    ]
    logging.info(
        "Current-policy causal ffill only in pandas: %d raw numeric columns with missing values; "
        "excluded meta/missingness/count columns remain unchanged; remaining NaNs preserved",
        len(num_cols),
    )
    flush_log_handlers()
    grouped = df.groupby("Patient_ID", sort=False)
    for i, col in enumerate(num_cols, start=1):
        logging.info("Causal ffill pandas %d/%d: %s", i, len(num_cols), col)
        flush_log_handlers()
        df[col] = grouped[col].ffill()
        if i % 8 == 0:
            gc.collect()
            log_mem(f"after causal ffill column {i}/{len(num_cols)}")
    gc.collect()
    log_mem("after causal ffill pandas complete")
    return df


def fit_fold_imputation_values(X_train, feature_names=None):
    """Fit fold-local fallback medians using only X_train rows."""
    X = np.asarray(X_train, dtype=np.float32)
    finite = np.where(np.isfinite(X), X, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        values = np.nanmedian(finite, axis=0).astype(np.float32)
    all_missing = ~np.isfinite(values)
    if np.any(all_missing):
        values[all_missing] = 0.0
        names = np.asarray(feature_names, dtype=object)[all_missing].tolist() if feature_names is not None else np.where(all_missing)[0].tolist()
        logging.warning(
            "Fold-local imputation: %d columns all NaN/non-finite in X_train; filling with 0.0. Columns: %s",
            int(np.sum(all_missing)), names[:50],
        )
    return values


def apply_fold_imputation_values(X, imputation_values, tag=""):
    """Apply fold-local imputation values and assert finite model inputs."""
    X_out = np.asarray(X, dtype=np.float32).copy()
    non_finite_mask = ~np.isfinite(X_out)
    before = int(non_finite_mask.sum())
    if before:
        rows, cols = np.where(non_finite_mask)
        X_out[rows, cols] = imputation_values[cols]
    after = int((~np.isfinite(X_out)).sum())
    logging.info(
        "Fold-local imputation %s: non-finite before=%s after=%s",
        tag, f"{before:,}", f"{after:,}",
    )
    if after:
        raise ValueError(f"Non-finite values remain after fold-local imputation ({tag}): {after}")
    return X_out


def fit_and_apply_fold_standardization(X_train, X_val, feature_names=None, fold=None):
    """
    Impute and standardize a fold without validation leakage.

    Medians, means, and standard deviations are fitted on imputed X_train only
    and then applied to both X_train and X_val.
    """
    fold_tag = f"fold {fold}" if fold is not None else "fold"
    logging.info("%s: fitting imputation/normalization on X_train only", fold_tag)
    impute_values = fit_fold_imputation_values(X_train, feature_names=feature_names)
    X_tr_imp = apply_fold_imputation_values(X_train, impute_values, tag=f"{fold_tag} train")
    X_val_imp = apply_fold_imputation_values(X_val, impute_values, tag=f"{fold_tag} val")

    mean_tr = X_tr_imp.mean(axis=0, dtype=np.float64).astype(np.float32)
    std_tr = X_tr_imp.std(axis=0, dtype=np.float64).astype(np.float32)
    std_tr[(std_tr == 0) | ~np.isfinite(std_tr)] = 1.0
    mean_tr[~np.isfinite(mean_tr)] = 0.0

    X_tr = ((X_tr_imp - mean_tr) / std_tr).astype(np.float32)
    X_val = ((X_val_imp - mean_tr) / std_tr).astype(np.float32)
    X_tr = apply_fold_imputation_values(X_tr, np.zeros(X_tr.shape[1], dtype=np.float32), tag=f"{fold_tag} train standardized")
    X_val = apply_fold_imputation_values(X_val, np.zeros(X_val.shape[1], dtype=np.float32), tag=f"{fold_tag} val standardized")
    return X_tr, X_val


def make_feature_cache_payload(df, context):
    return {
        "metadata": {
            "feature_cache_version": FEATURE_CACHE_VERSION,
            "pipeline_policy_version": PIPELINE_POLICY_VERSION,
            "imputation_policy": FOLD_IMPUTATION_POLICY,
            "sampen_early_policy": SAMPEN_EARLY_POLICY,
            "context": context,
        },
        "dataframe": df,
    }


def save_feature_cache(df, cache_file, context):
    logging.info(
        "Saving feature cache version=%s imputation_policy=%s sampen_policy=%s: %s",
        FEATURE_CACHE_VERSION, FOLD_IMPUTATION_POLICY, SAMPEN_EARLY_POLICY, cache_file,
    )
    joblib.dump(make_feature_cache_payload(df, context), cache_file, compress=3)


def load_feature_cache(cache_file, required_context=None):
    if not os.path.exists(cache_file):
        return None
    logging.info(f"Loading feature cache candidate: {cache_file}")
    obj = joblib.load(cache_file)
    if not isinstance(obj, dict) or "metadata" not in obj or "dataframe" not in obj:
        logging.warning(
            "Rejecting legacy feature cache without current-policy metadata: %s",
            cache_file,
        )
        return None
    metadata = obj.get("metadata", {})
    if metadata.get("feature_cache_version") != FEATURE_CACHE_VERSION:
        logging.warning(
            "Rejecting incompatible feature cache %s: version=%s expected=%s",
            cache_file, metadata.get("feature_cache_version"), FEATURE_CACHE_VERSION,
        )
        return None
    if metadata.get("imputation_policy") != FOLD_IMPUTATION_POLICY:
        logging.warning("Rejecting feature cache with incompatible imputation policy: %s", metadata)
        return None
    if metadata.get("sampen_early_policy") != SAMPEN_EARLY_POLICY:
        logging.warning("Rejecting feature cache with incompatible SampEn policy: %s", metadata)
        return None
    if required_context is not None and metadata.get("context") not in {required_context, "enhanced_partial"}:
        logging.warning("Feature cache context differs from requested context: %s", metadata)
    logging.info("Accepted feature cache metadata: %s", metadata)
    return obj["dataframe"]


def baseline_cache_path_for_enhanced(output_dir):
    """Infer the compatible current baseline cache path for enhanced reruns."""
    output_abs = os.path.abspath(output_dir.rstrip(os.sep))
    results_root = os.path.dirname(output_abs)
    output_name = os.path.basename(output_abs)
    if output_name.startswith("enhanced"):
        baseline_dir = "baseline" + output_name[len("enhanced"):]
    else:
        # Current-policy fallback for nonstandard enhanced output names; never use pre-fix baseline_psv_only.
        baseline_dir = "baseline"
    return os.path.join(
        results_root, baseline_dir, "precomputed_features_baseline_v2.joblib"
    )


def enhanced_partial_cache_path(output_dir):
    return os.path.join(output_dir, "precomputed_features_enhanced_v2.partial.joblib")


def validate_feature_cache_for_enhanced(df):
    required_meta = ["Patient_ID", "TimeStep", "SepsisLabel", "Age", "Gender"]
    missing_meta = [c for c in required_meta if c not in df.columns]
    if missing_meta:
        return False, f"missing required columns {missing_meta}"

    clinical_aliases = {
        "HeartRate": ["HeartRate", "HR"],
        "O2Sat": ["O2Sat"],
        "Temperature": ["Temperature", "Temp"],
        "SysBP": ["SysBP", "SBP"],
        "MeanBP": ["MeanBP", "MAP"],
        "DiaBP": ["DiaBP", "DBP", "DiasBP"],
        "RespRate": ["RespRate", "Resp"],
    }
    missing_clinical = []
    for canonical, aliases in clinical_aliases.items():
        if not any(alias in df.columns for alias in aliases):
            missing_clinical.append(canonical)
    if missing_clinical:
        return False, f"missing clinical columns {missing_clinical}"

    return True, "ok"


def expected_hemo_feature_columns(df):
    cols = []
    for col in HEMO_VARS:
        if col in df.columns:
            cols.extend([f"{col}_cv_8h", f"{col}_iqr_8h", f"{col}_sampen_24h"])
    return cols


def save_enhanced_partial(df, partial_cache_file, context):
    if partial_cache_file is None:
        return
    logging.info(f"Saving enhanced partial checkpoint ({context}): {partial_cache_file}")
    log_mem(f"before saving enhanced partial {context}")
    save_feature_cache(df, partial_cache_file, context="enhanced_partial")
    gc.collect()
    maybe_free_gpu()
    log_mem(f"after saving enhanced partial {context}")


def add_missing_hemodynamic_features_stable(df, partial_cache_file=None):
    """Add current-policy hemodynamic features with causal rolling definitions.

    CV uses rolling std / (abs(rolling mean) + 1e-8), IQR uses rolling
    Q75 - Q25, and SampEn uses a causal 24-hour window with early rows left
    missing until fold-local train-only imputation.
    """
    if not isinstance(df, pd.DataFrame):
        logging.info("Converting feature frame to pandas for stable hemodynamic features")
        df = df.to_pandas()
        maybe_free_gpu()
        gc.collect()

    expected = expected_hemo_feature_columns(df)
    existing = [c for c in expected if c in df.columns]
    missing = [c for c in expected if c not in df.columns]
    logging.info(
        f"Enhanced hemodynamic features expected={len(expected)}, "
        f"existing={len(existing)}, missing={len(missing)}"
    )
    logging.info(
        "current-policy hemodynamic definitions: CV_8h=rolling_std/(abs(rolling_mean)+1e-8); "
        "IQR_8h=rolling_Q75-rolling_Q25; SampEn_24h=causal window with early NaNs kept for fold-local imputation."
    )

    for idx, col in enumerate([c for c in HEMO_VARS if c in df.columns], start=1):
        target_cols = [f"{col}_cv_8h", f"{col}_iqr_8h", f"{col}_sampen_24h"]
        missing_for_col = [c for c in target_cols if c not in df.columns]
        if not missing_for_col:
            logging.info(f"Hemodynamic features for {col} already present; skipping")
            continue

        logging.info(
            f"Stable hemodynamic block {col} ({idx}/{len(HEMO_VARS)}), "
            f"missing={missing_for_col}"
        )
        log_mem(f"before stable hemo {col}")
        grp = df.groupby("Patient_ID", sort=False)[col]

        if f"{col}_cv_8h" in missing_for_col:
            roll_m = grp.transform(lambda x: x.rolling(8, min_periods=2).mean())
            roll_s = grp.transform(lambda x: x.rolling(8, min_periods=2).std())
            df[f"{col}_cv_8h"] = (roll_s / (roll_m.abs() + 1e-8)).astype("float32")
            del roll_m, roll_s

        if f"{col}_iqr_8h" in missing_for_col:
            q75 = grp.transform(lambda x: x.rolling(8, min_periods=2).quantile(0.75))
            q25 = grp.transform(lambda x: x.rolling(8, min_periods=2).quantile(0.25))
            df[f"{col}_iqr_8h"] = (q75 - q25).astype("float32")
            del q75, q25

        if f"{col}_sampen_24h" in missing_for_col:
            df[f"{col}_sampen_24h"] = (
                grp.transform(
                    lambda x: pd.Series(
                        sample_entropy_causal_cpu(x.values, window=W_HUGE),
                        index=x.index,
                    )
                )
                .astype("float32")
            )

        gc.collect()
        maybe_free_gpu()
        log_mem(f"after stable hemo {col}")
        save_enhanced_partial(df, partial_cache_file, f"after_{col}")

    final_missing = [c for c in expected_hemo_feature_columns(df) if c not in df.columns]
    if final_missing:
        logging.warning(f"Enhanced hemodynamic features still missing: {final_missing}")
    else:
        logging.info("All expected enhanced hemodynamic features are present")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Feature engineering principal
# ─────────────────────────────────────────────────────────────────────────────
def compute_all_features(df_pd, add_hemo, partial_cache_file=None):
    logging.info(
        "Feature engineering starts in pandas for missingness/causal ffill; "
        "baseline rolling uses cuDF when available; XGBoost training uses GPU."
    )
    logging.info(
        "Skipping early pandas->cuDF->pandas roundtrip before ffill; "
        "cuDF transfer is delayed until it feeds baseline rolling."
    )
    flush_log_handlers()
    t0 = time.time()
    use_cudf_features = False

    # Keep current-policy feature definitions unchanged while avoiding the full
    # pandas->cuDF->pandas copy before the first rolling feature.
    df = df_pd.copy()

    meta = ["Patient_ID", "TimeStep", "SepsisLabel"]
    clinical_base = [c for c in df.columns if c not in meta]
    raw_numeric_cols_for_ffill = [
        c for c in clinical_base
        if c != "SourceSet" and pd.api.types.is_numeric_dtype(df[c])
    ]

    missing_cols = [c for c in clinical_base if c != "Age"]
    for c in missing_cols:
        df[f"{c}_is_missing"] = df[c].isna().astype("int8")

    df["Measurement_Count"] = (
        df[missing_cols].notna().sum(axis=1).astype("int16")
    )
    logging.info("Flags de missingness listos")
    log_mem("after missingness")

    df = impute_causal_pandas(df, candidate_cols=raw_numeric_cols_for_ffill)
    logging.info("Imputación lista")

    base_watch = [
        "HeartRate", "O2Sat", "Temperature", "SysBP",
        "MeanBP", "DiaBP", "RespRate", "Age"
    ]
    log_feature_stats(
        df if not use_cudf_features else df.to_pandas(),
        base_watch,
        "after_imputation_base"
    )
    log_mem("after imputation")

    if USE_CUDF:
        logging.info(
            "Converting pandas feature frame to cuDF for baseline rolling only."
        )
        flush_log_handlers()
        log_mem("before pandas->cuDF baseline rolling transfer")
        df = cudf.DataFrame.from_pandas(df)
        use_cudf_features = True
        log_mem("after pandas->cuDF baseline rolling transfer")
        flush_log_handlers()

    for i, col in enumerate(SLIDING_VARS):
        if col not in df.columns:
            continue
        logging.info(f"Rolling features {col} ({i + 1}/{len(SLIDING_VARS)})...")
        flush_log_handlers()
        log_mem(f"before rolling {col}")

        for w, wlbl in [(W_SHORT, "5"), (W_LONG, "11")]:
            if use_cudf_features:
                df = add_rolling_features_cudf_safe(df, col, w, wlbl)
            else:
                df = add_rolling_features_pandas(df, col, w, wlbl)

        grp = df.groupby("Patient_ID")[col]
        df[f"{col}_last_obs"] = grp.ffill().astype("float32")
        df[f"{col}_diff_1"]   = grp.diff(1).astype("float32")
        df[f"{col}_diff_4"]   = grp.diff(4).astype("float32")

        if use_cudf_features:
            col_pd = df[["Patient_ID", col]].to_pandas()
            g = col_pd.groupby("Patient_ID")[col]

            energy = g.transform(
                lambda x: (x ** 2).expanding().mean()
            ).astype("float32")
            mean_diff_1 = g.transform(
                lambda x: x.diff(1).expanding().mean()
            ).astype("float32")
            shannon = g.transform(shannon_entropy_5h_cpu).astype("float32")

            df[f"{col}_energy"]      = cudf.Series(energy.values,     index=df.index)
            df[f"{col}_mean_diff_1"] = cudf.Series(mean_diff_1.values, index=df.index)
            df[f"{col}_shannon_5h"]  = cudf.Series(shannon.values,     index=df.index)
            del col_pd, g, energy, mean_diff_1, shannon
            gc.collect()
        else:
            df[f"{col}_energy"] = (
                grp.transform(lambda x: (x ** 2).expanding().mean()).astype("float32")
            )
            df[f"{col}_mean_diff_1"] = (
                grp.transform(lambda x: x.diff(1).expanding().mean()).astype("float32")
            )
            df[f"{col}_shannon_5h"] = (
                df.groupby("Patient_ID")[col]
                .transform(shannon_entropy_5h_cpu)
                .astype("float32")
            )

        log_mem(f"after rolling {col}")
        flush_log_handlers()
        gc.collect()
        maybe_free_gpu()

    logging.info("SLIDING_VARS rolling features completados")
    flush_log_handlers()

    if use_cudf_features:
        logging.info(
            "Transferring baseline rolling features back to pandas for stable "
            "hemo/CV/IQR/SampEn computation."
        )
        flush_log_handlers()
        log_mem("before cuDF->pandas after baseline rolling")
        df = df.to_pandas()
        use_cudf_features = False
        maybe_free_gpu()
        gc.collect()
        log_mem("after cuDF->pandas after baseline rolling")
        flush_log_handlers()

    if add_hemo:
        logging.info("Calculando features hemodinámicas con backend pandas estable...")
        df = add_missing_hemodynamic_features_stable(
            df, partial_cache_file=partial_cache_file
        )
        hemo_watch = []
        for col in HEMO_VARS:
            hemo_watch.extend([
                f"{col}_cv_8h",
                f"{col}_iqr_8h",
                f"{col}_sampen_24h",
            ])
        log_feature_stats(df, hemo_watch, "after_hemo_features")

    elapsed = time.time() - t0
    logging.info(f"Feature engineering completo en {elapsed / 60:.1f} min")

    if use_cudf_features:
        logging.info("Transfiriendo features de vuelta a CPU...")
        df_out = df.to_pandas()
        del df
        maybe_free_gpu()
        gc.collect()
        return df_out

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Fold assignments y checkpoints
# ─────────────────────────────────────────────────────────────────────────────
def build_or_load_fold_assignments(df, data_dir, n_splits=5):
    fold_file = os.path.join(
        data_dir, f"patient_fold_assignments_{n_splits}cv.csv"
    )
    if os.path.exists(fold_file):
        fold_df = pd.read_csv(fold_file, dtype={"Patient_ID": str})
        return dict(zip(fold_df["Patient_ID"], fold_df["Fold"]))

    patients  = df["Patient_ID"].astype(str).unique()
    y_patient = df.groupby("Patient_ID")["SepsisLabel"].max().reindex(patients).values
    skf       = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED)

    rows = []
    for fold, (_, val_idx) in enumerate(
        skf.split(patients, y_patient, groups=patients), start=1
    ):
        for pid in patients[val_idx]:
            rows.append({"Patient_ID": pid, "Fold": fold})

    fold_df = (
        pd.DataFrame(rows)
        .sort_values(["Fold", "Patient_ID"])
        .reset_index(drop=True)
    )
    fold_df.to_csv(fold_file, index=False)
    return dict(zip(fold_df["Patient_ID"], fold_df["Fold"]))


def get_fold_artifact_paths(output_dir, fold):
    return {
        "model":      os.path.join(output_dir, f"fold{fold}_model.joblib"),
        "learning":   os.path.join(output_dir, f"fold{fold}_learning.png"),
        "importance": os.path.join(output_dir, f"fold{fold}_feature_importance.csv"),
        "oof":        os.path.join(output_dir, f"fold{fold}_oof.csv"),
        "metrics":    os.path.join(output_dir, f"fold{fold}_metrics.json"),
    }


def save_fold_checkpoint(output_dir, fold, fold_oof, fold_metric_row):
    paths = get_fold_artifact_paths(output_dir, fold)
    fold_oof.to_csv(paths["oof"], index=False)
    with open(paths["metrics"], "w") as f:
        json.dump(fold_metric_row, f, indent=2)


def fold_oof_has_current_policy(fold_oof, fold=None):
    """Return True only for OOF artifacts generated by the current policy."""
    if "PipelineVersion" not in fold_oof.columns:
        return False
    versions = set(fold_oof["PipelineVersion"].astype(str).unique())
    if versions != {PIPELINE_POLICY_VERSION}:
        return False
    if fold is not None:
        observed_folds = set(pd.to_numeric(fold_oof["Fold"], errors="coerce").dropna().astype(int))
        if observed_folds != {int(fold)}:
            return False
    return True


def fold_metrics_has_current_policy(fold_metric_row):
    """Return True when fold metrics carry current current-policy metadata."""
    return fold_metric_row.get("pipeline_policy_version") == PIPELINE_POLICY_VERSION


def try_load_completed_fold(output_dir, fold):
    paths = get_fold_artifact_paths(output_dir, fold)
    if os.path.exists(paths["oof"]) and os.path.exists(paths["metrics"]):
        fold_oof = pd.read_csv(paths["oof"], dtype={"Patient_ID": str})
        with open(paths["metrics"]) as f:
            fold_metric_row = json.load(f)
        if not fold_oof_has_current_policy(fold_oof, fold=fold) or not fold_metrics_has_current_policy(fold_metric_row):
            logging.warning(
                "Ignoring stale fold %s artifacts without current policy metadata (%s). "
                "They will be overwritten only if training is run.",
                fold, PIPELINE_POLICY_VERSION,
            )
            return {"status": "stale", "paths": paths}
        imp_df = None
        if os.path.exists(paths["importance"]):
            imp_df = pd.read_csv(paths["importance"])
        return {
            "status": "complete",
            "fold_oof": fold_oof,
            "fold_metric_row": fold_metric_row,
            "importance_df": imp_df,
            "paths": paths,
        }
    if os.path.exists(paths["model"]):
        logging.warning(
            "Ignoring model-only fold %s artifact because policy metadata cannot be verified: %s",
            fold, paths["model"],
        )
        return {"status": "stale", "paths": paths}
    return {"status": "missing", "paths": paths}


def load_existing_oof_if_complete(output_dir, expected_folds=5):
    required_cols = {"Patient_ID", "TimeStep", "SepsisLabel", "Fold", "prob_raw", "PipelineVersion"}
    oof_frames = []
    detected_files = []
    rows_by_fold = {}

    log_mem("before loading existing OOF folds")
    for fold in range(1, expected_folds + 1):
        path = get_fold_artifact_paths(output_dir, fold)["oof"]
        if not os.path.exists(path):
            logging.info(f"Existing OOF resume disabled: missing {path}")
            return None

        fold_oof = pd.read_csv(path, dtype={"Patient_ID": str})
        missing_cols = required_cols - set(fold_oof.columns)
        if missing_cols:
            logging.info(
                f"Existing OOF resume disabled: {path} missing columns "
                f"{sorted(missing_cols)}"
            )
            return None

        if not fold_oof_has_current_policy(fold_oof, fold=fold):
            logging.info(
                f"Existing OOF resume disabled: {path} lacks current PipelineVersion "
                f"{PIPELINE_POLICY_VERSION}"
            )
            return None

        detected_files.append(path)
        rows_by_fold[fold] = int(len(fold_oof))
        oof_frames.append(fold_oof)

    oof_df = pd.concat(oof_frames, axis=0, ignore_index=True)
    folds_present = set(pd.to_numeric(oof_df["Fold"], errors="coerce").dropna().astype(int))
    expected = set(range(1, expected_folds + 1))
    if folds_present != expected:
        logging.info(
            f"Existing OOF resume disabled: folds present {sorted(folds_present)} "
            f"!= expected {sorted(expected)}"
        )
        return None

    oof_df = oof_df.sort_values(["Fold", "Patient_ID", "TimeStep"]).reset_index(drop=True)
    logging.info(f"Existing OOF files detected: {detected_files}")
    logging.info(f"Existing OOF rows by fold: {rows_by_fold}")
    logging.info(f"Existing OOF total rows: {len(oof_df):,}")
    logging.info(f"Existing OOF unique patients: {oof_df['Patient_ID'].nunique():,}")
    logging.info(
        "Existing OOF SepsisLabel prevalence: %.4f",
        float(oof_df["SepsisLabel"].mean()),
    )
    logging.info(
        "Existing complete OOF folds detected; skipping model training and resuming post-processing."
    )
    log_mem("after loading existing OOF folds")
    return oof_df


def load_existing_fold_metrics(output_dir, oof_df, expected_folds=5):
    rows = []
    for fold in range(1, expected_folds + 1):
        path = get_fold_artifact_paths(output_dir, fold)["metrics"]
        if os.path.exists(path):
            with open(path) as f:
                row = json.load(f)
            rows.append(row)
            continue

        fold_df = oof_df[oof_df["Fold"].astype(int) == fold]
        y_true = fold_df["SepsisLabel"].values.astype(np.int8)
        y_prob = fold_df["prob_raw"].values.astype(np.float32)
        rows.append({
            "Fold": int(fold),
            "AUROC": float(roc_auc_score(y_true, y_prob)),
            "AUPRC": float(average_precision_score(y_true, y_prob)),
        })
        logging.warning(f"Fold metrics file missing for fold {fold}; recomputed AUROC/AUPRC from OOF.")
    return rows


def load_existing_importances(output_dir, expected_folds=5):
    frames = []
    for fold in range(1, expected_folds + 1):
        path = get_fold_artifact_paths(output_dir, fold)["importance"]
        if os.path.exists(path):
            frames.append(pd.read_csv(path))
        else:
            logging.warning(f"Feature importance file missing for fold {fold}: {path}")
    return frames


# ─────────────────────────────────────────────────────────────────────────────
# GPU metrics
# ─────────────────────────────────────────────────────────────────────────────
def roc_auc_gpu(y_true, y_score):
    if cp is None:
        return roc_auc_score(np.asarray(y_true), np.asarray(y_score))

    y_true  = cp.asarray(y_true,  dtype=cp.int32)
    y_score = cp.asarray(y_score, dtype=cp.float32)

    n_pos = int(cp.sum(y_true).get())
    n_neg = int((y_true.size - n_pos))
    if n_pos == 0 or n_neg == 0:
        return np.nan

    order  = cp.argsort(-y_score)
    y_true = y_true[order]
    y_score = y_score[order]

    dv_idx = cp.where(cp.diff(y_score) != 0)[0]
    th_idx = cp.concatenate(
        [dv_idx, cp.array([y_true.size - 1], dtype=cp.int32)]
    )

    tps = cp.cumsum(y_true)[th_idx].astype(cp.float32)
    fps = (1 + th_idx - tps).astype(cp.float32)

    tpr = cp.concatenate([
        cp.array([0.0], dtype=cp.float32),
        tps / max(n_pos, 1),
        cp.array([1.0], dtype=cp.float32),
    ])
    fpr = cp.concatenate([
        cp.array([0.0], dtype=cp.float32),
        fps / max(n_neg, 1),
        cp.array([1.0], dtype=cp.float32),
    ])

    return float(cp.trapz(tpr, fpr).get())


def average_precision_gpu(y_true, y_score):
    if cp is None:
        return average_precision_score(np.asarray(y_true), np.asarray(y_score))

    y_true  = cp.asarray(y_true,  dtype=cp.int32)
    y_score = cp.asarray(y_score, dtype=cp.float32)

    n_pos = int(cp.sum(y_true).get())
    if n_pos == 0:
        return np.nan

    order  = cp.argsort(-y_score)
    y_true = y_true[order]

    tp     = cp.cumsum(y_true).astype(cp.float32)
    ranks  = cp.arange(1, y_true.size + 1, dtype=cp.float32)
    prec   = tp / ranks
    ap     = cp.sum(prec * y_true) / float(n_pos)
    return float(ap.get())


def binary_metrics_at_threshold_gpu(y_true, y_prob, threshold):
    if cp is None or not isinstance(y_true, cp.ndarray) or not isinstance(y_prob, cp.ndarray):
        y_true = np.asarray(y_true).astype(int)
        y_prob = np.asarray(y_prob).astype(float)
        y_hat  = (y_prob >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_hat, labels=[0, 1]).ravel()
        precision   = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall      = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        npv         = tn / (tn + fn) if (tn + fn) > 0 else 0.0
        f1          = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0 else 0.0
        )
        return {
            "threshold": float(threshold),
            "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
            "Precision": float(precision), "Recall": float(recall),
            "Specificity": float(specificity), "NPV": float(npv),
            "F1": float(f1),
        }

    y_hat = (y_prob >= threshold).astype(cp.int32)
    tp = int(cp.sum((y_true == 1) & (y_hat == 1)).get())
    tn = int(cp.sum((y_true == 0) & (y_hat == 0)).get())
    fp = int(cp.sum((y_true == 0) & (y_hat == 1)).get())
    fn = int(cp.sum((y_true == 1) & (y_hat == 0)).get())

    precision   = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall      = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    npv         = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    f1          = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0 else 0.0
    )

    return {
        "threshold": float(threshold),
        "TP": tp, "TN": tn, "FP": fp, "FN": fn,
        "Precision": float(precision), "Recall": float(recall),
        "Specificity": float(specificity), "NPV": float(npv),
        "F1": float(f1),
    }


def binary_metrics_at_threshold_cpu(y_true, y_prob, threshold):
    """Low-memory CPU threshold metrics for post-processing sweeps."""
    y_true = np.asarray(y_true, dtype=np.int8)
    y_prob = np.asarray(y_prob, dtype=np.float32)
    y_hat = y_prob >= threshold
    pos = y_true == 1
    neg = ~pos
    tp = int(np.sum(pos & y_hat))
    tn = int(np.sum(neg & ~y_hat))
    fp = int(np.sum(neg & y_hat))
    fn = int(np.sum(pos & ~y_hat))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        "threshold": float(threshold),
        "TP": tp, "TN": tn, "FP": fp, "FN": fn,
        "Precision": float(precision), "Recall": float(recall),
        "Specificity": float(specificity), "NPV": float(npv),
        "F1": float(f1),
    }


def calibration_metrics_quantile(y_true, y_prob, n_bins=10):
    if cp is None:
        y_true = np.asarray(y_true).astype(int)
        y_prob = np.asarray(y_prob).astype(np.float64)
        order  = np.argsort(y_prob)
        splits = [idx for idx in np.array_split(order, n_bins) if len(idx) > 0]

        rows, ece = [], 0.0
        for i, idx in enumerate(splits, start=1):
            frac_pos  = float(np.mean(y_true[idx]))
            mean_pred = float(np.mean(y_prob[idx]))
            weight    = len(idx) / len(y_true)
            ece      += abs(frac_pos - mean_pred) * weight
            rows.append({"bin": i, "count": int(len(idx)),
                         "mean_pred": mean_pred, "frac_pos": frac_pos})

        brier = float(brier_score_loss(y_true, y_prob))
        return float(ece), brier, pd.DataFrame(rows)

    y_true_cp = cp.asarray(y_true, dtype=cp.int32)
    y_prob_cp = cp.asarray(y_prob, dtype=cp.float32)
    order     = cp.argsort(y_prob_cp)
    n         = int(y_true_cp.size)

    splits, start = [], 0
    for i in range(n_bins):
        end = int(round((i + 1) * n / n_bins))
        if end > start:
            splits.append(order[start:end])
        start = end

    rows, ece = [], 0.0
    for i, idx in enumerate(splits, start=1):
        frac_pos  = float(cp.mean(y_true_cp[idx]).get())
        mean_pred = float(cp.mean(y_prob_cp[idx]).get())
        weight    = len(idx) / n
        ece      += abs(frac_pos - mean_pred) * weight
        rows.append({"bin": i, "count": int(len(idx)),
                     "mean_pred": mean_pred, "frac_pos": frac_pos})

    brier = float(
        cp.mean((y_prob_cp - y_true_cp.astype(cp.float32)) ** 2).get()
    )
    return float(ece), brier, pd.DataFrame(rows)


def crossfit_calibration(y_true, y_prob, fold_ids, method="platt"):
    y_true   = np.asarray(y_true).astype(int)
    y_prob   = np.asarray(y_prob).astype(np.float32)
    fold_ids = np.asarray(fold_ids)
    calibrated = np.zeros_like(y_prob, dtype=np.float32)

    for fold in np.unique(fold_ids):
        log_step(f"Calibración {method}: fold {fold}")
        tr = fold_ids != fold
        te = fold_ids == fold
        x_tr, y_tr, x_te = y_prob[tr], y_true[tr], y_prob[te]

        if len(np.unique(y_tr)) < 2:
            calibrated[te] = x_te
            continue

        if method == "platt":
            if USE_CUML:
                try:
                    model = CuLogisticRegression(max_iter=1000)
                    Xtr = cp.asarray(x_tr, dtype=cp.float32).reshape(-1, 1)
                    Ytr = cp.asarray(y_tr, dtype=cp.int32)
                    Xte = cp.asarray(x_te, dtype=cp.float32).reshape(-1, 1)
                    model.fit(Xtr, Ytr)
                    out = model.predict_proba(Xte)
                    if hasattr(out, "to_output"):
                        out = out.to_output("cupy")
                    elif hasattr(out, "values"):
                        out = cp.asarray(out.values)
                    else:
                        out = cp.asarray(out)
                    calibrated[te] = cp.asnumpy(out[:, 1]).astype(np.float32)
                    del model, Xtr, Ytr, Xte, out
                    maybe_free_gpu()
                    gc.collect()
                    continue
                except Exception as e:
                    logging.warning(
                        f"cuML Platt falló en fold {fold}, fallback sklearn: {e}"
                    )
            model = SkLogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
            model.fit(x_tr.reshape(-1, 1), y_tr)
            calibrated[te] = (
                model.predict_proba(x_te.reshape(-1, 1))[:, 1].astype(np.float32)
            )

        elif method == "isotonic":
            model = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
            model.fit(x_tr, y_tr)
            calibrated[te] = model.predict(x_te).astype(np.float32)

        else:
            raise ValueError(f"Unknown calibration method: {method}")

    return calibrated.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Utility — CORRECCIÓN: vectorizado con NumPy, elimina bucle Python por fila
# ─────────────────────────────────────────────────────────────────────────────
def compute_prediction_utility(
    labels, predictions,
    dt_early=-12, dt_optimal=-6, dt_late=3,
    max_u_tp=1.0, min_u_fn=-2.0, u_fp=-0.05, u_tn=0.0
):
    """
    CORRECCIÓN: el original usaba un bucle Python O(n).
    Esta versión usa operaciones NumPy vectorizadas ~10–50× más rápidas.
    """
    labels      = np.asarray(labels).astype(int)
    predictions = np.asarray(predictions).astype(int)

    is_septic = bool(np.any(labels > 0))
    t_sepsis  = int(np.argmax(labels)) if is_septic else -1

    m1 = max_u_tp / float(dt_optimal - dt_early)
    b1 = -m1 * dt_early
    m2 = -max_u_tp / float(dt_late - dt_optimal)
    b2 = -m2 * dt_late
    m3 = min_u_fn / float(dt_late - dt_optimal)
    b3 = -m3 * dt_optimal

    n  = len(labels)
    ts = np.arange(n)
    utility = np.zeros(n, dtype=np.float64)

    if is_septic:
        dt = ts - t_sepsis
        in_window = dt <= dt_late  # Rows where septic-window logic applies.

        # Positivos predichos dentro de ventana
        pred_pos_win = predictions.astype(bool) & in_window
        # -- zona early (<dt_early)
        early_zone   = pred_pos_win & (dt < dt_early)
        utility[early_zone] = u_fp
        # -- zona óptima (dt_early <= dt <= dt_optimal)
        optimal_zone = pred_pos_win & (dt >= dt_early) & (dt <= dt_optimal)
        utility[optimal_zone] = m1 * dt[optimal_zone] + b1
        # -- zona tardía (dt_optimal < dt <= dt_late)
        late_zone_tp = pred_pos_win & (dt > dt_optimal) & (dt <= dt_late)
        utility[late_zone_tp] = m2 * dt[late_zone_tp] + b2

        # Negativos predichos dentro de ventana
        pred_neg_win = (~predictions.astype(bool)) & in_window
        # -- antes del óptimo: 0
        # -- después del óptimo
        late_neg = pred_neg_win & (dt > dt_optimal)
        utility[late_neg] = m3 * dt[late_neg] + b3

        # Fuera de ventana (no séptico para ese instante)
        out_win = ~in_window
        utility[out_win & predictions.astype(bool)] = u_fp
        # utility[out_win & ~predictions.astype(bool)] = u_tn = 0.0
    else:
        utility[predictions.astype(bool)] = u_fp
        # utility[~predictions.astype(bool)] = u_tn = 0.0

    return float(utility.sum())


def best_prediction_vector(labels, dt_early=-12, dt_late=3):
    labels = np.asarray(labels).astype(int)
    preds  = np.zeros_like(labels)
    if np.any(labels > 0):
        t_sepsis = int(np.argmax(labels))
        start    = max(0, t_sepsis + dt_early)
        end      = min(len(labels), t_sepsis + dt_late + 1)
        preds[start:end] = 1
    return preds.astype(int)


def prepare_patient_segments(oof_df):
    oof_sorted = oof_df.sort_values(
        ["Patient_ID", "TimeStep"]
    ).reset_index(drop=True)
    pid = oof_sorted["Patient_ID"].astype(str).to_numpy()

    if len(pid) <= 1:
        starts = np.array([0], dtype=int) if len(pid) == 1 else np.array([], dtype=int)
        ends   = np.array([1], dtype=int) if len(pid) == 1 else np.array([], dtype=int)
        return oof_sorted, starts, ends

    boundaries = np.flatnonzero(pid[1:] != pid[:-1]) + 1
    starts = np.concatenate([[0], boundaries])
    ends   = np.concatenate([boundaries, [len(pid)]])
    return oof_sorted, starts, ends


def precompute_utility_components(labels_sorted, starts, ends):
    """Precompute constant and row-level utility terms once for threshold sweep.

    The mathematical Utility Score definition is unchanged. For each row we
    compute the utility contribution if the thresholded prediction is positive
    and if it is negative. A threshold then only selects between those two
    precomputed values, avoiding millions of per-patient array allocations.
    """
    labels_sorted = np.asarray(labels_sorted, dtype=np.int8)
    n_rows = len(labels_sorted)
    best = np.empty(len(starts), dtype=np.float64)
    inaction = np.empty(len(starts), dtype=np.float64)
    worst = np.empty(len(starts), dtype=np.float64)
    pos_value = np.zeros(n_rows, dtype=np.float64)
    neg_value = np.zeros(n_rows, dtype=np.float64)

    dt_early, dt_optimal, dt_late = -12, -6, 3
    max_u_tp, min_u_fn, u_fp = 1.0, -2.0, -0.05
    m1 = max_u_tp / float(dt_optimal - dt_early)
    b1 = -m1 * dt_early
    m2 = -max_u_tp / float(dt_late - dt_optimal)
    b2 = -m2 * dt_late
    m3 = min_u_fn / float(dt_late - dt_optimal)
    b3 = -m3 * dt_optimal

    for i, (s, e) in enumerate(zip(starts, ends)):
        labels = labels_sorted[s:e]
        best[i] = compute_prediction_utility(labels, best_prediction_vector(labels))
        inaction[i] = compute_prediction_utility(labels, np.zeros_like(labels))
        all_pos = compute_prediction_utility(labels, np.ones_like(labels))
        worst[i] = min(inaction[i], all_pos)

        n = e - s
        if np.any(labels > 0):
            t_sepsis = int(np.argmax(labels))
            dt = np.arange(n, dtype=np.int32) - t_sepsis
            in_window = dt <= dt_late

            pos = np.zeros(n, dtype=np.float64)
            early_zone = in_window & (dt < dt_early)
            optimal_zone = in_window & (dt >= dt_early) & (dt <= dt_optimal)
            late_zone_tp = in_window & (dt > dt_optimal) & (dt <= dt_late)
            out_win = ~in_window
            pos[early_zone] = u_fp
            pos[optimal_zone] = m1 * dt[optimal_zone] + b1
            pos[late_zone_tp] = m2 * dt[late_zone_tp] + b2
            pos[out_win] = u_fp

            neg = np.zeros(n, dtype=np.float64)
            late_neg = in_window & (dt > dt_optimal)
            neg[late_neg] = m3 * dt[late_neg] + b3
        else:
            pos = np.full(n, u_fp, dtype=np.float64)
            neg = np.zeros(n, dtype=np.float64)

        pos_value[s:e] = pos
        neg_value[s:e] = neg

    return {
        "best": best,
        "inaction": inaction,
        "worst": worst,
        "pos_value": pos_value,
        "neg_value": neg_value,
        "delta_value": pos_value - neg_value,
        "total_best": float(np.sum(best)),
        "total_inaction": float(np.sum(inaction)),
        "total_worst": float(np.sum(worst)),
    }


def normalized_utility_from_sorted_arrays(
    labels_sorted, probs_sorted, starts, ends, threshold, utility_components=None
):
    """Compute normalized PhysioNet utility with compact vectorized row terms."""
    if utility_components is None:
        utility_components = precompute_utility_components(labels_sorted, starts, ends)

    probs_sorted = np.asarray(probs_sorted, dtype=np.float32)
    if "delta_value" in utility_components and "neg_value" in utility_components:
        mask = probs_sorted >= threshold
        total_observed = float(
            utility_components["total_inaction"]
            + np.sum(utility_components["delta_value"][mask], dtype=np.float64)
        )
    else:
        preds_all = (probs_sorted >= threshold).astype(np.int8)
        total_observed = 0.0
        for s, e in zip(starts, ends):
            total_observed += compute_prediction_utility(labels_sorted[s:e], preds_all[s:e])

    total_best = float(utility_components.get("total_best", np.sum(utility_components["best"])))
    total_inaction = float(utility_components.get("total_inaction", np.sum(utility_components["inaction"])))
    total_worst = float(utility_components.get("total_worst", np.sum(utility_components["worst"])))

    if total_observed >= total_inaction:
        denom = total_best - total_inaction
        return 0.0 if denom == 0.0 else float((total_observed - total_inaction) / denom)
    denom = total_inaction - total_worst
    return 0.0 if denom == 0.0 else float((total_observed - total_inaction) / denom)


def patient_level_metrics_sorted(
    labels_sorted, probs_sorted, starts, ends, threshold=0.5
):
    leads = []
    tp_patients = fn_patients = tn_patients = fp_patients = 0

    for s, e in zip(starts, ends):
        y_true = labels_sorted[s:e]
        probs  = probs_sorted[s:e]
        septic = bool(np.any(y_true == 1))

        if septic:
            first_onset = int(np.argmax(y_true == 1))
            alerts_pre  = np.where(probs[:first_onset] >= threshold)[0]
            if len(alerts_pre) > 0:
                leads.append((first_onset - alerts_pre[0]) + 6)
                tp_patients += 1
            else:
                fn_patients += 1
        else:
            alerts_all = np.where(probs >= threshold)[0]
            if len(alerts_all) == 0:
                tn_patients += 1
            else:
                fp_patients += 1

    sensitivity = (
        tp_patients / (tp_patients + fn_patients)
        if (tp_patients + fn_patients) > 0 else np.nan
    )
    specificity = (
        tn_patients / (tn_patients + fp_patients)
        if (tn_patients + fp_patients) > 0 else np.nan
    )
    median_lead = np.median(leads) if leads else np.nan
    q25, q75    = np.percentile(leads, [25, 75]) if leads else (np.nan, np.nan)

    return {
        "patient_sensitivity":  float(sensitivity) if not np.isnan(sensitivity) else np.nan,
        "patient_specificity":  float(specificity) if not np.isnan(specificity) else np.nan,
        "lead_time_median":     float(median_lead) if not np.isnan(median_lead) else np.nan,
        "lead_time_iqr_lower":  float(q25) if not np.isnan(q25) else np.nan,
        "lead_time_iqr_upper":  float(q75) if not np.isnan(q75) else np.nan,
        "tp_patients": int(tp_patients),
        "fn_patients": int(fn_patients),
        "tn_patients": int(tn_patients),
        "fp_patients": int(fp_patients),
    }


def log_feature_stats(df, cols, tag, max_cols=30):
    cols = [c for c in cols if c in df.columns]
    if not cols:
        logging.info(f"[STATS] {tag}: no se encontraron columnas")
        return
    if len(cols) > max_cols:
        logging.info(f"[STATS] {tag}: mostrando {max_cols}/{len(cols)} columnas")
        cols = cols[:max_cols]

    rows = []
    for c in cols:
        s = pd.to_numeric(df[c], errors="coerce")
        rows.append({
            "feature":   c,
            "n":         int(len(s)),
            "nan_count": int(s.isna().sum()),
            "nan_pct":   float(100.0 * s.isna().mean()),
            "mean":      float(np.nanmean(s.values)) if np.isfinite(np.nanmean(s.values)) else np.nan,
            "std":       float(np.nanstd(s.values))  if np.isfinite(np.nanstd(s.values))  else np.nan,
            "min":       float(np.nanmin(s.values))  if np.isfinite(np.nanmin(s.values))  else np.nan,
            "max":       float(np.nanmax(s.values))  if np.isfinite(np.nanmax(s.values))  else np.nan,
            "n_unique":  int(s.nunique(dropna=True)),
        })

    stats_df = pd.DataFrame(rows)
    logging.info(f"\n[STATS] {tag}\n{stats_df.to_string(index=False)}")


# ─────────────────────────────────────────────────────────────────────────────
# Bootstrap / tests estadísticos
# ─────────────────────────────────────────────────────────────────────────────
def bootstrap_ci_patient(y_true, y_pred, patient_ids, n_bootstrap=200, ci=95):
    log_step(f"Bootstrap, n_bootstrap={n_bootstrap}")

    y_true_np      = np.asarray(y_true).astype(np.int32)
    y_pred_np      = np.asarray(y_pred).astype(np.float32)
    patient_ids_np = np.asarray(patient_ids).astype(str)

    patients, inverse = np.unique(patient_ids_np, return_inverse=True)
    patient_row_idx   = [
        np.where(inverse == i)[0].astype(np.int32) for i in range(len(patients))
    ]

    if cp is not None:
        y_true_cp = cp.asarray(y_true_np, dtype=cp.int32)
        y_pred_cp = cp.asarray(y_pred_np, dtype=cp.float32)
    else:
        y_true_cp = y_pred_cp = None

    rng = np.random.RandomState(SEED)
    aucs, auprcs = [], []

    for i in range(n_bootstrap):
        if (i + 1) % 25 == 0:
            logging.info(f"Bootstrap {i + 1}/{n_bootstrap}")
            log_mem(f"bootstrap iter {i + 1}")

        sampled = rng.randint(0, len(patients), size=len(patients))
        idx_np  = np.concatenate([patient_row_idx[j] for j in sampled])

        if len(np.unique(y_true_np[idx_np])) < 2:
            continue

        if cp is not None:
            idx_cp = cp.asarray(idx_np, dtype=cp.int32)
            aucs.append(roc_auc_gpu(y_true_cp[idx_cp], y_pred_cp[idx_cp]))
            auprcs.append(average_precision_gpu(y_true_cp[idx_cp], y_pred_cp[idx_cp]))
            del idx_cp
        else:
            aucs.append(roc_auc_score(y_true_np[idx_np], y_pred_np[idx_np]))
            auprcs.append(average_precision_score(y_true_np[idx_np], y_pred_np[idx_np]))

    lo       = (100 - ci) / 2
    auc_ci   = (np.percentile(aucs, lo), np.percentile(aucs, 100 - lo))
    auprc_lo = np.percentile(auprcs, lo)
    auprc_hi = np.percentile(auprcs, 100 - lo)

    if cp is not None:
        del y_true_cp, y_pred_cp
        maybe_free_gpu()
    del patient_row_idx
    gc.collect()

    return auc_ci, auprc_lo, auprc_hi


def compute_midrank(x):
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=np.float64)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out       = np.empty(N, dtype=np.float64)
    out[J]    = T
    return out


def fast_delong(predictions_sorted_transposed, label_1_count):
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    k = predictions_sorted_transposed.shape[0]

    pos_examples = predictions_sorted_transposed[:, :m]
    neg_examples = predictions_sorted_transposed[:, m:]

    tx = np.empty((k, m), dtype=np.float64)
    ty = np.empty((k, n), dtype=np.float64)
    tz = np.empty((k, m + n), dtype=np.float64)

    for r in range(k):
        tx[r, :] = compute_midrank(pos_examples[r, :])
        ty[r, :] = compute_midrank(neg_examples[r, :])
        tz[r, :] = compute_midrank(predictions_sorted_transposed[r, :])

    aucs      = tz[:, :m].sum(axis=1) / (m * n) - (m + 1.0) / (2.0 * n)
    v01       = (tz[:, :m] - tx) / n
    v10       = 1.0 - (tz[:, m:] - ty) / m
    sx        = np.cov(v01)
    sy        = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov


def delong_roc_test(y_true, pred_a, pred_b):
    y_true = np.asarray(y_true).astype(int)
    pred_a = np.asarray(pred_a).astype(np.float64)
    pred_b = np.asarray(pred_b).astype(np.float64)

    if len(np.unique(y_true)) < 2:
        return np.nan

    order        = np.argsort(-y_true)
    label_1_count = int(y_true.sum())
    preds        = np.vstack([pred_a, pred_b])[:, order]
    aucs, cov    = fast_delong(preds, label_1_count)

    if np.ndim(cov) == 0:
        return np.nan

    l      = np.array([[1, -1]], dtype=np.float64)
    z_num  = np.abs(np.diff(aucs))[0]
    z_den  = np.sqrt(np.dot(np.dot(l, cov), l.T))[0, 0]
    if z_den <= 0:
        return np.nan

    z = z_num / z_den
    return float(2 * norm.sf(z))


def paired_bootstrap_compare(
    y_true, pred_ref, pred_new, patient_ids,
    metric_name="auroc", n_bootstrap=200
):
    y_true      = np.asarray(y_true).astype(np.int32)
    pred_ref    = np.asarray(pred_ref).astype(np.float32)
    pred_new    = np.asarray(pred_new).astype(np.float32)
    patient_ids = np.asarray(patient_ids).astype(str)

    patients, inverse = np.unique(patient_ids, return_inverse=True)
    patient_row_idx   = [
        np.where(inverse == i)[0].astype(np.int32) for i in range(len(patients))
    ]
    rng   = np.random.RandomState(SEED)
    diffs = []

    for i in range(n_bootstrap):
        if (i + 1) % 25 == 0:
            logging.info(f"Bootstrap pareado {metric_name}: {i + 1}/{n_bootstrap}")

        sampled = rng.randint(0, len(patients), size=len(patients))
        idx     = np.concatenate([patient_row_idx[j] for j in sampled])

        if len(np.unique(y_true[idx])) < 2:
            continue

        if metric_name == "auroc":
            ref_metric = roc_auc_score(y_true[idx], pred_ref[idx])
            new_metric = roc_auc_score(y_true[idx], pred_new[idx])
        else:
            ref_metric = average_precision_score(y_true[idx], pred_ref[idx])
            new_metric = average_precision_score(y_true[idx], pred_new[idx])

        diffs.append(new_metric - ref_metric)

    diffs = np.array(diffs, dtype=np.float64)
    if len(diffs) == 0:
        return {
            "delta": np.nan, "ci_lower": np.nan,
            "ci_upper": np.nan, "p_value": np.nan
        }

    lo, hi       = np.percentile(diffs, [2.5, 97.5])
    p_two_sided  = 2 * min(np.mean(diffs <= 0), np.mean(diffs >= 0))
    return {
        "delta":    float(np.mean(diffs)),
        "ci_lower": float(lo),
        "ci_upper": float(hi),
        "p_value":  float(min(p_two_sided, 1.0)),
    }


def mcnemar_test(y_true, pred_ref_bin, pred_new_bin):
    y_true       = np.asarray(y_true).astype(int)
    pred_ref_bin = np.asarray(pred_ref_bin).astype(int)
    pred_new_bin = np.asarray(pred_new_bin).astype(int)

    correct_ref = pred_ref_bin == y_true
    correct_new = pred_new_bin == y_true
    b = int(np.sum(correct_ref & (~correct_new)))
    c = int(np.sum((~correct_ref) & correct_new))

    if b + c == 0:
        return {"b": b, "c": c, "statistic": 0.0, "p_value": 1.0}

    stat    = (abs(b - c) - 1.0) ** 2 / (b + c)
    p_value = float(chi2.sf(stat, 1))
    return {"b": b, "c": c, "statistic": float(stat), "p_value": p_value}


def benjamini_hochberg(pvals_dict):
    items = [(k, v) for k, v in pvals_dict.items() if pd.notna(v)]
    if not items:
        return {}
    items.sort(key=lambda x: x[1])
    m, adjusted, prev = len(items), {}, 1.0
    for i in range(m - 1, -1, -1):
        key, p = items[i]
        rank       = i + 1
        adj        = min(prev, p * m / rank)
        adjusted[key] = float(min(adj, 1.0))
        prev       = adj
    return adjusted


# ─────────────────────────────────────────────────────────────────────────────
# Summaries / plots
# ─────────────────────────────────────────────────────────────────────────────
def categorize_feature(feature_name):
    if feature_name.endswith("_is_missing") or feature_name == "Measurement_Count":
        return "missingness"
    if any(tag in feature_name for tag in ["_sampen_24h", "_cv_8h", "_iqr_8h"]):
        return "complexity"
    if any(tag in feature_name for tag in [
        "_last_obs", "_diff_1", "_diff_4", "_energy", "_mean_diff_1", "_shannon_5h"
    ]):
        return "non_sliding"
    if any(tag in feature_name for tag in [
        "_mean_5h", "_std_5h", "_min_5h", "_max_5h", "_median_5h", "_var_5h",
        "_p01_5h", "_p05_5h", "_p95_5h", "_p99_5h",
        "_mean_11h", "_std_11h", "_min_11h", "_max_11h", "_median_11h", "_var_11h",
        "_p01_11h", "_p05_11h", "_p95_11h", "_p99_11h",
    ]):
        return "sliding"
    return "raw"


def save_fold_curve_plots(fold_curves, pooled_true, pooled_prob, output_dir, prefix):
    plt.figure(figsize=(6, 5))
    for item in fold_curves:
        fpr, tpr, _ = roc_curve(item["y_true"], item["y_prob"])
        plt.plot(fpr, tpr, alpha=0.25, lw=1)
    fpr, tpr, _  = roc_curve(pooled_true, pooled_prob)
    pooled_auc   = roc_auc_score(pooled_true, pooled_prob)
    plt.plot(fpr, tpr, lw=2.5, label=f"Pooled AUROC={pooled_auc:.3f}")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC con curvas por fold")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{prefix}_roc_folds.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(6, 5))
    for item in fold_curves:
        prec, rec, _ = precision_recall_curve(item["y_true"], item["y_prob"])
        plt.plot(rec, prec, alpha=0.25, lw=1)
    prec, rec, _ = precision_recall_curve(pooled_true, pooled_prob)
    pooled_auprc = average_precision_score(pooled_true, pooled_prob)
    plt.plot(rec, prec, lw=2.5, label=f"Pooled AUPRC={pooled_auprc:.3f}")
    plt.axhline(y=float(np.mean(pooled_true)), ls="--", color="gray",
                label=f"Prevalencia {np.mean(pooled_true):.3f}")
    plt.xlabel("Recall"); plt.ylabel("Precisión"); plt.title("PR con curvas por fold")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{prefix}_pr_folds.png"), dpi=150)
    plt.close()


def save_calibration_plot(calib_tables, output_dir, filename="calibration_curves.png"):
    plt.figure(figsize=(6, 5))
    for name, df_plot in calib_tables.items():
        plt.plot(df_plot["mean_pred"], df_plot["frac_pos"], marker="o", label=name)
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("Mean predicted probability"); plt.ylabel("Fraction positive")
    plt.title("Calibration curves"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=150)
    plt.close()


def compute_subgroup_metrics(oof_df, prob_col="prob_raw"):
    rows = []
    for col_name, label in [("Age", "Age"), ("Gender", "Gender"), ("Hospital", "Hospital")]:
        if col_name not in oof_df.columns:
            continue
        tmp = oof_df.copy()
        if col_name == "Age":
            age_series = pd.to_numeric(oof_df["Age"], errors="coerce")
            tmp["_grp"] = pd.cut(
                age_series,
                bins=[-np.inf, 50, 70, np.inf],
                labels=["<50", "50-70", ">70"],
                right=False,
            ).astype(str)
        else:
            tmp["_grp"] = tmp[col_name].astype(str)

        for group, gdf in tmp.groupby("_grp"):
            if group == "nan" or len(np.unique(gdf["SepsisLabel"])) < 2:
                continue
            rows.append({
                "type": label, "group": group,
                "n_rows": int(len(gdf)),
                "n_patients": int(gdf["Patient_ID"].nunique()),
                "prevalence": float(gdf["SepsisLabel"].mean()),
                "AUROC": float(roc_auc_score(gdf["SepsisLabel"], gdf[prob_col])),
                "AUPRC": float(average_precision_score(gdf["SepsisLabel"], gdf[prob_col])),
            })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Comparación con referencia
# ─────────────────────────────────────────────────────────────────────────────
def load_reference_results(compare_to):
    if compare_to is None:
        return None
    if compare_to.endswith(".joblib"):
        obj = joblib.load(compare_to)
        if "oof_predictions" not in obj:
            raise ValueError('El joblib de referencia debe contener "oof_predictions".')
        return obj
    if compare_to.endswith(".csv"):
        return {"oof_predictions": pd.read_csv(compare_to)}
    raise ValueError("compare_to debe ser .joblib o .csv")


def compare_against_reference(oof_df, reference_obj, output_dir):
    log_step("Comparando contra corrida de referencia")
    ref_oof = reference_obj["oof_predictions"].copy()

    for col in ["Patient_ID", "TimeStep", "SepsisLabel"]:
        if col not in ref_oof.columns:
            raise ValueError(f"OOF de referencia falta columna: {col}")

    merged = oof_df.merge(
        ref_oof,
        on=["Patient_ID", "TimeStep", "SepsisLabel"],
        suffixes=("_new", "_ref"),
        how="inner",
    )

    results, pvals = {}, {}
    for current_col, ref_col, label in [
        ("prob_raw_new",      "prob_raw_ref",      "raw"),
        ("prob_platt_new",    "prob_platt_ref",    "platt"),
        ("prob_isotonic_new", "prob_isotonic_ref", "isotonic"),
    ]:
        if current_col not in merged.columns or ref_col not in merged.columns:
            continue

        y_true   = merged["SepsisLabel"].values.astype(int)
        pid      = merged["Patient_ID"].values.astype(str)
        pred_new = merged[current_col].values.astype(np.float64)
        pred_ref = merged[ref_col].values.astype(np.float64)

        auroc_boot = paired_bootstrap_compare(y_true, pred_ref, pred_new, pid, "auroc", 200)
        auprc_boot = paired_bootstrap_compare(y_true, pred_ref, pred_new, pid, "auprc", 200)
        delong_p   = delong_roc_test(y_true, pred_ref, pred_new)
        mcnemar    = mcnemar_test(y_true, (pred_ref >= 0.5).astype(int), (pred_new >= 0.5).astype(int))

        results[label] = {
            "AUROC_ref": float(roc_auc_score(y_true, pred_ref)),
            "AUROC_new": float(roc_auc_score(y_true, pred_new)),
            "AUPRC_ref": float(average_precision_score(y_true, pred_ref)),
            "AUPRC_new": float(average_precision_score(y_true, pred_new)),
            "AUROC_paired_bootstrap": auroc_boot,
            "AUPRC_paired_bootstrap": auprc_boot,
            "AUROC_DeLong_p_value":   float(delong_p) if pd.notna(delong_p) else np.nan,
            "McNemar_threshold_0.5":  mcnemar,
        }
        pvals[f"{label}_auroc_bootstrap"] = auroc_boot["p_value"]
        pvals[f"{label}_auprc_bootstrap"] = auprc_boot["p_value"]
        pvals[f"{label}_auroc_delong"]    = delong_p
        pvals[f"{label}_mcnemar"]         = mcnemar["p_value"]

    results["BH_adjusted_p_values"] = benjamini_hochberg(pvals)
    with open(os.path.join(output_dir, "comparison_to_reference.json"), "w") as f:
        json.dump(results, f, indent=2)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Worker for a specific GPU in fold-level multi-GPU execution.
# ─────────────────────────────────────────────────────────────────────────────
def _train_fold_worker(
    gpu_id, fold, X_tr, Y_tr, X_val, Y_val, pos_weight,
    feat_cols, fold_paths, n_jobs
):
    """
    Train one fold on a specific GPU.
    Se lanza como proceso hijo para aislar el contexto CUDA.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    logging.info(f"[Fold {fold}] Training on GPU {gpu_id}")

    model = XGBClassifier(
        max_depth=5,
        learning_rate=0.05,
        n_estimators=3000,
        early_stopping_rounds=100,
        eval_metric="aucpr",
        scale_pos_weight=pos_weight,
        tree_method="hist",
        device="cuda",      # A100: hist+cuda uses native GPU execution.
        n_jobs=n_jobs,
        random_state=SEED,
        verbosity=0,
    )
    model.fit(
        X_tr, Y_tr,
        eval_set=[(X_tr, Y_tr), (X_val, Y_val)],
        verbose=False,
    )
    joblib.dump(model, fold_paths["model"])

    evals = model.evals_result()
    plt.figure(figsize=(6, 4))
    plt.plot(evals["validation_0"]["aucpr"], label="Train")
    plt.plot(evals["validation_1"]["aucpr"], label="Val")
    plt.xlabel("Iteración"); plt.ylabel("AUPRC")
    plt.title(f"Fold {fold} Curva de aprendizaje")
    plt.legend(); plt.tight_layout()
    plt.savefig(fold_paths["learning"], dpi=150)
    plt.close()

    return model


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline principal
# ─────────────────────────────────────────────────────────────────────────────
def run_pipeline(
    data_dir, output_dir, use_hemo=False,
    n_jobs=8, compare_to=None, n_gpus=1, stage="all"
):
    if stage not in {"features", "train", "all"}:
        raise ValueError(f"Unsupported stage: {stage}")
    os.makedirs(output_dir, exist_ok=True)
    suffix = "enhanced" if use_hemo else "baseline"
    configure_output_logging(output_dir)
    log_run_context(data_dir, output_dir, suffix, n_jobs, n_gpus)
    cache_file = os.path.join(output_dir, f"precomputed_features_{suffix}_v2.joblib")
    partial_cache_file = (
        enhanced_partial_cache_path(output_dir)
        if use_hemo and stage in {"features", "all", "train"}
        else None
    )
    allowed_extra_artifacts = [partial_cache_file] if partial_cache_file is not None else None
    logging.info("requested_stage=%s", stage)
    assert_current_output_dir_for_stage(
        output_dir,
        stage=stage,
        cache_file=cache_file,
        allowed_extra_artifacts=allowed_extra_artifacts,
    )

    # Final reruns must start from a clean output directory. The legacy OOF
    # resume path remains implemented for historical inspection helpers, but the
    # main pipeline intentionally does not use it to finalize failed runs.
    existing_oof_df = None

    if existing_oof_df is not None:
        oof_df = existing_oof_df
        fold_metrics = load_existing_fold_metrics(output_dir, oof_df, expected_folds=5)
        all_importances = load_existing_importances(output_dir, expected_folds=5)
        if all_importances and all("feature" in imp.columns for imp in all_importances):
            feat_cols = sorted(
                pd.concat(all_importances, axis=0, ignore_index=True)["feature"]
                .astype(str)
                .unique()
                .tolist()
            )
        else:
            feat_cols = []

        val_true_all = oof_df["SepsisLabel"].values.astype(np.int8)
        val_probs_all = oof_df["prob_raw"].values.astype(np.float32)
        val_patients_all = oof_df["Patient_ID"].astype(str).values
        val_folds_all = oof_df["Fold"].astype(int).values
        fold_curves = []
        for fold in sorted(oof_df["Fold"].astype(int).unique()):
            fold_df = oof_df[oof_df["Fold"].astype(int) == fold]
            fold_curves.append({
                "fold": int(fold),
                "y_true": fold_df["SepsisLabel"].values.astype(np.int8),
                "y_prob": fold_df["prob_raw"].values.astype(np.float32),
            })
        logging.info(f"Resumed OOF shape={oof_df.shape}")
        logging.info(f"Resumed feature count from importance files: {len(feat_cols)}")
        gc.collect()
        log_mem("after OOF resume setup")
    else:
        # Feature loading/computation.
        baseline_cache_file = baseline_cache_path_for_enhanced(output_dir) if use_hemo else None
        df = None
        should_save_feature_cache = False

        if stage == "train":
            df = load_feature_cache(cache_file, required_context=suffix)
            if df is None:
                raise RuntimeError(
                    f"stage=train requires a valid current-policy feature cache: {cache_file}. "
                    "Run --stage features first in the same clean output directory."
                )
            logging.info(
                "stage=train loaded valid feature cache and will not recalculate features: %s",
                cache_file,
            )
        else:
            df = load_feature_cache(cache_file, required_context=suffix)
            if df is not None:
                logging.info(f"Using compatible current-policy feature cache: {cache_file}")

        if df is None and use_hemo:
            partial_df = (
                load_feature_cache(partial_cache_file, required_context="enhanced_partial")
                if partial_cache_file is not None
                else None
            )
            if partial_df is not None:
                logging.info(f"Loading compatible enhanced partial feature cache: {partial_cache_file}")
                log_mem("before loading enhanced partial cache")
                valid, reason = validate_feature_cache_for_enhanced(partial_df)
                if valid:
                    logging.info("Using enhanced partial feature cache as starting point.")
                    df = add_missing_hemodynamic_features_stable(
                        partial_df, partial_cache_file=partial_cache_file
                    )
                    should_save_feature_cache = True
                    del partial_df
                    gc.collect()
                    log_mem("after loading enhanced partial cache")
                else:
                    logging.warning(f"Enhanced partial cache is not usable: {reason}")
                    del partial_df
                    gc.collect()

            if df is None and baseline_cache_file is not None and os.path.exists(baseline_cache_file):
                logging.info("Using baseline precomputed feature cache as starting point for enhanced features.")
                logging.info(f"Baseline cache path: {baseline_cache_file}")
                log_mem("before loading baseline cache for enhanced")
                baseline_df = load_feature_cache(baseline_cache_file, required_context="baseline")
                if baseline_df is None:
                    valid, reason = False, "missing or incompatible current baseline cache metadata"
                else:
                    valid, reason = validate_feature_cache_for_enhanced(baseline_df)
                if valid:
                    log_mem("after loading baseline cache for enhanced")
                    df = add_missing_hemodynamic_features_stable(
                        baseline_df, partial_cache_file=partial_cache_file
                    )
                    should_save_feature_cache = True
                    del baseline_df
                    gc.collect()
                else:
                    logging.warning(
                        f"Baseline cache cannot seed enhanced features: {reason}. "
                        "Falling back to CSV feature computation."
                    )
                    del baseline_df
                    gc.collect()
            elif df is None:
                logging.info(
                    f"Baseline cache unavailable for enhanced features: {baseline_cache_file}. "
                    "Falling back to CSV feature computation."
                )

        if df is None:
            logging.info(
                "Calculando features con ruta HYBRID segura: pandas para missingness/ffill, "
                "cuDF para rolling baseline cuando sea seguro, pandas estable para hemo/SampEn."
            )
            kaggle_file = os.path.join(data_dir, "kaggle_harmonized.csv")
            logging.info("Leyendo CSV en pandas...")
            df_raw = pd.read_csv(kaggle_file, dtype={"Patient_ID": str})
            logging.info(f"Shape crudo: {df_raw.shape}")
            df = compute_all_features(
                df_raw,
                add_hemo=use_hemo,
                partial_cache_file=partial_cache_file,
            )
            del df_raw
            gc.collect()
            should_save_feature_cache = True

        if should_save_feature_cache:
            logging.info(f"Preparing to save feature cache: {cache_file}")
            log_mem("before saving feature cache")
            save_feature_cache(df, cache_file, context=suffix)
            gc.collect()
            maybe_free_gpu()
            log_mem("after saving feature cache")
            logging.info(f"Features cacheados en: {cache_file}")
        else:
            logging.info(f"Using existing feature cache without rewriting: {cache_file}")

        if stage == "features":
            logging.info(
                "stage=features complete. Feature cache ready; no folds, training, calibration, or threshold sweep executed."
            )
            flush_log_handlers()
            return

        meta = ["Patient_ID", "TimeStep", "SourceSet", "SepsisLabel"]
        initial_candidate_cols = list(df.columns)
        excluded_meta_cols = [c for c in meta if c in df.columns]
        candidate_feat_cols = [c for c in initial_candidate_cols if c not in meta]
        non_numeric_cols = [
            c for c in candidate_feat_cols
            if not pd.api.types.is_numeric_dtype(df[c])
        ]
        feat_cols = [c for c in candidate_feat_cols if c not in non_numeric_cols]

        logging.info(f"Initial candidate columns before exclusions: {len(initial_candidate_cols)}")
        logging.info(f"Excluded meta columns present: {excluded_meta_cols}")
        logging.info(f"Excluded non-numeric feature columns: {non_numeric_cols}")
        logging.info(f"Final numeric features used: {len(feat_cols)}")
        log_mem("after feature load")

        fold_map     = build_or_load_fold_assignments(df, data_dir, n_splits=5)
        row_folds    = df["Patient_ID"].astype(str).map(fold_map).astype(int).values
        unique_folds = np.sort(np.unique(row_folds))

        use_gpu  = detect_gpu()
        n_avail  = count_gpus()
        n_gpus   = min(n_gpus, max(n_avail, 1))
        logging.info(
            f"XGBoost device: {'cuda (A100 x' + str(n_gpus) + ')' if use_gpu else 'cpu'} "
            f"| detected GPUs: {n_avail}"
        )

        non_numeric_selected = [
            c for c in feat_cols
            if not pd.api.types.is_numeric_dtype(df[c])
        ]
        if non_numeric_selected:
            raise ValueError(
                f"Non-numeric feature columns detected: {non_numeric_selected}"
            )

        # Convert to float32; pinned memory can be added by the runtime when available.
        X_raw  = df[feat_cols].values.astype(np.float32)
        y_all  = df["SepsisLabel"].values.astype(np.int8)
        pid_all = df["Patient_ID"].astype(str).values

        keep_cols = ["Patient_ID", "TimeStep", "SepsisLabel"]
        for extra_col in ["Age", "Gender", "SourceSet", "Hospital"]:
            if extra_col in df.columns:
                keep_cols.append(extra_col)

        all_val_true, all_val_probs, all_val_patients, all_val_folds = [], [], [], []
        oof_frames, fold_metrics, fold_curves, all_importances = [], [], [], []

        for fold in unique_folds:
            fold_paths = get_fold_artifact_paths(output_dir, fold)
            fold_state = try_load_completed_fold(output_dir, fold)
            log_step(f"Fold {fold} estado = {fold_state['status']}")

            train_mask = row_folds != fold
            val_mask   = row_folds == fold

            if fold_state["status"] == "complete":
                fold_oof        = fold_state["fold_oof"]
                fold_metric_row = fold_state["fold_metric_row"]
                val_probs       = fold_oof["prob_raw"].values.astype(np.float32)
                y_val_loaded    = fold_oof["SepsisLabel"].values.astype(np.int8)
                pid_loaded      = fold_oof["Patient_ID"].astype(str).values

                fold_metrics.append(fold_metric_row)
                all_val_true.append(y_val_loaded)
                all_val_probs.append(val_probs)
                all_val_patients.extend(pid_loaded.tolist())
                all_val_folds.extend([int(fold)] * len(y_val_loaded))
                fold_curves.append({"fold": int(fold), "y_true": y_val_loaded, "y_prob": val_probs})
                oof_frames.append(fold_oof)
                if fold_state["importance_df"] is not None:
                    all_importances.append(fold_state["importance_df"])
                gc.collect()
                continue

            X_tr_raw = X_raw[train_mask]
            X_val_raw = X_raw[val_mask]

            watch_cols = [
                c for c in feat_cols
                if any(k in c for k in ["_cv_8h", "_iqr_8h", "_sampen_24h"])
            ]
            if watch_cols:
                idxs = [feat_cols.index(c) for c in watch_cols]
                for c, j in zip(watch_cols, idxs):
                    tr_col  = X_tr_raw[:, j]
                    val_col = X_val_raw[:, j]
                    logging.info(
                        f"[FOLD {fold}] {c} | "
                        f"TR nan={np.isnan(tr_col).sum()} "
                        f"mean={np.nanmean(tr_col):.6f} std={np.nanstd(tr_col):.6f} | "
                        f"VAL nan={np.isnan(val_col).sum()} "
                        f"mean={np.nanmean(val_col):.6f} std={np.nanstd(val_col):.6f}"
                    )

            Y_tr          = y_all[train_mask]
            Y_val         = y_all[val_mask]
            val_patient_ids = pid_all[val_mask]

            # Current policy: imputation and normalization are fitted on train rows only.
            # This replaces the previous global fallback fill done before CV split.
            X_tr, X_val = fit_and_apply_fold_standardization(
                X_tr_raw, X_val_raw, feature_names=feat_cols, fold=fold
            )

            pos_weight = (len(Y_tr) - Y_tr.sum()) / max(1, Y_tr.sum())
            logging.info(
                f"Fold {fold}: pos_weight={pos_weight:.2f}, features={X_tr.shape[1]}"
            )

            if fold_state["status"] == "model_only":
                log_step(f"Fold {fold}: model found, reconstructing predictions")
                model = joblib.load(fold_paths["model"])
            else:
                # Multi-GPU: assign one GPU per fold in round-robin order.
                gpu_id = int((fold - 1) % n_gpus) if use_gpu else 0
                log_step(f"Fold {fold}: training on GPU {gpu_id}")

                if use_gpu:
                    # Each fold uses its assigned GPU directly.
                    model = XGBClassifier(
                        max_depth=5,
                        learning_rate=0.05,
                        n_estimators=3000,
                        early_stopping_rounds=100,
                        eval_metric="aucpr",
                        scale_pos_weight=pos_weight,
                        tree_method="hist",
                        device=f"cuda:{gpu_id}",   # A100 específica
                        n_jobs=n_jobs,
                        random_state=SEED,
                        verbosity=0,
                    )
                else:
                    model = XGBClassifier(
                        max_depth=5,
                        learning_rate=0.05,
                        n_estimators=3000,
                        early_stopping_rounds=100,
                        eval_metric="aucpr",
                        scale_pos_weight=pos_weight,
                        tree_method="hist",
                        device="cpu",
                        n_jobs=n_jobs,
                        random_state=SEED,
                        verbosity=0,
                    )

                model.fit(
                    X_tr, Y_tr,
                    eval_set=[(X_tr, Y_tr), (X_val, Y_val)],
                    verbose=False,
                )
                joblib.dump(model, fold_paths["model"])

                evals = model.evals_result()
                plt.figure(figsize=(6, 4))
                plt.plot(evals["validation_0"]["aucpr"], label="Train")
                plt.plot(evals["validation_1"]["aucpr"], label="Val")
                plt.xlabel("Iteración"); plt.ylabel("AUPRC")
                plt.title(f"Fold {fold} Curva de aprendizaje")
                plt.legend(); plt.tight_layout()
                plt.savefig(fold_paths["learning"], dpi=150)
                plt.close()

            log_step(f"Fold {fold}: prediciendo validación")
            val_probs   = model.predict_proba(X_val)[:, 1]
            fold_auroc  = roc_auc_score(Y_val, val_probs)
            fold_auprc  = average_precision_score(Y_val, val_probs)

            fold_metric_row = {
                "Fold":       int(fold),
                "AUROC":      float(fold_auroc),
                "AUPRC":      float(fold_auprc),
                "n_rows":     int(len(Y_val)),
                "n_patients": int(len(np.unique(val_patient_ids))),
                "pipeline_policy_version": PIPELINE_POLICY_VERSION,
                "imputation_policy": FOLD_IMPUTATION_POLICY,
                "sampen_early_policy": SAMPEN_EARLY_POLICY,
            }
            fold_metrics.append(fold_metric_row)
            logging.info(f"Fold {fold}: AUROC={fold_auroc:.4f}, AUPRC={fold_auprc:.4f}")

            all_val_true.append(Y_val)
            all_val_probs.append(val_probs)
            all_val_patients.extend(val_patient_ids.tolist())
            all_val_folds.extend([int(fold)] * len(Y_val))
            fold_curves.append({"fold": int(fold), "y_true": Y_val, "y_prob": val_probs})

            # CORRECCIÓN: dtype explícito en el OOF para evitar upcasts silenciosos
            fold_oof = df.loc[val_mask, keep_cols].copy().reset_index(drop=True)
            fold_oof["Fold"]     = int(fold)
            fold_oof["prob_raw"] = val_probs.astype(np.float32)
            fold_oof["PipelineVersion"] = PIPELINE_POLICY_VERSION
            oof_frames.append(fold_oof)

            save_fold_checkpoint(output_dir, fold, fold_oof, fold_metric_row)
            log_step(f"Fold {fold}: checkpoint OOF guardado")

            booster = model.get_booster()
            gain_scores = booster.get_score(importance_type="gain")
            weight_scores = booster.get_score(importance_type="weight")
            imp_df = pd.DataFrame({
                "feature": feat_cols,
                "gain": [float(gain_scores.get(f"f{i}", 0.0)) for i in range(len(feat_cols))],
                "weight": [float(weight_scores.get(f"f{i}", 0.0)) for i in range(len(feat_cols))],
                "fold": int(fold),
                "pipeline_policy_version": PIPELINE_POLICY_VERSION,
            })
            imp_df["selected"] = (imp_df["gain"] > 0).astype(int)
            imp_df["category"] = imp_df["feature"].apply(categorize_feature)
            imp_df.to_csv(fold_paths["importance"], index=False)

            all_importances.append(imp_df)

            del model, X_tr_raw, X_val_raw, X_tr, X_val, Y_tr, Y_val
            del val_probs, val_patient_ids
            gc.collect()
            maybe_free_gpu()
            log_mem(f"after fold {fold}")

        log_step("Todos los folds completados")
        log_mem("before pooled post-processing")

        del X_raw, y_all, pid_all
        gc.collect()

        # ── Post-procesamiento pooled ─────────────────────────────────────────────
        log_step("Concatenando arrays de validación")
        val_true_all     = np.concatenate(all_val_true)
        val_probs_all    = np.concatenate(all_val_probs)
        val_patients_all = np.array(all_val_patients, dtype=str)
        val_folds_all    = np.array(all_val_folds, dtype=int)
        log_step(
            f"Concatenado: rows={len(val_true_all)} "
            f"pacientes={len(np.unique(val_patients_all))}"
        )
        log_mem("after concatenation")

        log_step("Construyendo OOF dataframe")
        # CORRECCIÓN: dtype explícito en concat
        oof_df = pd.concat(oof_frames, axis=0, ignore_index=True)
        oof_df = oof_df.sort_values(
            ["Fold", "Patient_ID", "TimeStep"]
        ).reset_index(drop=True)
        log_step(f"OOF shape={oof_df.shape}")



    log_step("Saving raw OOF predictions")
    atomic_write_csv(oof_df, os.path.join(output_dir, "oof_predictions.csv"), index=False)
    cleanup_after_phase("after raw OOF save")

    log_mem("before calibration")
    if "prob_platt" in oof_df.columns:
        logging.info("prob_platt already present in OOF; reusing existing calibration column")
    else:
        log_step("Calibración Platt")
        oof_df["prob_platt"] = crossfit_calibration(
            val_true_all, val_probs_all, val_folds_all, method="platt"
        )
    if "prob_isotonic" in oof_df.columns:
        logging.info("prob_isotonic already present in OOF; reusing existing calibration column")
    else:
        log_step("Calibración isotónica")
        oof_df["prob_isotonic"] = crossfit_calibration(
            val_true_all, val_probs_all, val_folds_all, method="isotonic"
        )
    log_step("Saving calibrated OOF predictions")
    atomic_write_csv(oof_df, os.path.join(output_dir, "oof_predictions.csv"), index=False)
    cleanup_after_phase("after calibration")

    log_step("Preparando segmentos de pacientes")
    oof_sorted, starts, ends = prepare_patient_segments(oof_df)
    labels_sorted      = oof_sorted["SepsisLabel"].values.astype(np.int8)
    probs_raw_sorted   = oof_sorted["prob_raw"].values.astype(np.float32)
    probs_platt_sorted = oof_sorted["prob_platt"].values.astype(np.float32)
    probs_iso_sorted   = oof_sorted["prob_isotonic"].values.astype(np.float32)
    log_step(f"Segmentos preparados: {len(starts)} pacientes")
    log_step("Precomputando componentes constantes de Utility Score")
    utility_components = precompute_utility_components(labels_sorted, starts, ends)
    gc.collect()
    log_mem("after utility component precompute")

    log_step("Bootstrap CI")
    auc_ci, auprc_lo, auprc_hi = bootstrap_ci_patient(
        val_true_all, val_probs_all, val_patients_all, n_bootstrap=200, ci=95
    )
    logging.info(f"AUROC 95% CI: [{auc_ci[0]:.3f}, {auc_ci[1]:.3f}]")
    logging.info(f"AUPRC 95% CI: [{auprc_lo:.3f}, {auprc_hi:.3f}]")
    log_mem("after bootstrap")

    log_step("Calibration tables")
    ece_raw,   brier_raw,   calib_raw   = calibration_metrics_quantile(val_true_all, oof_df["prob_raw"].values)
    ece_platt, brier_platt, calib_platt = calibration_metrics_quantile(val_true_all, oof_df["prob_platt"].values)
    ece_iso,   brier_iso,   calib_iso   = calibration_metrics_quantile(val_true_all, oof_df["prob_isotonic"].values)

    atomic_write_csv(calib_raw, os.path.join(output_dir, "calibration_bins_raw.csv"), index=False)
    atomic_write_csv(calib_platt, os.path.join(output_dir, "calibration_bins_platt.csv"), index=False)
    atomic_write_csv(calib_iso, os.path.join(output_dir, "calibration_bins_isotonic.csv"), index=False)
    log_step("Generating calibration figure")
    save_calibration_plot(
        {"Raw": calib_raw, "Platt": calib_platt, "Isotonic": calib_iso}, output_dir
    )
    cleanup_after_phase("after calibration tables/plot", close_figures=True)

    log_step("Running low-memory sequential threshold sweep")
    log_mem("before threshold sweep")
    thresholds = np.linspace(0, 1, 101, dtype=np.float32)
    raw_f1s = np.empty(len(thresholds), dtype=np.float64)
    raw_utils = np.empty(len(thresholds), dtype=np.float64)
    platt_utils = np.empty(len(thresholds), dtype=np.float64)
    iso_utils = np.empty(len(thresholds), dtype=np.float64)
    prob_raw_values = oof_df["prob_raw"].to_numpy(dtype=np.float32, copy=False)

    for i, th in enumerate(thresholds):
        if i == 0 or (i + 1) % 10 == 0 or (i + 1) == len(thresholds):
            logging.info("Threshold sweep %d/%d threshold=%.2f", i + 1, len(thresholds), float(th))
            flush_log_handlers()

        raw_f1s[i] = binary_metrics_at_threshold_cpu(val_true_all, prob_raw_values, float(th))["F1"]
        raw_utils[i] = normalized_utility_from_sorted_arrays(
            labels_sorted, probs_raw_sorted, starts, ends, float(th), utility_components
        )
        platt_utils[i] = normalized_utility_from_sorted_arrays(
            labels_sorted, probs_platt_sorted, starts, ends, float(th), utility_components
        )
        iso_utils[i] = normalized_utility_from_sorted_arrays(
            labels_sorted, probs_iso_sorted, starts, ends, float(th), utility_components
        )

    cleanup_after_phase("after threshold sweep")

    best_th_f1      = float(thresholds[int(np.argmax(raw_f1s))])
    best_th_u_raw   = float(thresholds[int(np.argmax(raw_utils))])
    best_th_u_platt = float(thresholds[int(np.argmax(platt_utils))])
    best_th_u_iso   = float(thresholds[int(np.argmax(iso_utils))])

    threshold_rows = []
    for label, prob_col, sorted_probs, chosen_thresholds in [
        ("raw",      "prob_raw",      probs_raw_sorted,   [0.5, best_th_f1, best_th_u_raw]),
        ("platt",    "prob_platt",    probs_platt_sorted, [0.5, best_th_u_platt]),
        ("isotonic", "prob_isotonic", probs_iso_sorted,   [0.5, best_th_u_iso]),
    ]:
        prob_values = oof_df[prob_col].to_numpy(dtype=np.float32, copy=False)
        for th in chosen_thresholds:
            binary = binary_metrics_at_threshold_cpu(val_true_all, prob_values, float(th))
            patient_eval = patient_level_metrics_sorted(
                labels_sorted, sorted_probs, starts, ends, float(th)
            )
            utility_u = normalized_utility_from_sorted_arrays(
                labels_sorted, sorted_probs, starts, ends, th, utility_components
            )
            row = {"probability_source": label}
            row.update(binary)
            row["Utility_U"] = float(utility_u)
            row.update({
                "Patient_sensitivity":  patient_eval["patient_sensitivity"],
                "Patient_specificity":  patient_eval["patient_specificity"],
                "Lead_time_median":     patient_eval["lead_time_median"],
                "Lead_time_IQR_lower":  patient_eval["lead_time_iqr_lower"],
                "Lead_time_IQR_upper":  patient_eval["lead_time_iqr_upper"],
                "TP_patients":          patient_eval["tp_patients"],
                "FN_patients":          patient_eval["fn_patients"],
                "TN_patients":          patient_eval["tn_patients"],
                "FP_patients":          patient_eval["fp_patients"],
            })
            threshold_rows.append(row)

    threshold_df = (
        pd.DataFrame(threshold_rows)
        .drop_duplicates(subset=["probability_source", "threshold"])
        .sort_values(["probability_source", "threshold"])
        .reset_index(drop=True)
    )
    atomic_write_csv(threshold_df, os.path.join(output_dir, "threshold_metrics.csv"), index=False)
    log_step("Métricas de threshold guardadas")
    cleanup_after_phase("after threshold metrics save")

    best_method              = "raw"
    best_method_prob_col     = "prob_raw"
    best_method_brier        = brier_raw
    best_method_probs_sorted = probs_raw_sorted

    if brier_platt < best_method_brier:
        best_method, best_method_prob_col   = "platt", "prob_platt"
        best_method_brier, best_method_probs_sorted = brier_platt, probs_platt_sorted
    if brier_iso < best_method_brier:
        best_method, best_method_prob_col   = "isotonic", "prob_isotonic"
        best_method_brier, best_method_probs_sorted = brier_iso, probs_iso_sorted

    patient_eval_best = patient_level_metrics_sorted(
        labels_sorted, best_method_probs_sorted, starts, ends, 0.5
    )

    log_step("Métricas de subgrupo")
    subgroup_df = compute_subgroup_metrics(oof_df, prob_col=best_method_prob_col)
    atomic_write_csv(subgroup_df, os.path.join(output_dir, "subgroup_metrics.csv"), index=False)
    cleanup_after_phase("after subgroup metrics")

    log_step("Saving fold metrics and feature importances")
    fold_metrics_df = pd.DataFrame(fold_metrics)
    atomic_write_csv(fold_metrics_df, os.path.join(output_dir, "fold_metrics.csv"), index=False)

    if all_importances:
        importances_all_df = pd.concat(all_importances, axis=0, ignore_index=True)
    else:
        logging.warning("No fold feature importance files available; writing empty importance summaries.")
        importances_all_df = pd.DataFrame(
            columns=["feature", "gain", "weight", "fold", "selected", "category"]
        )
    atomic_write_csv(
        importances_all_df,
        os.path.join(output_dir, "feature_importance_all_folds.csv"),
        index=False,
    )

    if importances_all_df.empty:
        importance_summary = pd.DataFrame(
            columns=[
                "feature", "category", "gain_mean", "gain_std", "weight_mean",
                "folds_selected", "selection_freq",
            ]
        )
        category_summary = pd.DataFrame(
            columns=["category", "n_features", "mean_selection_freq", "mean_gain"]
        )
    else:
        importance_summary = (
            importances_all_df
            .groupby(["feature", "category"], as_index=False)
            .agg(
                gain_mean=("gain", "mean"),
                gain_std=("gain", "std"),
                weight_mean=("weight", "mean"),
                folds_selected=("selected", "sum"),
                selection_freq=("selected", "mean"),
            )
            .sort_values(["selection_freq", "gain_mean"], ascending=[False, False])
            .reset_index(drop=True)
        )
        category_summary = (
            importance_summary
            .groupby("category", as_index=False)
            .agg(
                n_features=("feature", "count"),
                mean_selection_freq=("selection_freq", "mean"),
                mean_gain=("gain_mean", "mean"),
            )
            .sort_values("mean_selection_freq", ascending=False)
        )

    atomic_write_csv(
        importance_summary,
        os.path.join(output_dir, "feature_importance_summary.csv"),
        index=False,
    )
    atomic_write_csv(
        category_summary,
        os.path.join(output_dir, "feature_category_summary.csv"),
        index=False,
    )
    cleanup_after_phase("after feature importance summaries")

    log_step("Saving ROC/PR plots")
    save_fold_curve_plots(
        fold_curves, val_true_all, oof_df["prob_raw"].values,
        output_dir, prefix="internal_cv_raw"
    )
    save_fold_curve_plots(
        fold_curves, val_true_all, oof_df[best_method_prob_col].values,
        output_dir, prefix=f"internal_cv_{best_method}"
    )

    fpr, tpr, _ = roc_curve(val_true_all, oof_df[best_method_prob_col].values)
    plt.figure()
    plt.plot(
        fpr, tpr,
        label=f"AUROC={roc_auc_score(val_true_all, oof_df[best_method_prob_col].values):.3f}"
    )
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.title(f"ROC pooled ({best_method})")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "internal_cv_roc.png"), dpi=150)
    plt.close()

    prec, rec, _ = precision_recall_curve(
        val_true_all, oof_df[best_method_prob_col].values
    )
    plt.figure()
    plt.plot(
        rec, prec,
        label=f"AUPRC={average_precision_score(val_true_all, oof_df[best_method_prob_col].values):.3f}"
    )
    plt.axhline(
        y=val_true_all.mean(), ls="--", color="gray",
        label=f"Prevalencia {val_true_all.mean():.3f}"
    )
    plt.xlabel("Recall"); plt.ylabel("Precisión")
    plt.title(f"PR curve ({best_method})")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "internal_cv_pr.png"), dpi=150)
    plt.close()
    cleanup_after_phase("after ROC/PR plots", close_figures=True)

    log_step("Saving OOF predictions")
    atomic_write_csv(oof_df, os.path.join(output_dir, "oof_predictions.csv"), index=False)
    cleanup_after_phase("after final OOF save")

    cv_metrics = {
        "model_variant": suffix,
        "n_features":    int(len(feat_cols)),
        "n_gpus_used":   int(n_gpus),
        "pipeline_policy_version": PIPELINE_POLICY_VERSION,
        "feature_cache_version": FEATURE_CACHE_VERSION,
        "imputation_policy": FOLD_IMPUTATION_POLICY,
        "sampen_early_policy": SAMPEN_EARLY_POLICY,

        "AUROC_raw":      float(roc_auc_score(val_true_all, oof_df["prob_raw"].values)),
        "AUROC_CI_lower": float(auc_ci[0]),
        "AUROC_CI_upper": float(auc_ci[1]),

        "AUPRC_raw":      float(average_precision_score(val_true_all, oof_df["prob_raw"].values)),
        "AUPRC_CI_lower": float(auprc_lo),
        "AUPRC_CI_upper": float(auprc_hi),

        "AUROC_platt":    float(roc_auc_score(val_true_all, oof_df["prob_platt"].values)),
        "AUPRC_platt":    float(average_precision_score(val_true_all, oof_df["prob_platt"].values)),

        "AUROC_isotonic": float(roc_auc_score(val_true_all, oof_df["prob_isotonic"].values)),
        "AUPRC_isotonic": float(average_precision_score(val_true_all, oof_df["prob_isotonic"].values)),

        "ECE_raw":         float(ece_raw),   "Brier_raw":    float(brier_raw),
        "ECE_platt":       float(ece_platt), "Brier_platt":  float(brier_platt),
        "ECE_isotonic":    float(ece_iso),   "Brier_isotonic": float(brier_iso),

        "best_calibration_method_by_brier": best_method,

        "Utility_raw_at_0.5":       float(normalized_utility_from_sorted_arrays(labels_sorted, probs_raw_sorted,   starts, ends, 0.5, utility_components)),
        "Utility_raw_best":         float(np.max(raw_utils)),
        "Utility_raw_best_threshold": float(best_th_u_raw),

        "Utility_platt_at_0.5":       float(normalized_utility_from_sorted_arrays(labels_sorted, probs_platt_sorted, starts, ends, 0.5, utility_components)),
        "Utility_platt_best":         float(np.max(platt_utils)),
        "Utility_platt_best_threshold": float(best_th_u_platt),

        "Utility_isotonic_at_0.5":       float(normalized_utility_from_sorted_arrays(labels_sorted, probs_iso_sorted, starts, ends, 0.5, utility_components)),
        "Utility_isotonic_best":         float(np.max(iso_utils)),
        "Utility_isotonic_best_threshold": float(best_th_u_iso),

        "best_F1_threshold_raw": float(best_th_f1),

        "Patient_sensitivity_best_method_at_0.5": patient_eval_best["patient_sensitivity"],
        "Patient_specificity_best_method_at_0.5": patient_eval_best["patient_specificity"],
        "Lead_time_median_best_method_at_0.5":    patient_eval_best["lead_time_median"],
        "Lead_time_IQR_lower_best_method_at_0.5": patient_eval_best["lead_time_iqr_lower"],
        "Lead_time_IQR_upper_best_method_at_0.5": patient_eval_best["lead_time_iqr_upper"],

        "Fold_AUROC_mean": float(fold_metrics_df["AUROC"].mean()),
        "Fold_AUROC_std":  float(fold_metrics_df["AUROC"].std(ddof=0)),
        "Fold_AUPRC_mean": float(fold_metrics_df["AUPRC"].mean()),
        "Fold_AUPRC_std":  float(fold_metrics_df["AUPRC"].std(ddof=0)),
    }

    reference_comparison = None
    if compare_to is not None:
        reference_obj        = load_reference_results(compare_to)
        reference_comparison = compare_against_reference(oof_df, reference_obj, output_dir)

    log_step("Saving final summary atomically")
    log_mem("before final metrics save")
    results_summary = {
        "cv_metrics": cv_metrics,
        "fold_metrics": fold_metrics_df.to_dict(orient="records"),
        "feature_importance_top50": importance_summary.head(50).to_dict(orient="records"),
        "threshold_metrics": threshold_df.to_dict(orient="records"),
        "subgroup_metrics": subgroup_df.to_dict(orient="records"),
        "reference_comparison": reference_comparison,
        "artifact_paths": {
            "oof_predictions": "oof_predictions.csv",
            "fold_metrics": "fold_metrics.csv",
            "threshold_metrics": "threshold_metrics.csv",
            "feature_importance_summary": "feature_importance_summary.csv",
            "subgroup_metrics": "subgroup_metrics.csv",
        },
        "note": "Compact summary only. Large OOF predictions are stored in oof_predictions.csv, not embedded in this joblib.",
    }
    atomic_joblib_dump(
        results_summary,
        os.path.join(output_dir, "zabihi_results.joblib"),
        compress=3,
    )
    atomic_write_json(cv_metrics, os.path.join(output_dir, "metrics.json"))
    atomic_write_json(cv_metrics, os.path.join(output_dir, "summary_metrics.json"))
    cleanup_after_phase("after final metrics save", close_figures=True)

    logging.info("Done. Results written to %s", output_dir)
    logging.info(
        f"AUROC={cv_metrics['AUROC_raw']:.4f} "
        f"AUPRC={cv_metrics['AUPRC_raw']:.4f} "
        f"Best calibration={best_method} "
        f"GPUs used={n_gpus}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    run_pipeline(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        use_hemo=args.use_hemo,
        n_jobs=args.n_jobs,
        compare_to=args.compare_to,
        n_gpus=args.n_gpus,
        stage=args.stage,
    )


if __name__ == "__main__":
    main()
