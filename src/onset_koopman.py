"""Direct-onset target and fold-local Koopman representations.

The functions in this module are deliberately independent of the trainer so
their temporal and mathematical contracts can be tested with tiny arrays.
"""

from __future__ import annotations

import math
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


TARGET_COLUMN = "OnsetWithin6h"
ELIGIBLE_COLUMN = "PrimaryOutcomeEligible"
HOURS_TO_ONSET_COLUMN = "HoursToTrueOnset"
TARGET_POLICY = {
    "positive_window_hours": (1, 6),
    "exclude_onset_and_post_onset": True,
    "exclude_left_censored_patients": True,
    "control_terminal_hours_excluded": 6,
}
KOOPMAN_POLICY = {
    "minimum_observed_row_fraction": 0.05,
    "minimum_patients_with_two_observations": 1000,
    "maximum_signals": 20,
    "normal_dynamics_minimum_hours_before_onset": 12,
    "maximum_observation_age_hours": 24,
    "ridge_alpha": 1.0,
    "ridge_solver": "lsqr",
    "maximum_training_transitions_per_signal": 20000,
    "transform_batch_rows": 50000,
    "maximum_operator_threads": 32,
    "lifts": ("identity", "quadratic"),
    "innovation_energy_window_hours": 8,
}
CALIBRATION_POLICY = {
    "candidates": ("identity", "logistic_logit"),
    "selection_metric": "patient_balanced_brier_on_outer_train_inner_oof",
    "logistic_penalty": None,
    "logistic_solver": "lbfgs",
    "maximum_iterations": 1000,
}
ALARM_POLICY = {
    "useful_window_hours_before_onset": (1, 6),
    "refractory_hours": 6,
    "maximum_false_alarm_episodes_per_patient_day": 0.25,
    "threshold_grid": tuple(np.round(np.arange(0.01, 1.01, 0.01), 2)),
}
REPRESENTATIONS = ("C0", "C1", "C2", "C3")


class OnsetKoopmanError(RuntimeError):
    """A direct-onset or representation validity gate failed."""


def _finite_numeric(values: Any, context: str) -> np.ndarray:
    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise OnsetKoopmanError(f"{context} must be numeric") from exc
    if array.ndim != 1 or not np.isfinite(array).all():
        raise OnsetKoopmanError(f"{context} must be a finite one-dimensional array")
    return array


def add_primary_target(frame: pd.DataFrame) -> pd.DataFrame:
    """Add the exact 1--6 h onset target and its explicit eligibility mask."""
    required = {
        "Patient_ID", "ICULOS", "SepsisLabel", "TrueSepsisOnset_ICULOS",
        "OnsetReconstructionStatus",
    }
    if missing := required.difference(frame.columns):
        raise OnsetKoopmanError(f"Primary target requires {sorted(missing)}")
    output = frame.copy()
    output[TARGET_COLUMN] = np.nan
    output[ELIGIBLE_COLUMN] = np.int8(0)
    output[HOURS_TO_ONSET_COLUMN] = np.nan
    for patient_id, patient in output.groupby("Patient_ID", sort=False):
        ordered = patient.sort_values("ICULOS", kind="mergesort")
        times = _finite_numeric(ordered["ICULOS"], f"{patient_id} ICULOS")
        if (np.diff(times) <= 0).any():
            raise OnsetKoopmanError(f"{patient_id} ICULOS must be strictly increasing")
        labels = _finite_numeric(ordered["SepsisLabel"], f"{patient_id} labels")
        if not np.isin(labels, [0, 1]).all() or (np.diff(labels) < 0).any():
            raise OnsetKoopmanError(f"{patient_id} Challenge labels must be binary and persistent")
        statuses = ordered["OnsetReconstructionStatus"].dropna().unique()
        onsets = pd.to_numeric(ordered["TrueSepsisOnset_ICULOS"], errors="coerce").dropna().unique()
        if len(statuses) != 1 or len(onsets) > 1:
            raise OnsetKoopmanError(f"{patient_id} has inconsistent onset provenance")
        status = str(statuses[0])
        index = ordered.index
        if status == "septic_onset_left_censored":
            if not labels.max() or len(onsets):
                raise OnsetKoopmanError(f"{patient_id} has inconsistent left-censor status")
            continue
        if status == "nonseptic":
            if labels.max() or len(onsets):
                raise OnsetKoopmanError(f"{patient_id} has inconsistent nonseptic status")
            eligible = times <= times[-1] - TARGET_POLICY["control_terminal_hours_excluded"]
            output.loc[index[eligible], TARGET_COLUMN] = 0
            output.loc[index, ELIGIBLE_COLUMN] = eligible.astype("int8")
            continue
        if status != "exact_from_shift_transition" or len(onsets) != 1 or not labels.max():
            raise OnsetKoopmanError(f"{patient_id} has unsupported onset provenance {status!r}")
        first_positive_time = times[int(np.flatnonzero(labels == 1)[0])]
        if not np.isclose(float(onsets[0]), first_positive_time + 6):
            raise OnsetKoopmanError(f"{patient_id} onset is inconsistent with the Challenge shift")
        hours = float(onsets[0]) - times
        eligible = hours > 0
        target = ((hours >= TARGET_POLICY["positive_window_hours"][0]) &
                  (hours <= TARGET_POLICY["positive_window_hours"][1])).astype("int8")
        output.loc[index, HOURS_TO_ONSET_COLUMN] = hours
        output.loc[index[eligible], TARGET_COLUMN] = target[eligible].astype(float)
        output.loc[index, ELIGIBLE_COLUMN] = eligible.astype("int8")
    eligible = output[ELIGIBLE_COLUMN].to_numpy(dtype=int) == 1
    target = output[TARGET_COLUMN].to_numpy(dtype=float)
    if (
        not np.isin(target[eligible], [0, 1]).all()
        or np.isfinite(target[~eligible]).any()
    ):
        raise OnsetKoopmanError("Primary target eligibility is internally inconsistent")
    return output


def primary_decisions(frame: pd.DataFrame) -> pd.DataFrame:
    """Return only rows eligible for the primary fixed-horizon estimand."""
    if {TARGET_COLUMN, ELIGIBLE_COLUMN}.difference(frame.columns):
        frame = add_primary_target(frame)
    decisions = frame.loc[frame[ELIGIBLE_COLUMN] == 1].copy()
    if decisions.empty or not decisions[TARGET_COLUMN].isin([0, 1]).all():
        raise OnsetKoopmanError("Primary decision cohort is empty or non-binary")
    decisions[TARGET_COLUMN] = decisions[TARGET_COLUMN].astype("int8")
    return decisions


def raw_column(signal: str) -> str:
    return f"raw__{signal}"


def innovation_column(signal: str) -> str:
    return f"koopman_innovation__{signal}"


def delta_column(signal: str) -> str:
    return f"causal_delta__{signal}"


def slope_column(signal: str) -> str:
    return f"causal_slope__{signal}"


def select_dynamic_signals(
    train: pd.DataFrame,
    dynamic_columns: Iterable[str],
    minimum_fraction: float = KOOPMAN_POLICY["minimum_observed_row_fraction"],
    minimum_patients: int = KOOPMAN_POLICY["minimum_patients_with_two_observations"],
    maximum_signals: int = KOOPMAN_POLICY["maximum_signals"],
) -> tuple[str, ...]:
    """Select signals using training rows only; coverage ties break alphabetically."""
    if not 0 <= minimum_fraction <= 1 or minimum_patients < 1 or maximum_signals < 1:
        raise OnsetKoopmanError("Invalid dynamic-signal support policy")
    rows: list[tuple[str, float]] = []
    for signal in dynamic_columns:
        column = raw_column(signal)
        if column not in train:
            raise OnsetKoopmanError(f"Missing raw dynamic signal {column}")
        observed = pd.to_numeric(train[column], errors="coerce").notna()
        coverage = float(observed.mean())
        per_patient = observed.groupby(train["Patient_ID"], sort=False).sum()
        patients = int((per_patient >= 2).sum())
        if coverage >= minimum_fraction and patients >= minimum_patients:
            rows.append((signal, coverage))
    rows.sort(key=lambda item: (-item[1], item[0]))
    return tuple(signal for signal, _ in rows[:maximum_signals])


def _previous_state(frame: pd.DataFrame, signals: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
    ordered = frame.sort_values(["Patient_ID", "ICULOS"], kind="mergesort")
    if not ordered.index.equals(frame.index):
        raise OnsetKoopmanError("Representation input must be sorted by Patient_ID and ICULOS")
    patients = frame["Patient_ID"]
    times = pd.to_numeric(frame["ICULOS"], errors="coerce")
    if times.isna().any():
        raise OnsetKoopmanError("Representation ICULOS must be numeric")
    previous_exists = patients.eq(patients.shift())
    state_parts: list[np.ndarray] = []
    for signal in signals:
        raw = pd.to_numeric(frame[raw_column(signal)], errors="coerce")
        last = raw.groupby(patients, sort=False).ffill().groupby(patients, sort=False).shift()
        observed_at = times.where(raw.notna()).groupby(patients, sort=False).ffill().groupby(patients, sort=False).shift()
        age = (times - observed_at).clip(lower=0, upper=KOOPMAN_POLICY["maximum_observation_age_hours"])
        state_parts.extend([last.to_numpy(dtype=float), age.to_numpy(dtype=float)])
    state = np.column_stack(state_parts) if state_parts else np.empty((len(frame), 0), dtype=float)
    return state, previous_exists.to_numpy(dtype=bool)


def _lift_matrix(state: np.ndarray, lift: str) -> np.ndarray:
    state = np.asarray(state, dtype=np.float32)
    if state.ndim != 2:
        raise OnsetKoopmanError("Koopman state must be a matrix")
    if lift == "identity":
        return state
    if lift != "quadratic":
        raise OnsetKoopmanError(f"Unknown Koopman lift {lift!r}")
    upper = np.triu_indices(state.shape[1])
    quadratic = state[:, upper[0]] * state[:, upper[1]]
    return np.concatenate([state, quadratic], axis=1, dtype=np.float32)


@dataclass
class KoopmanFit:
    all_dynamic_columns: tuple[str, ...]
    selected_signals: tuple[str, ...]
    lift: str
    locations: np.ndarray
    scales: np.ndarray
    state_fill: np.ndarray
    models: dict[str, Ridge]
    residual_locations: dict[str, float]
    residual_scales: dict[str, float]
    training_transition_counts: dict[str, int]
    training_patient_hash: str


def _location_scale(train: pd.DataFrame, signals: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
    locations = []
    scales = []
    for signal in signals:
        observed = pd.to_numeric(train[raw_column(signal)], errors="coerce").dropna().to_numpy(dtype=float)
        if not len(observed):
            raise OnsetKoopmanError(f"Selected signal {signal} has no training observations")
        location = float(np.median(observed))
        scale = float(np.quantile(observed, 0.75) - np.quantile(observed, 0.25))
        locations.append(location)
        scales.append(scale if np.isfinite(scale) and scale > np.finfo(float).eps else 1.0)
    return np.asarray(locations), np.asarray(scales)


def _normalized_state(
    frame: pd.DataFrame,
    signals: tuple[str, ...],
    locations: np.ndarray,
    scales: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    state, previous_exists = _previous_state(frame, signals)
    for index in range(len(signals)):
        state[:, 2 * index] = (state[:, 2 * index] - locations[index]) / scales[index]
        state[:, 2 * index + 1] /= KOOPMAN_POLICY["maximum_observation_age_hours"]
    return state, previous_exists


def _training_patient_hash(frame: pd.DataFrame) -> str:
    import hashlib

    patients = "\n".join(sorted(frame["Patient_ID"].astype(str).unique()))
    return hashlib.sha256(patients.encode()).hexdigest()


def fit_koopman(
    train: pd.DataFrame,
    dynamic_columns: Iterable[str],
    lift: str,
    *,
    selected_signals: Iterable[str] | None = None,
    minimum_fraction: float = KOOPMAN_POLICY["minimum_observed_row_fraction"],
    minimum_patients: int = KOOPMAN_POLICY["minimum_patients_with_two_observations"],
    maximum_signals: int = KOOPMAN_POLICY["maximum_signals"],
) -> KoopmanFit:
    """Fit non-imminent dynamics without using validation/test rows."""
    all_dynamic = tuple(dynamic_columns)
    selected = tuple(selected_signals) if selected_signals is not None else select_dynamic_signals(
        train, all_dynamic, minimum_fraction, minimum_patients, maximum_signals
    )
    if not selected or not set(selected).issubset(all_dynamic):
        raise OnsetKoopmanError("No supported Koopman signals or signal outside official dynamic schema")
    locations, scales = _location_scale(train, selected)
    state, previous_exists = _normalized_state(train, selected, locations, scales)
    state_fill = np.nanmedian(state[previous_exists], axis=0)
    state_fill = np.where(np.isfinite(state_fill), state_fill, 0.0)
    state = np.where(np.isfinite(state), state, state_fill).astype(np.float32)
    status = train["OnsetReconstructionStatus"].astype(str)
    hours = pd.to_numeric(train.get(HOURS_TO_ONSET_COLUMN), errors="coerce")
    normal = status.eq("nonseptic") | (status.eq("exact_from_shift_transition") & (hours > KOOPMAN_POLICY["normal_dynamics_minimum_hours_before_onset"]))
    normal = normal.to_numpy(dtype=bool) & previous_exists
    patient_hash = _training_patient_hash(train)
    try:
        allocated_threads = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
    except ValueError as exc:
        raise OnsetKoopmanError("SLURM_CPUS_PER_TASK must be an integer") from exc
    if not 1 <= allocated_threads <= KOOPMAN_POLICY["maximum_operator_threads"]:
        raise OnsetKoopmanError("Koopman operator threads exceed the approved CPU bounds")

    def fit_signal(index: int, signal: str) -> tuple[str, Ridge | None, float, float, int]:
        response = pd.to_numeric(train[raw_column(signal)], errors="coerce").to_numpy(dtype=float)
        observed = normal & np.isfinite(response)
        if observed.sum() < 2:
            return signal, None, math.nan, math.nan, 0
        observed_index = np.flatnonzero(observed)
        maximum = KOOPMAN_POLICY["maximum_training_transitions_per_signal"]
        if len(observed_index) > maximum:
            import hashlib

            seed = int(hashlib.sha256(f"{patient_hash}:{signal}".encode()).hexdigest()[:16], 16)
            observed_index = np.sort(
                np.random.default_rng(seed).choice(observed_index, maximum, replace=False)
            )
        lifted = _lift_matrix(state[observed_index], lift)
        normalized = (response[observed_index] - locations[index]) / scales[index]
        model = Ridge(
            alpha=KOOPMAN_POLICY["ridge_alpha"],
            solver=KOOPMAN_POLICY["ridge_solver"],
        ).fit(lifted, normalized)
        residual = normalized - model.predict(lifted)
        center = float(np.median(residual))
        scale = float(np.quantile(residual, 0.75) - np.quantile(residual, 0.25))
        scale = scale if np.isfinite(scale) and scale > np.finfo(float).eps else 1.0
        return signal, model, center, scale, int(len(observed_index))

    workers = min(allocated_threads, len(selected))
    from threadpoolctl import threadpool_limits

    with threadpool_limits(limits=1), ThreadPoolExecutor(max_workers=workers) as executor:
        fitted_signals = list(
            executor.map(lambda item: fit_signal(*item), enumerate(selected))
        )
    models = {signal: model for signal, model, _, _, _ in fitted_signals if model is not None}
    residual_locations = {
        signal: center for signal, model, center, _, _ in fitted_signals if model is not None
    }
    residual_scales = {
        signal: scale for signal, model, _, scale, _ in fitted_signals if model is not None
    }
    transition_counts = {signal: count for signal, _, _, _, count in fitted_signals}
    return KoopmanFit(
        all_dynamic_columns=all_dynamic,
        selected_signals=selected,
        lift=lift,
        locations=locations,
        scales=scales,
        state_fill=state_fill,
        models=models,
        residual_locations=residual_locations,
        residual_scales=residual_scales,
        training_transition_counts=transition_counts,
        training_patient_hash=patient_hash,
    )


def _rolling_energy(frame: pd.DataFrame, energy: np.ndarray, hours: int) -> tuple[np.ndarray, np.ndarray]:
    means = np.full(len(frame), np.nan, dtype=float)
    maxima = np.full(len(frame), np.nan, dtype=float)
    for _, positions in frame.groupby("Patient_ID", sort=False).indices.items():
        position = np.asarray(positions, dtype=int)
        time = pd.to_numeric(frame.iloc[position]["ICULOS"], errors="raise").to_numpy(dtype=float)
        values = energy[position]
        for local_end, absolute_end in enumerate(position):
            start = np.searchsorted(time, time[local_end] - hours, side="right")
            window = values[start : local_end + 1]
            finite = window[np.isfinite(window)]
            if len(finite):
                means[absolute_end] = float(finite.mean())
                maxima[absolute_end] = float(finite.max())
    return means, maxima


def transform_koopman(frame: pd.DataFrame, fitted: KoopmanFit) -> pd.DataFrame:
    """Emit fixed-schema innovations only at genuinely observed target times."""
    state, previous_exists = _normalized_state(
        frame, fitted.selected_signals, fitted.locations, fitted.scales
    )
    state = np.where(np.isfinite(state), state, fitted.state_fill).astype(np.float32)
    matrix = np.full((len(frame), len(fitted.all_dynamic_columns)), np.nan, dtype=np.float32)
    responses = {
        signal: pd.to_numeric(frame[raw_column(signal)], errors="coerce").to_numpy(dtype=float)
        for signal in fitted.selected_signals
    }
    batch = KOOPMAN_POLICY["transform_batch_rows"]
    for start in range(0, len(frame), batch):
        end = min(start + batch, len(frame))
        lifted = _lift_matrix(state[start:end], fitted.lift)
        prior = previous_exists[start:end]
        for selected_index, signal in enumerate(fitted.selected_signals):
            if signal not in fitted.models:
                continue
            response = responses[signal][start:end]
            observed = np.isfinite(response) & prior
            if not observed.any():
                continue
            prediction = fitted.models[signal].predict(lifted[observed])
            residual = (
                (response[observed] - fitted.locations[selected_index])
                / fitted.scales[selected_index]
                - prediction
            )
            column = fitted.all_dynamic_columns.index(signal)
            batch_values = matrix[start:end, column]
            batch_values[observed] = (
                (residual - fitted.residual_locations[signal])
                / fitted.residual_scales[signal]
            ).astype(np.float32)
    output = pd.DataFrame(
        matrix,
        index=frame.index,
        columns=[innovation_column(signal) for signal in fitted.all_dynamic_columns],
    )
    count = np.isfinite(matrix).sum(axis=1).astype("int16")
    with np.errstate(invalid="ignore"):
        energy = np.nansum(matrix * matrix, axis=1) / np.where(count > 0, count, np.nan)
    mean, maximum = _rolling_energy(frame, energy, KOOPMAN_POLICY["innovation_energy_window_hours"])
    output["koopman_energy"] = energy.astype(np.float32)
    output["koopman_energy_mean_8h"] = mean.astype(np.float32)
    output["koopman_energy_max_8h"] = maximum.astype(np.float32)
    output["koopman_innovation_count"] = count
    return output


def transform_deltas(
    frame: pd.DataFrame,
    all_dynamic_columns: Iterable[str],
    selected_signals: Iterable[str],
) -> pd.DataFrame:
    """Emit fixed-schema observed deltas/slopes without treating LOCF as measurement."""
    all_dynamic = tuple(all_dynamic_columns)
    selected = set(selected_signals)
    output = pd.DataFrame(index=frame.index)
    patients = frame["Patient_ID"]
    times = pd.to_numeric(frame["ICULOS"], errors="raise")
    for signal in all_dynamic:
        delta = np.full(len(frame), np.nan, dtype=np.float32)
        slope = np.full(len(frame), np.nan, dtype=np.float32)
        if signal in selected:
            raw = pd.to_numeric(frame[raw_column(signal)], errors="coerce")
            previous_value = raw.groupby(patients, sort=False).ffill().groupby(patients, sort=False).shift()
            previous_time = times.where(raw.notna()).groupby(patients, sort=False).ffill().groupby(patients, sort=False).shift()
            observed = raw.notna() & previous_value.notna() & previous_time.notna()
            difference = raw[observed] - previous_value[observed]
            elapsed = times[observed] - previous_time[observed]
            if (elapsed <= 0).any():
                raise OnsetKoopmanError("Delta transform requires strictly increasing observation times")
            delta[observed.to_numpy()] = difference.to_numpy(dtype=np.float32)
            slope[observed.to_numpy()] = (difference / elapsed).to_numpy(dtype=np.float32)
        output[delta_column(signal)] = delta
        output[slope_column(signal)] = slope
    return output


def equal_patient_weights(frame: pd.DataFrame) -> np.ndarray:
    counts = frame.groupby("Patient_ID", sort=False)["Patient_ID"].transform("size").to_numpy(dtype=float)
    if not len(counts) or (counts <= 0).any():
        raise OnsetKoopmanError("Patient-balanced weights require nonempty patient identities")
    return 1.0 / counts


def patient_balanced_average_precision(frame: pd.DataFrame, probability_column: str) -> float:
    decisions = primary_decisions(frame)
    probability = _finite_numeric(decisions[probability_column], "Primary probability")
    if ((probability < 0) | (probability > 1)).any() or set(decisions[TARGET_COLUMN]) != {0, 1}:
        raise OnsetKoopmanError("Patient-balanced AP requires both classes and probabilities in [0,1]")
    return float(average_precision_score(
        decisions[TARGET_COLUMN], probability, sample_weight=equal_patient_weights(decisions)
    ))


@dataclass
class CalibrationFit:
    method: str
    model: LogisticRegression | None
    identity_brier: float
    logistic_brier: float


def _logit(probability: np.ndarray) -> np.ndarray:
    clipped = np.clip(probability, 1e-6, 1 - 1e-6)
    return np.log(clipped / (1 - clipped))


def fit_calibration_policy(inner_oof: pd.DataFrame, probability_column: str = "prob_raw") -> CalibrationFit:
    decisions = primary_decisions(inner_oof)
    probability = _finite_numeric(decisions[probability_column], "Calibration probability")
    target = decisions[TARGET_COLUMN].to_numpy(dtype=int)
    weights = equal_patient_weights(decisions)
    if set(target) != {0, 1} or ((probability < 0) | (probability > 1)).any():
        raise OnsetKoopmanError("Nested calibration requires both classes and probabilities in [0,1]")
    model = LogisticRegression(
        penalty=CALIBRATION_POLICY["logistic_penalty"],
        solver=CALIBRATION_POLICY["logistic_solver"],
        max_iter=CALIBRATION_POLICY["maximum_iterations"],
    )
    model.fit(_logit(probability).reshape(-1, 1), target, sample_weight=weights)
    calibrated = model.predict_proba(_logit(probability).reshape(-1, 1))[:, 1]
    identity_brier = float(brier_score_loss(target, probability, sample_weight=weights))
    logistic_brier = float(brier_score_loss(target, calibrated, sample_weight=weights))
    return CalibrationFit(
        method="logistic_logit" if logistic_brier < identity_brier else "identity",
        model=model,
        identity_brier=identity_brier,
        logistic_brier=logistic_brier,
    )


def apply_calibration(fitted: CalibrationFit, probability: Any) -> np.ndarray:
    values = _finite_numeric(probability, "Calibration application")
    if ((values < 0) | (values > 1)).any():
        raise OnsetKoopmanError("Calibration probabilities must lie in [0,1]")
    if fitted.method == "identity":
        return values
    if fitted.method != "logistic_logit" or fitted.model is None:
        raise OnsetKoopmanError("Unknown fitted calibration policy")
    return fitted.model.predict_proba(_logit(values).reshape(-1, 1))[:, 1]


def alarm_metrics(
    frame: pd.DataFrame,
    probability_column: str,
    threshold: float | str,
    refractory_hours: int = ALARM_POLICY["refractory_hours"],
) -> dict[str, float | int]:
    decisions = primary_decisions(frame).sort_values(["Patient_ID", "ICULOS"], kind="mergesort")
    if (
        (not isinstance(threshold, str) and not 0 <= float(threshold) <= 1)
        or refractory_hours < 1
    ):
        raise OnsetKoopmanError("Invalid alarm policy")
    useful_patients = eligible_septic = false_episodes = total_episodes = 0
    lead_times: list[float] = []
    for _, patient in decisions.groupby("Patient_ID", sort=False):
        probability = _finite_numeric(patient[probability_column], "Alarm probability")
        if ((probability < 0) | (probability > 1)).any():
            raise OnsetKoopmanError("Alarm probabilities must lie in [0,1]")
        times = patient["ICULOS"].to_numpy(dtype=float)
        if isinstance(threshold, str):
            if threshold not in patient:
                raise OnsetKoopmanError(f"Missing alarm threshold column {threshold}")
            threshold_values = pd.to_numeric(patient[threshold], errors="coerce").dropna().unique()
            if len(threshold_values) != 1 or not 0 <= float(threshold_values[0]) <= 1:
                raise OnsetKoopmanError("Each patient must have one valid nested alarm threshold")
            patient_threshold = float(threshold_values[0])
        else:
            patient_threshold = float(threshold)
        positive = probability >= patient_threshold
        episodes: list[float] = []
        previous_positive = False
        for time, is_positive in zip(times, positive):
            if is_positive and not previous_positive and (not episodes or time - episodes[-1] >= refractory_hours):
                episodes.append(float(time))
            previous_positive = bool(is_positive)
        total_episodes += len(episodes)
        onset = pd.to_numeric(patient["TrueSepsisOnset_ICULOS"], errors="coerce").dropna().unique()
        if len(onset):
            eligible_septic += 1
            minimum, maximum = ALARM_POLICY["useful_window_hours_before_onset"]
            useful = [time for time in episodes if float(onset[0]) - maximum <= time <= float(onset[0]) - minimum]
            remote = [time for time in episodes if time < float(onset[0]) - maximum]
            false_episodes += len(remote)
            if useful:
                useful_patients += 1
                lead_times.append(float(onset[0]) - useful[0])
        else:
            false_episodes += len(episodes)
    patient_days = len(decisions) / 24
    return {
        "n_decision_hours": int(len(decisions)),
        "n_alarm_episodes": int(total_episodes),
        "n_onset_eligible_septic_patients": int(eligible_septic),
        "tp_patients": int(useful_patients),
        "useful_sensitivity": useful_patients / eligible_septic if eligible_septic else math.nan,
        "false_alarm_episodes": int(false_episodes),
        "false_alarm_episodes_per_patient_day": false_episodes / patient_days if patient_days else math.nan,
        "median_lead_time_hours": float(np.median(lead_times)) if lead_times else math.nan,
    }


def select_alarm_threshold(
    inner_oof: pd.DataFrame,
    probability_column: str,
    maximum_false_alarms_per_patient_day: float = ALARM_POLICY["maximum_false_alarm_episodes_per_patient_day"],
    thresholds: Iterable[float] = ALARM_POLICY["threshold_grid"],
) -> tuple[float, dict[str, float | int]]:
    feasible = []
    for threshold in thresholds:
        metrics = alarm_metrics(inner_oof, probability_column, float(threshold))
        if metrics["false_alarm_episodes_per_patient_day"] <= maximum_false_alarms_per_patient_day:
            feasible.append((float(threshold), metrics))
    if not feasible:
        raise OnsetKoopmanError("No inner-OOF threshold satisfies the alarm budget")
    return min(
        feasible,
        key=lambda item: (-float(item[1]["useful_sensitivity"]), float(item[1]["false_alarm_episodes_per_patient_day"]), item[0]),
    )


def primary_performance(frame: pd.DataFrame, probability_column: str) -> dict[str, float]:
    decisions = primary_decisions(frame)
    target = decisions[TARGET_COLUMN].to_numpy(dtype=int)
    probability = _finite_numeric(decisions[probability_column], "Primary performance probability")
    weights = equal_patient_weights(decisions)
    if set(target) != {0, 1} or ((probability < 0) | (probability > 1)).any():
        raise OnsetKoopmanError("Primary performance requires both classes and valid probabilities")
    return {
        "patient_balanced_average_precision": float(average_precision_score(target, probability, sample_weight=weights)),
        "patient_balanced_auroc": float(roc_auc_score(target, probability, sample_weight=weights)),
        "patient_balanced_brier": float(brier_score_loss(target, probability, sample_weight=weights)),
    }


def paired_patient_bootstrap(
    comparator: pd.DataFrame,
    candidate: pd.DataFrame,
    probability_column: str = "prob_calibrated",
    repeats: int = 300,
    seed: int = 20260906,
) -> list[dict[str, float | int | str]]:
    """Paired patient-cluster intervals for C3 minus C0 primary metrics."""
    key = ["Patient_ID", "ICULOS", TARGET_COLUMN, ELIGIBLE_COLUMN]
    left = comparator.sort_values(["Patient_ID", "ICULOS"], kind="mergesort").reset_index(drop=True)
    right = candidate.sort_values(["Patient_ID", "ICULOS"], kind="mergesort").reset_index(drop=True)
    if not left[key].equals(right[key]):
        raise OnsetKoopmanError("Paired primary inference requires identical decisions")
    eligible = left[ELIGIBLE_COLUMN].to_numpy(dtype=int) == 1
    left = left.loc[eligible].reset_index(drop=True)
    right = right.loc[eligible].reset_index(drop=True)
    target = left[TARGET_COLUMN].to_numpy(dtype=int)
    p0 = _finite_numeric(left[probability_column], "Comparator probability")
    p1 = _finite_numeric(right[probability_column], "Candidate probability")
    base_weight = equal_patient_weights(left)
    codes, patients = pd.factorize(left["Patient_ID"], sort=True)
    if repeats < 1 or len(patients) < 2 or set(target) != {0, 1}:
        raise OnsetKoopmanError("Invalid paired patient-bootstrap inputs")

    def differences(weight: np.ndarray) -> dict[str, float]:
        return {
            "patient_balanced_average_precision": float(
                average_precision_score(target, p1, sample_weight=weight)
                - average_precision_score(target, p0, sample_weight=weight)
            ),
            "patient_balanced_brier": float(
                brier_score_loss(target, p1, sample_weight=weight)
                - brier_score_loss(target, p0, sample_weight=weight)
            ),
        }

    point = differences(base_weight)
    samples = {name: [] for name in point}
    rng = np.random.default_rng(seed)
    accepted = 0
    attempts = 0
    while accepted < repeats and attempts < repeats * 10:
        attempts += 1
        multiplicity = rng.multinomial(len(patients), np.full(len(patients), 1 / len(patients)))[codes]
        weight = base_weight * multiplicity
        if set(target[weight > 0]) != {0, 1}:
            continue
        for name, value in differences(weight).items():
            samples[name].append(value)
        accepted += 1
    if accepted != repeats:
        raise OnsetKoopmanError("Patient bootstrap could not retain both classes")
    rows = []
    for name, value in point.items():
        low, high = np.quantile(samples[name], [0.025, 0.975])
        rows.append({
            "metric": name,
            "candidate_minus_comparator": value,
            "paired_patient_bootstrap_ci_95_low": float(low),
            "paired_patient_bootstrap_ci_95_high": float(high),
            "bootstrap_repeats": repeats,
            "inference_unit": "patient",
        })
    return rows


def decision_curve(
    frame: pd.DataFrame,
    probability_column: str,
    thresholds: Iterable[float],
    repeats: int = 300,
    seed: int = 20260906,
) -> list[dict[str, float | int | str]]:
    """Fixed-horizon DCA with patient-cluster uncertainty."""
    decisions = primary_decisions(frame)
    target = decisions[TARGET_COLUMN].to_numpy(dtype=int)
    probability = _finite_numeric(decisions[probability_column], "DCA probability")
    codes, patients = pd.factorize(decisions["Patient_ID"], sort=True)
    patient_counts = np.bincount(codes, minlength=len(patients))
    rows = []
    for threshold in thresholds:
        threshold = float(threshold)
        if not 0 < threshold < 1:
            raise OnsetKoopmanError("DCA thresholds must lie strictly inside (0,1)")
        odds = threshold / (1 - threshold)
        action = probability >= threshold
        model_contribution = action * target - action * (1 - target) * odds
        all_contribution = target - (1 - target) * odds
        model_by_patient = np.bincount(codes, weights=model_contribution, minlength=len(patients))
        all_by_patient = np.bincount(codes, weights=all_contribution, minlength=len(patients))
        rng = np.random.default_rng(seed + int(threshold * 10000))
        model_samples = []
        all_samples = []
        for _ in range(repeats):
            multiplicity = rng.multinomial(len(patients), np.full(len(patients), 1 / len(patients)))
            denominator = float(multiplicity @ patient_counts)
            model_samples.append(float((multiplicity @ model_by_patient) / denominator))
            all_samples.append(float((multiplicity @ all_by_patient) / denominator))
        model_low, model_high = np.quantile(model_samples, [0.025, 0.975])
        all_low, all_high = np.quantile(all_samples, [0.025, 0.975])
        rows.append({
            "outcome_estimand": "true onset in 1--6 hours",
            "threshold_probability": threshold,
            "model_net_benefit": float(model_contribution.mean()),
            "model_net_benefit_ci_95_low": float(model_low),
            "model_net_benefit_ci_95_high": float(model_high),
            "treat_all_net_benefit": float(all_contribution.mean()),
            "treat_all_net_benefit_ci_95_low": float(all_low),
            "treat_all_net_benefit_ci_95_high": float(all_high),
            "treat_none_net_benefit": 0.0,
            "n_decision_hours": int(len(decisions)),
            "n_patients": int(len(patients)),
            "uncertainty_unit": "patient-cluster bootstrap",
        })
    return rows
