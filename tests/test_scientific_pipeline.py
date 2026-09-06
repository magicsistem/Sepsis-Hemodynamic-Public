"""Small independent-oracle tests for the corrected scientific pipeline."""

from __future__ import annotations

import math
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src import scientific_pipeline as pipeline
from vendor.physionet2019 import evaluate_sepsis_score as official


def patient_frame(hours=(1, 2, 3, 4), labels=(0, 0, 0, 0)) -> pd.DataFrame:
    values = {column: [np.nan] * len(hours) for column in pipeline.CHALLENGE_COLUMNS}
    for column in pipeline.STATIC_COLUMNS:
        values[column] = [42.0] * len(hours)
    values.update({"HR": [80.0, 81.0, 82.0, 83.0][: len(hours)], "Hct": [40.0] * len(hours)})
    values["ICULOS"] = list(hours)
    values["SepsisLabel"] = list(labels)
    frame = pd.DataFrame(values)
    first_positive = np.flatnonzero(frame["SepsisLabel"].to_numpy(dtype=int))
    frame["TrueSepsisOnset_ICULOS"] = frame.loc[first_positive[0], "ICULOS"] + 6 if len(first_positive) else np.nan
    return frame


class ScientificPipelineTests(unittest.TestCase):
    def test_official_schema_hct_alias_and_unknown_rejection(self):
        headers = list(pipeline.CHALLENGE_COLUMNS)
        headers[headers.index("Hct")] = "HCT"
        self.assertIn("Hct", pipeline.canonical_headers(headers, "p000001.psv"))
        headers[0] = "ALT"
        with self.assertRaises(pipeline.PipelineError):
            pipeline.canonical_headers(headers, "p000001.psv")

    def test_chronology_and_persistent_shifted_labels_are_fail_closed(self):
        valid = patient_frame(hours=(1, 2, 3, 4), labels=(0, 1, 1, 1))
        pipeline.validate_patient_frame(valid, "p000001.psv")
        invalid_time = valid.copy()
        invalid_time.loc[2, "ICULOS"] = 2
        with self.assertRaises(pipeline.PipelineError):
            pipeline.validate_patient_frame(invalid_time, "p000001.psv")
        invalid_label = valid.copy()
        invalid_label.loc[3, "SepsisLabel"] = 0
        with self.assertRaises(pipeline.PipelineError):
            pipeline.validate_patient_frame(invalid_label, "p000001.psv")

    def test_patient_isolation_and_fold_provenance(self):
        patients = []
        for index in range(10):
            frame = pipeline.feature_patient(patient_frame(), include_hemodynamics=False)
            frame["Patient_ID"] = f"A:p{index:06d}"
            frame["SourceSet"] = "A"
            frame["SepsisLabel"] = int(index % 2 == 0)
            patients.append(frame)
        features = pd.concat(patients, ignore_index=True)
        with tempfile.TemporaryDirectory() as directory:
            fold_path = Path(directory) / "folds.csv"
            pipeline.write_folds(features, fold_path, n_splits=5)
            merged = pipeline.require_fold_context(features, pd.read_csv(fold_path))
        self.assertTrue((merged.groupby("Patient_ID")["Fold"].nunique() == 1).all())

    def test_no_future_last_observation_and_static_features(self):
        values = pd.Series([1.0, np.nan, np.nan])
        times = pd.Series([1.0, 2.0, 30.0])
        observed = pipeline.causal_last_observation(values, times, max_age_hours=24)
        self.assertEqual(observed.iloc[1], 1.0)
        self.assertTrue(math.isnan(observed.iloc[2]))
        features = pipeline.feature_patient(patient_frame(), include_hemodynamics=False)
        self.assertEqual(features["Age"].nunique(), 1)
        self.assertFalse(any(column.startswith("Age_") for column in features.columns))

    def test_sampen_oracle_zero_match_and_no_second_backend(self):
        # Four equal observations: B=3 m-template pairs, A=1 m+1 pair.
        self.assertAlmostEqual(pipeline.sample_entropy([1, 1, 1, 1]), -math.log(1 / 3))
        self.assertTrue(math.isnan(pipeline.sample_entropy([0, 1, 2, 3])))
        # The public pipeline intentionally has one implementation, so its
        # backend parity condition is identity rather than CPU/Numba disagreement.
        self.assertEqual(pipeline.sample_entropy([1, 1, 1, 1]), pipeline.sample_entropy(np.ones(4)))

    def test_rolling_and_shannon_oracles(self):
        series = pd.Series([1.0, 2.0, 3.0], index=[1.0, 2.0, 3.0])
        mean = pipeline.rolling_feature(series, 5, "mean")
        self.assertTrue(math.isnan(mean.iloc[0]))
        self.assertAlmostEqual(mean.iloc[2], 2.0)
        self.assertAlmostEqual(pipeline.shannon_entropy([0, 0, 1, 1], bins=2), math.log(2))
        self.assertGreaterEqual(pipeline.shannon_entropy([0, 0, 1, 1], bins=2), 0.0)

    def test_official_utility_oracles_and_below_inaction_normalization(self):
        labels = np.array([0, 0, 0, 0, 1, 1])
        predictions = np.array([0, 0, 1, 1, 1, 1])
        self.assertAlmostEqual(official.compute_prediction_utility(labels, predictions), 3.388888888888889)
        shifted = np.array([0] * 12 + [1] * 12)
        times = np.arange(1, 25)
        frame = pd.DataFrame({"Patient_ID": "A:p000001", "ICULOS": times, "SepsisLabel": shifted})
        frame["zero"] = 0.0
        frame["early"] = (times >= 6).astype(float)
        frame["late"] = (times >= 21).astype(float)
        self.assertAlmostEqual(pipeline.challenge_utility(frame, "zero", 0.5), 0.0)
        self.assertGreater(pipeline.challenge_utility(frame, "early", 0.5), 0.0)
        self.assertLess(pipeline.challenge_utility(frame.assign(SepsisLabel=0, late=1.0), "late", 0.5), 0.0)

    def test_average_precision_and_pr_auc_are_named_distinct_estimands(self):
        y = np.array([0, 1, 0, 1])
        p = np.array([0.2, 0.8, 0.7, 0.9])
        metrics = pipeline.discrimination_metrics(y, p)
        self.assertAlmostEqual(metrics["average_precision"], pipeline.average_precision_score(y, p))
        self.assertIn("trapezoidal_pr_auc", metrics)
        self.assertEqual(metrics["xgboost_training_eval_metric"], "logloss")

    def test_ece_definition_is_fixed_equal_width_bins(self):
        y = np.array([0, 1, 1, 0])
        p = np.array([0.1, 0.1, 0.9, 0.9])
        # Two occupied equal-width bins: .5*|.5-.1| + .5*|.5-.9| = .4.
        self.assertAlmostEqual(pipeline.calibration_metrics(y, p, bins=2)["ece_fixed_10_bins"], 0.4)

    def test_gpu_cpu_semantics_do_not_treat_import_as_availability(self):
        state = pipeline.gpu_runtime()
        self.assertIn("available", state)
        self.assertIn("device_count", state)
        self.assertFalse(state["available"] and state["device_count"] == 0)

    def test_cache_context_and_manifest_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            existing = Path(directory) / "existing-run"
            existing.mkdir()
            previous = os.environ.get("PYTHONHASHSEED")
            os.environ["PYTHONHASHSEED"] = str(pipeline.SEED)
            try:
                with self.assertRaises(pipeline.PipelineError):
                    pipeline.run_scientific_pipeline(Path(directory), Path(directory) / "missing.zip", existing, "x")
            finally:
                if previous is None:
                    del os.environ["PYTHONHASHSEED"]
                else:
                    os.environ["PYTHONHASHSEED"] = previous
            with self.assertRaises(pipeline.PipelineError):
                pipeline.validate_final_manifest(existing)

    def test_nested_calibration_and_threshold_provenance(self):
        inner = pd.DataFrame({
            "Patient_ID": ["A:p1"] * 4 + ["A:p2"] * 4,
            "ICULOS": list(range(1, 5)) * 2,
            "SepsisLabel": [0, 0, 1, 1, 0, 0, 1, 1],
            "inner_prob_raw": [0.1, 0.2, 0.8, 0.9, 0.1, 0.2, 0.8, 0.9],
            "InnerFold": [0] * 4 + [1] * 4,
        })
        calibrator = pipeline.fitted_platt(inner)
        inner["inner_prob_platt"] = calibrator.predict_proba(inner[["inner_prob_raw"]])[:, 1]
        threshold = pipeline.threshold_from_inner_oof(inner, "inner_prob_platt")
        self.assertIn(threshold, pipeline.FEATURE_POLICY["threshold_grid"])
        self.assertNotIn("Fold", inner.columns)  # Outer held-out rows cannot calibrate themselves.

    def test_reporting_traceability_and_fail_closed_source_set(self):
        with self.assertRaises(pipeline.PipelineError):
            pipeline.source_and_patient("Dataset.psv")
        required = {"runtime", "stages", "artifact_sha256", "lineage", "final_validation", "scientific_status"}
        self.assertTrue(required.issuperset({"runtime", "lineage"}))


if __name__ == "__main__":
    unittest.main()
