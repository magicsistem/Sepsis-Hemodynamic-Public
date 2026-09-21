"""Small independent-oracle tests for the corrected scientific pipeline."""

from __future__ import annotations

import inspect
import json
import math
import os
import sys
import tempfile
import unittest
import warnings
import zipfile
from unittest import mock
from pathlib import Path

import numpy as np
import pandas as pd

from scripts import source_provenance
from src import onset_koopman as onset
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
    frame.insert(0, "Patient_ID", "A:p000001")
    frame.insert(1, "SourceSet", "A")
    onset, onset_status = pipeline.reconstruct_true_onset(frame["SepsisLabel"], frame["ICULOS"])
    frame["TrueSepsisOnset_ICULOS"] = onset
    frame["OnsetReconstructionStatus"] = onset_status
    return frame


class ScientificPipelineTests(unittest.TestCase):
    def test_official_schema_hct_alias_and_unknown_rejection(self):
        self.assertEqual(len(pipeline.PREDICTOR_COLUMNS), 40)
        self.assertTrue({"Hct", "ICULOS"}.issubset(pipeline.PREDICTOR_COLUMNS))
        self.assertTrue({"ALT", "PT", "INR"}.isdisjoint(pipeline.CHALLENGE_COLUMNS))
        headers = list(pipeline.CHALLENGE_COLUMNS)
        headers[headers.index("Hct")] = "HCT"
        self.assertIn("Hct", pipeline.canonical_headers(headers, "p000001.psv"))
        headers[0] = "ALT"
        with self.assertRaises(pipeline.PipelineError):
            pipeline.canonical_headers(headers, "p000001.psv")
        invalid = patient_frame().astype(object)
        invalid.loc[1, "HR"] = "corrupt"
        with self.assertRaises(pipeline.PipelineError):
            pipeline.validate_patient_frame(invalid, "p000001.psv")
        invalid.loc[1, "HR"] = math.inf
        with self.assertRaises(pipeline.PipelineError):
            pipeline.validate_patient_frame(invalid, "p000001.psv")

    def test_archive_inventory_is_filename_sorted_and_rejects_csv_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "input.zip"
            with zipfile.ZipFile(archive, "w") as handle:
                handle.writestr("training_setB/training_setB/p000002.psv", "header\n")
                handle.writestr("Dataset.csv", "historical fallback\n")
                handle.writestr("training_setA/training/p000001.psv", "header\n")
            psv, inventory = pipeline.archive_inventory(archive)
            self.assertEqual([member.filename for member in psv], ["training_setA/training/p000001.psv", "training_setB/training_setB/p000002.psv"])
            self.assertEqual(inventory["member_count"], 3)
            self.assertEqual(pipeline.source_and_patient("training_setB/training_setB/p100001.psv"), ("B", "B:p100001"))
            with self.assertRaises(pipeline.PipelineError):
                pipeline.source_and_patient("training_setB/training/p100001.psv")
            with zipfile.ZipFile(archive, "w") as handle:
                handle.writestr("Dataset.csv", "only fallback\n")
            with self.assertRaises(pipeline.PipelineError):
                pipeline.archive_inventory(archive)

    def test_chronology_and_persistent_shifted_labels_are_fail_closed(self):
        valid = patient_frame(hours=(1, 2, 3, 4), labels=(0, 1, 1, 1))
        pipeline.validate_patient_frame(valid, "p000001.psv")
        strings = valid.astype(object)
        for column in pipeline.CHALLENGE_COLUMNS:
            strings[column] = [str(value) if pd.notna(value) else value for value in strings[column]]
        converted = pipeline.validate_patient_frame(strings, "p000001.psv")
        self.assertTrue(np.issubdtype(converted["ICULOS"].dtype, np.number))
        invalid_time = valid.copy()
        invalid_time.loc[2, "ICULOS"] = 2
        with self.assertRaises(pipeline.PipelineError):
            pipeline.validate_patient_frame(invalid_time, "p000001.psv")
        invalid_label = valid.copy()
        invalid_label.loc[3, "SepsisLabel"] = 0
        with self.assertRaises(pipeline.PipelineError):
            pipeline.validate_patient_frame(invalid_label, "p000001.psv")

    def test_left_censored_sepsis_is_not_assigned_false_onset(self):
        patient = patient_frame(labels=(1, 1, 1, 1))
        reconstructed_onset, status = pipeline.reconstruct_true_onset([0, 1, 1], [1, 2, 3])
        self.assertEqual((reconstructed_onset, status), (8.0, "exact_from_shift_transition"))
        self.assertTrue(patient["TrueSepsisOnset_ICULOS"].isna().all())
        self.assertEqual(patient["OnsetReconstructionStatus"].unique().tolist(), ["septic_onset_left_censored"])
        targeted = onset.add_primary_target(patient)
        self.assertEqual(targeted[onset.ELIGIBLE_COLUMN].sum(), 0)
        self.assertTrue(targeted[onset.TARGET_COLUMN].isna().all())
        with self.assertRaises(pipeline.PipelineError):
            pipeline.reconstruct_true_onset([0, 0.5, 1], [1, 2, 3])
        with self.assertRaises(pipeline.PipelineError):
            pipeline.reconstruct_true_onset([0, 1], [2, 1])

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
            first = pipeline.write_folds(features, fold_path, n_splits=5)
            second_path = Path(directory) / "folds_second.csv"
            second = pipeline.write_folds(features, second_path, n_splits=5, split_seed=pipeline.SEED + 101)
            merged = pipeline.require_fold_context(features, pd.read_csv(fold_path))
            invalid = pd.read_csv(fold_path)
            invalid.loc[0, "SepsisLabel"] = 1 - invalid.loc[0, "SepsisLabel"]
            with self.assertRaises(pipeline.PipelineError):
                pipeline.require_fold_context(features, invalid)
            fractional_label = pd.read_csv(fold_path)
            fractional_label.loc[fractional_label["SepsisLabel"] == 0, "SepsisLabel"] = 0.5
            with self.assertRaises(pipeline.PipelineError):
                pipeline.require_fold_context(features, fractional_label)
            fractional_fold = pd.read_csv(fold_path)
            fractional_fold.loc[0, "Fold"] = 0.5
            with self.assertRaises(pipeline.PipelineError):
                pipeline.require_fold_context(features, fractional_fold)
            missing_fold = pd.read_csv(fold_path)
            missing_fold.loc[missing_fold["Fold"] == 4, "Fold"] = 3
            with self.assertRaises(pipeline.PipelineError):
                pipeline.require_fold_context(features, missing_fold)
        self.assertTrue((merged.groupby("Patient_ID")["Fold"].nunique() == 1).all())
        self.assertEqual(first["seed"], pipeline.SEED)
        self.assertEqual(second["seed"], pipeline.SEED + 101)

    def test_cohort_flow_is_current_run_data_driven(self):
        features = pd.DataFrame({
            "Patient_ID": ["A:p1", "A:p1", "B:p1"], "SourceSet": ["A", "A", "B"], "SepsisLabel": [0, 1, 0],
        })
        flow = pipeline.cohort_flow_summary(features, {"row_count": 3, "patient_count": 2})
        self.assertEqual((flow["available_patients"], flow["included_patients"], flow["excluded_patients"]), (2, 2, 0))
        self.assertEqual((flow["septic_patients"], flow["nonseptic_patients"]), (1, 1))
        self.assertEqual(flow["source_sets"]["A"]["rows"], 2)
        with self.assertRaises(pipeline.PipelineError):
            pipeline.cohort_flow_summary(features, {"row_count": 4, "patient_count": 2})
        with mock.patch.dict(pipeline.DATA_POLICY, {"psv_file_count": 2, "patient_count": 2, "row_count": 3, "source_patient_counts": {"A": 1, "B": 1}}):
            self.assertEqual(pipeline.validate_cohort_identity(features, {"psv_file_count": 2}), {"A": 1, "B": 1})
            with self.assertRaises(pipeline.PipelineError):
                pipeline.validate_cohort_identity(features.iloc[:2], {"psv_file_count": 2})

    def test_source_provenance_hash_validation_is_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "run.sh"
            source.write_text("original\n")
            files = {"run.sh": source_provenance.sha256_file(str(source))}
            sidecar = root / ".source_provenance.json"
            payload = {"git_commit": "0" * 40, "git_dirty": False, "files": files, "source_inventory_sha256": source_provenance.inventory_hash(files)}
            sidecar.write_text(json.dumps(payload))
            self.assertEqual(source_provenance.validate_sidecar(str(root), str(sidecar))["files"], files)
            extra = root / "src" / "stale.py"
            extra.parent.mkdir()
            extra.write_text("raise RuntimeError\n")
            with self.assertRaises(source_provenance.SourceProvenanceError):
                source_provenance.validate_sidecar(str(root), str(sidecar))
            extra.unlink()
            extra = root / "sitecustomize.py"
            extra.write_text("raise RuntimeError\n")
            with self.assertRaises(source_provenance.SourceProvenanceError):
                source_provenance.validate_sidecar(str(root), str(sidecar))
            extra.unlink()
            source.write_text("altered\n")
            with self.assertRaises(source_provenance.SourceProvenanceError):
                source_provenance.validate_sidecar(str(root), str(sidecar))
        with mock.patch.object(source_provenance.subprocess, "check_output", return_value=b"data/raw/archive.zip\0results/old.csv\0run.sh\0"):
            self.assertEqual(source_provenance.tracked_source_files("unused"), ["data/raw/archive.zip", "run.sh"])

    def test_no_future_last_observation_and_static_features(self):
        values = pd.Series([1.0, np.nan, np.nan])
        times = pd.Series([1.0, 2.0, 30.0])
        observed = pipeline.causal_last_observation(values, times, max_age_hours=24)
        self.assertEqual(observed.iloc[1], 1.0)
        self.assertTrue(math.isnan(observed.iloc[2]))
        features = pipeline.feature_patient(patient_frame(), include_hemodynamics=False)
        self.assertIn("raw__HR", features)
        self.assertNotIn("raw__HR", pipeline.model_features(features, "baseline"))
        self.assertIn(pipeline.onset.TARGET_COLUMN, features)
        self.assertNotIn(pipeline.onset.TARGET_COLUMN, pipeline.model_features(features, "baseline"))
        self.assertEqual(features["Age"].nunique(), 1)
        self.assertFalse(any(column.startswith("Age_") for column in features.columns))
        future_changed = patient_frame()
        future_changed.loc[3, "HR"] = 999.0
        before = pipeline.feature_patient(patient_frame(), include_hemodynamics=True)
        after = pipeline.feature_patient(future_changed, include_hemodynamics=True)
        pd.testing.assert_frame_equal(before.iloc[:3], after.iloc[:3])
        corrupt = features.copy()
        corrupt.loc[0, "HR_last_obs"] = "corrupt"
        with self.assertRaises(pipeline.PipelineError):
            pipeline.matrix(corrupt, ["HR_last_obs"])

    def test_feature_construction_does_not_fragment_dataframe(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error", pd.errors.PerformanceWarning)
            features = pipeline.feature_patient(patient_frame(), include_hemodynamics=True)
        self.assertIn("SBP_cv_8h", features.columns)
        self.assertFalse(any("sampen" in column or "_iqr_" in column for column in features.columns))

    def test_sampen_oracle_zero_match_and_no_second_backend(self):
        # Compatible starts only: four constants give B=1 and A=1.
        self.assertEqual(pipeline.sample_entropy([1, 1, 1, 1]), 0.0)
        # Five values give m-templates 00,00,00 (B=3) and
        # (m+1)-templates 000,000,001 (A=1).
        self.assertAlmostEqual(pipeline.sample_entropy([0, 0, 0, 0, 1]), -math.log(1 / 3))
        self.assertTrue(math.isinf(pipeline.sample_entropy([0, 0, 0, 10])))
        self.assertTrue(math.isnan(pipeline.sample_entropy([0, 1, 2, 3])))
        # The public pipeline intentionally has one implementation, so its
        # backend parity condition is identity rather than CPU/Numba disagreement.
        self.assertEqual(pipeline.sample_entropy([1, 1, 1, 1]), pipeline.sample_entropy(np.ones(4)))

    def test_sparse_sampen_minimum_does_not_use_forward_fill(self):
        patient = patient_frame()
        patient["HR"] = [80.0, np.nan, np.nan, 83.0]
        features = pipeline.feature_patient(patient, include_hemodynamics=True)
        self.assertEqual(features["HR_last_obs"].tolist(), [80.0, 80.0, 80.0, 83.0])
        self.assertEqual(features["HR_observation_age_hours"].tolist(), [0.0, 1.0, 2.0, 0.0])
        self.assertTrue(math.isnan(pipeline.sample_entropy(patient["HR"])))
        self.assertFalse(any("sampen" in column for column in pipeline.model_features(features, "enhanced")))

    def test_rolling_and_shannon_oracles(self):
        series = pd.Series([1.0, 2.0, 3.0], index=[1.0, 2.0, 3.0])
        mean = pipeline.rolling_feature(series, 5, "mean")
        self.assertTrue(math.isnan(mean.iloc[0]))
        self.assertAlmostEqual(mean.iloc[2], 2.0)
        self.assertAlmostEqual(pipeline.shannon_entropy([0, 0, 1, 1], bins=2), math.log(2))
        self.assertGreaterEqual(pipeline.shannon_entropy([0, 0, 1, 1], bins=2), 0.0)
        self.assertNotIn("cudf", inspect.getsource(pipeline.rolling_feature).lower())

    def test_official_utility_oracles_and_below_inaction_normalization(self):
        labels = np.array([0, 0, 0, 0, 1, 1])
        predictions = np.array([0, 0, 1, 1, 1, 1])
        self.assertAlmostEqual(official.compute_prediction_utility(labels, predictions), 3.388888888888889)
        shifted = np.array([0] * 12 + [1] * 12)
        optimal = np.zeros(24, dtype=int)
        optimal[6:22] = 1
        without_early = optimal.copy()
        without_early[9] = 0
        without_optimal = optimal.copy()
        without_optimal[12] = 0
        without_late = optimal.copy()
        without_late[18] = 0
        missed = np.zeros(24, dtype=int)
        best_utility = official.compute_prediction_utility(shifted, optimal)
        self.assertGreater(best_utility, official.compute_prediction_utility(shifted, without_early))
        self.assertGreater(best_utility, official.compute_prediction_utility(shifted, without_optimal))
        self.assertGreater(best_utility, official.compute_prediction_utility(shifted, without_late))
        self.assertLess(official.compute_prediction_utility(shifted, missed), 0.0)
        times = np.arange(1, 25)
        frame = pd.DataFrame({"Patient_ID": "A:p000001", "ICULOS": times, "SepsisLabel": shifted})
        frame["zero"] = 0.0
        frame["early"] = (times >= 6).astype(float)
        frame["late"] = (times >= 21).astype(float)
        self.assertAlmostEqual(pipeline.challenge_utility(frame, "zero", 0.5), 0.0)
        self.assertGreater(pipeline.challenge_utility(frame, "early", 0.5), 0.0)
        self.assertLess(official.compute_prediction_utility(np.zeros(24, dtype=int), np.ones(24, dtype=int)), 0.0)
        nonseptic = frame.assign(Patient_ID="B:p000001", SourceSet="B", SepsisLabel=0)
        below_inaction = pd.concat([frame.assign(below=0.0), nonseptic.assign(below=1.0)], ignore_index=True)
        observed = sum(
            official.compute_prediction_utility(patient["SepsisLabel"].to_numpy(int), (patient["below"] >= 0.5).to_numpy(int))
            for _, patient in below_inaction.groupby("Patient_ID")
        )
        # The official scorer is zero-based while ICULOS begins at one.
        best = official.compute_prediction_utility(shifted, ((times >= 7) & (times <= 22)).astype(int))
        inaction = official.compute_prediction_utility(shifted, np.zeros(24, dtype=int))
        expected = (observed - inaction) / (best - inaction)
        self.assertLess(expected, 0.0)
        self.assertAlmostEqual(pipeline.challenge_utility(below_inaction, "below", 0.5), expected)
        with self.assertRaises(pipeline.PipelineError):
            pipeline.challenge_utility(frame.assign(invalid=1.01), "invalid", 0.5)
        with self.assertRaises(pipeline.PipelineError):
            pipeline.challenge_utility(frame, "zero", "invalid")
        with self.assertRaises(pipeline.PipelineError):
            pipeline.challenge_utility(frame.assign(SepsisLabel=np.r_[np.zeros(10), np.ones(5), np.zeros(9)]), "zero", 0.5)

    def test_average_precision_and_pr_auc_are_named_distinct_estimands(self):
        y = np.array([0, 1, 0, 1])
        p = np.array([0.1, 0.35, 0.4, 0.8])
        metrics = pipeline.discrimination_metrics(y, p)
        self.assertAlmostEqual(metrics["average_precision"], (1.0 + 2 / 3) / 2)
        self.assertIn("trapezoidal_pr_auc", metrics)
        self.assertEqual(pipeline.MODEL_POLICY["xgboost_eval_metric"], "logloss")
        self.assertNotIn("xgboost_training_eval_metric", metrics)

    def test_python_hash_seed_is_exported_before_python_starts(self):
        root = Path(__file__).resolve().parents[1]
        entrypoint = (root / "run.sh").read_text(encoding="utf-8")
        job = (root / "jobs" / "run_experiment.slurm").read_text(encoding="utf-8")
        self.assertLess(entrypoint.index("export PYTHONHASHSEED=20260906"), entrypoint.index("python scripts/source_provenance.py"))
        self.assertIn("SOURCE_INVENTORY_SHA256", entrypoint)
        self.assertLess(entrypoint.index("TEST_JOB=$(submit_stage tests"), entrypoint.index("PREPARE_JOB=$(submit_stage prepare"))
        self.assertIn("--dependency=", entrypoint)
        self.assertIn("sacct -n -X", entrypoint)
        self.assertIn("export PYTHONWARNINGS=error", entrypoint)
        self.assertIn("PYTHONHASHSEED=20260906", job)
        self.assertIn('SOURCE_INVENTORY_SHA256="$SOURCE_INVENTORY_SHA256"', job)
        self.assertNotIn("#SBATCH --cpus-per-task", job)
        self.assertNotIn("#SBATCH --mem", job)
        self.assertNotIn("#SBATCH --mem=128G", job)
        self.assertIn("logs/run_ledger.tsv", job)
        self.assertIn("trap '", job)
        self.assertIn("CANCELLED_signal_TERM", job)
        self.assertIn("trap - EXIT", job)
        self.assertIn('NODE_NAME=$(hostname -s)', job)
        self.assertIn('[[ "$NODE_NAME" == compute-0-2 ]]', job)
        self.assertIn("resource_manifest.json", job)
        self.assertNotIn('os.environ["PYTHONHASHSEED"] =', (root / "src" / "scientific_pipeline.py").read_text(encoding="utf-8"))
        production = "\n".join(path.read_text(encoding="utf-8") for directory in (root / "src", root / "scripts") for path in directory.glob("*.py"))
        self.assertNotIn('filterwarnings("ignore")', production)

    def test_ece_definition_is_fixed_equal_width_bins(self):
        y = np.array([0, 1, 1, 0])
        p = np.array([0.1, 0.1, 0.9, 0.9])
        # Two occupied equal-width bins: .5*|.5-.1| + .5*|.5-.9| = .4.
        metrics = pipeline.calibration_metrics(y, p)
        self.assertAlmostEqual(metrics["ece_fixed_10_bins"], 0.4)
        self.assertAlmostEqual(metrics["calibration_in_the_large_intercept_slope_fixed_1"], 0.0, places=6)
        self.assertEqual(pipeline.FEATURE_POLICY["ece_equal_width_bins"], 10)
        boundary = pipeline.calibration_metrics(np.array([0, 1, 1, 0]), np.array([0.0, 1.0, 0.25, 0.75]))
        self.assertEqual(boundary["brier"], 0.28125)
        self.assertEqual(boundary["ece_fixed_10_bins"], 0.375)
        with self.assertRaises(pipeline.PipelineError):
            pipeline.probability_array([0.5, 1.01], "oracle")
        with self.assertRaises(pipeline.PipelineError):
            pipeline.binary_array([0, 0.5, 1], "oracle")

    def test_calibration_regression_standardizes_logits_and_restores_coefficients(self):
        class Recorder:
            intercept_ = np.array([2.0])
            coef_ = np.array([[3.0]])

            def fit(self, x, y, sample_weight):
                self.x = x[:, 0]
                self.weights = sample_weight
                return self

        y = np.array([0, 0, 1, 1])
        probability = np.array([1e-6, 0.2, 0.8, 1 - 1e-6])
        weights = np.array([1.0, 2.0, 3.0, 4.0])
        logit = np.log(probability / (1 - probability))
        mean = np.average(logit, weights=weights)
        scale = np.sqrt(np.average((logit - mean) ** 2, weights=weights))
        recorder = Recorder()
        with mock.patch.object(pipeline, "LogisticRegression", return_value=recorder):
            metrics = pipeline.calibration_metrics(y, probability, sample_weight=weights)
        self.assertAlmostEqual(np.average(recorder.x, weights=weights), 0.0)
        self.assertAlmostEqual(np.average(recorder.x ** 2, weights=weights), 1.0)
        self.assertAlmostEqual(metrics["calibration_slope"], 3.0 / scale)
        self.assertAlmostEqual(metrics["calibration_intercept_with_slope"], 2.0 - 3.0 * mean / scale)
        with self.assertRaisesRegex(pipeline.PipelineError, "not identifiable"):
            pipeline.calibration_metrics(y, np.full(4, 0.5), sample_weight=weights)

    def test_xgboost_fit_equalizes_patient_total_weight(self):
        class Recorder:
            def fit(self, x, y, **kwargs):
                self.kwargs = kwargs
                return self
        frame = pd.DataFrame({"Patient_ID": ["p1", "p1", "p2"], "Age": [1.0, 2.0, 3.0], "SepsisLabel": [0, 0, 1]})
        model = pipeline.fit_xgb(Recorder(), frame, ["Age"], frame)
        totals = pd.DataFrame({"Patient_ID": frame["Patient_ID"], "weight": model.kwargs["sample_weight"]}).groupby("Patient_ID")["weight"].sum()
        self.assertTrue(np.allclose(totals.to_numpy(), 1.0))
        self.assertTrue(np.allclose(model.kwargs["sample_weight_eval_set"][0], model.kwargs["sample_weight"]))

    def test_calibration_uncertainty_uses_patient_clusters(self):
        frame = pd.DataFrame([{"Patient_ID": f"A:p{patient:02d}", "SepsisLabel": patient % 2, "probability": 0.75 if patient % 2 else 0.25} for patient in range(20) for _ in range(1 + patient % 3)])
        weights = pipeline.equal_patient_weights(frame)
        totals = pd.DataFrame({"Patient_ID": frame["Patient_ID"], "weight": weights}).groupby("Patient_ID")["weight"].sum()
        self.assertTrue(np.allclose(totals.to_numpy(), 1.0))
        report = pipeline.calibration_metrics_with_patient_uncertainty(frame, "probability", repeats=20)
        self.assertEqual(report["uncertainty_unit"].split(";", 1)[0], "patient")
        repeated = pipeline.calibration_metrics_with_patient_uncertainty(frame, "probability", repeats=20)
        self.assertEqual(report, repeated)
        for metric in ("brier", "ece_fixed_10_bins", "calibration_in_the_large_intercept_slope_fixed_1", "calibration_intercept_with_slope", "calibration_slope"):
            self.assertLessEqual(report[f"{metric}_ci_95_low"], report[f"{metric}_ci_95_high"])

    def test_gpu_cpu_semantics_do_not_treat_import_as_availability(self):
        state = pipeline.gpu_runtime()
        self.assertIn("available", state)
        self.assertIn("device_count", state)
        self.assertFalse(state["available"] and state["device_count"] == 0)
        if os.environ.get("REQUIRE_GPU") == "true":
            self.assertTrue(state["available"], state)
            self.assertEqual((state["n_gpus_used"], state["device_backend"]), (1, "cuda"))
            backend = pipeline.xgb_backend(state)
            model = pipeline.xgb_model(pipeline.MODEL_CANDIDATES[0], pipeline.SEED, state, n_estimators=2)
            self.assertEqual(model.get_xgb_params()["tree_method"], backend["tree_method"])
            self.assertEqual(model.get_xgb_params()["eval_metric"], "logloss")
            model.fit(np.array([[0.0], [1.0], [0.0], [1.0]]), np.array([0, 1, 0, 1]), verbose=False)
            self.assertEqual(len(model.predict_proba(np.array([[0.0], [1.0]]))), 2)

    def test_optional_dependency_runtime_failure_is_recorded_in_cpu_mode(self):
        real_import = __import__

        def guarded_import(name, *args, **kwargs):
            if name == "cudf":
                raise RuntimeError("no CUDA driver")
            return real_import(name, *args, **kwargs)

        with mock.patch("builtins.__import__", side_effect=guarded_import):
            versions = pipeline.dependency_versions()
        self.assertEqual(versions["cudf"], "unavailable:RuntimeError")

    def test_cupy_without_devices_stays_in_cpu_mode(self):
        class Runtime:
            @staticmethod
            def getDeviceCount():
                return 0

        class FakeCuPy:
            class cuda:
                runtime = Runtime()

        with mock.patch.dict(sys.modules, {"cupy": FakeCuPy}):
            state = pipeline.gpu_runtime()
        self.assertFalse(state["available"])
        self.assertEqual(state["device_count"], 0)
        self.assertEqual(state["n_gpus_used"], 0)
        self.assertEqual(state["device_backend"], "cpu")

    def test_removed_historical_duplicate_and_selection_paths(self):
        features = pipeline.feature_patient(patient_frame(), include_hemodynamics=True)
        columns = pipeline.model_features(features, "enhanced")
        self.assertNotIn("HR", columns)
        self.assertIn("HR_last_obs", columns)
        baseline = pipeline.model_features(features, "baseline")
        self.assertIn("HR_last_obs", baseline)
        self.assertNotIn("HR_cv_8h", baseline)
        self.assertFalse(hasattr(pipeline, "model_summary"))
        self.assertFalse(hasattr(pipeline, "outer_oof"))
        self.assertFalse(hasattr(pipeline, "run_scientific_pipeline"))
        self.assertNotIn("quantile", inspect.getsource(pipeline.calibration_metrics))

    def test_prepare_accepts_only_current_wrapper_resource_scaffold(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(
            os.environ,
            {"PYTHONHASHSEED": str(pipeline.SEED), "SLURM_JOB_ID": "42"},
        ):
            root = Path(directory)
            safe = root / "safe"
            (safe / "resources").mkdir(parents=True)
            (safe / "profiles").mkdir()
            (safe / "resources" / "prepare-42.gpu.csv").touch()
            (safe / "resources" / "prepare-42.time").touch()
            with mock.patch.object(
                pipeline,
                "runtime_manifest",
                side_effect=pipeline.PipelineError("safe scaffold reached runtime"),
            ), self.assertRaisesRegex(pipeline.PipelineError, "safe scaffold reached runtime"):
                pipeline.prepare_direct_onset_stage(root, root / "archive.zip", safe, "run")

            unsafe = root / "unsafe"
            (unsafe / "resources").mkdir(parents=True)
            (unsafe / "features.csv").write_text("stale\n", encoding="utf-8")
            with self.assertRaisesRegex(pipeline.PipelineError, "unsafe existing run content"):
                pipeline.prepare_direct_onset_stage(root, root / "archive.zip", unsafe, "run")

    def test_outer_fold_assignment_is_never_a_model_feature(self):
        frame = pd.DataFrame(columns=["Patient_ID", "SourceSet", "SepsisLabel", "TrueSepsisOnset_ICULOS", "OnsetReconstructionStatus", "Fold", "Hct_last_obs"])
        self.assertNotIn("Fold", pipeline.model_features(frame, "baseline"))
        self.assertNotIn("Fold", pipeline.model_features(frame, "enhanced"))
        duplicate = pd.DataFrame([[1.0, 2.0]], columns=["Hct_last_obs", "Hct_last_obs"])
        with self.assertRaises(pipeline.PipelineError):
            pipeline.model_features(duplicate, "enhanced")

    def test_data_license_notice_distinguishes_local_repackaging(self):
        root = Path(__file__).resolve().parents[1]
        notice = (root / "docs" / "data_license.md").read_text(encoding="utf-8")
        self.assertIn("CC BY 4.0", notice)
        self.assertIn("official PhysioNet archive", notice)
        self.assertIn("repackaging", notice)
        policy = (root / "docs" / "results_policy.md").read_text(encoding="utf-8")
        self.assertIn("tracked, required raw-data dependency", policy)
        self.assertIn("not claimed to recreate this local repackaging", policy)

    def test_transitive_lineage_rejects_tampered_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.csv"
            result = root / "result.csv"
            source.write_text("source\n", encoding="utf-8")
            result.write_text("result\n", encoding="utf-8")
            source_hash = pipeline.sha256_file(source)
            lineage = {
                "source": {
                    "kind": "artifact_lineage", "artifact": "source.csv", "sha256": source_hash,
                    "inputs": {}, "generator": "external_input", "generator_git_commit": "1" * 40,
                    "definition_ids": ["source_v1"],
                },
                "result": {
                    "kind": "artifact_lineage", "artifact": "result.csv", "sha256": pipeline.sha256_file(result),
                    "inputs": {"source.csv": source_hash}, "generator": "oracle:transform",
                    "generator_git_commit": "1" * 40, "definition_ids": ["result_v1"],
                },
            }
            self.assertEqual(pipeline.validate_lineage_nodes(root, lineage), 2)
            lineage["result"]["inputs"]["source.csv"] = "0" * 64
            with self.assertRaises(pipeline.PipelineError):
                pipeline.validate_lineage_nodes(root, lineage)

    def test_stage_manifest_rejects_path_escape(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "prepare_stage_manifest.json").write_text(json.dumps({
                "stage": "prepare", "status": "PASS", "artifacts": {"../escape": "0" * 64},
            }), encoding="utf-8")
            with self.assertRaisesRegex(pipeline.PipelineError, "unsafe artifact path"):
                pipeline._require_stage(root, "prepare")

    def test_reporting_traceability_and_fail_closed_source_set(self):
        production = (Path(__file__).resolve().parents[1] / "src" / "scientific_pipeline.py").read_text(encoding="utf-8")
        for unsupported_claim in ("predeclared", "pre-specified", "pre_specified"):
            self.assertNotIn(unsupported_claim, production)
        with self.assertRaises(pipeline.PipelineError):
            pipeline.source_and_patient("Dataset.psv")
        required = {"runtime", "stages", "artifact_sha256", "lineage", "final_validation", "scientific_status"}
        self.assertTrue(required.issuperset({"runtime", "lineage"}))
        with mock.patch.dict(pipeline.DATA_POLICY, {"archive_sha256": pipeline.sha256_file(Path(__file__))}), \
             mock.patch.dict(os.environ, {"SOURCE_INVENTORY_SHA256": "1" * 64}):
            runtime = pipeline.runtime_manifest(Path.cwd(), "test", sys.argv, Path(__file__))
            self.assertIn("execution_environment", runtime)
            self.assertEqual(runtime["model_policy_hash"], pipeline.stable_hash(pipeline.MODEL_POLICY))
            self.assertEqual(runtime["source_inventory_sha256"], "1" * 64)


if __name__ == "__main__":
    unittest.main()
