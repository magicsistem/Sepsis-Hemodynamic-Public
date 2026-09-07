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
        onset, status = pipeline.reconstruct_true_onset([0, 1, 1], [1, 2, 3])
        self.assertEqual((onset, status), (8.0, "exact_from_shift_transition"))
        self.assertTrue(patient["TrueSepsisOnset_ICULOS"].isna().all())
        self.assertEqual(patient["OnsetReconstructionStatus"].unique().tolist(), ["septic_onset_left_censored"])
        patient["probability"] = 1.0
        patient["threshold"] = 0.5
        summary = pipeline.early_warning_metrics(patient, "probability", "threshold")["summary"]
        self.assertEqual((summary["n_septic_patients"], summary["n_nonseptic_patients"]), (1, 0))
        self.assertEqual((summary["n_onset_eligible_septic_patients"], summary["n_left_censored_septic_patients_excluded_from_onset_estimands"]), (0, 1))
        self.assertTrue(math.isnan(summary["useful_early_alert_sensitivity"]))

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
            source.write_text("altered\n")
            with self.assertRaises(source_provenance.SourceProvenanceError):
                source_provenance.validate_sidecar(str(root), str(sidecar))

    def test_no_future_last_observation_and_static_features(self):
        values = pd.Series([1.0, np.nan, np.nan])
        times = pd.Series([1.0, 2.0, 30.0])
        observed = pipeline.causal_last_observation(values, times, max_age_hours=24)
        self.assertEqual(observed.iloc[1], 1.0)
        self.assertTrue(math.isnan(observed.iloc[2]))
        features = pipeline.feature_patient(patient_frame(), include_hemodynamics=False)
        self.assertEqual(features["Age"].nunique(), 1)
        self.assertFalse(any(column.startswith("Age_") for column in features.columns))

    def test_feature_construction_does_not_fragment_dataframe(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error", pd.errors.PerformanceWarning)
            features = pipeline.feature_patient(patient_frame(), include_hemodynamics=True)
        self.assertIn("SBP_sampen_24h", features.columns)

    def test_sampen_oracle_zero_match_and_no_second_backend(self):
        # Four equal observations: B=3 m-template pairs, A=1 m+1 pair.
        self.assertAlmostEqual(pipeline.sample_entropy([1, 1, 1, 1]), -math.log(1 / 3))
        self.assertTrue(math.isnan(pipeline.sample_entropy([0, 1, 2, 3])))
        # The public pipeline intentionally has one implementation, so its
        # backend parity condition is identity rather than CPU/Numba disagreement.
        self.assertEqual(pipeline.sample_entropy([1, 1, 1, 1]), pipeline.sample_entropy(np.ones(4)))

    def test_sampen_support_counts_observations_not_forward_fill(self):
        patient = patient_frame()
        patient["HR"] = [80.0, np.nan, np.nan, 83.0]
        features = pipeline.feature_patient(patient, include_hemodynamics=True)
        self.assertEqual(features["HR_last_obs"].tolist(), [80.0, 80.0, 80.0, 83.0])
        self.assertEqual(features["HR_observation_age_hours"].tolist(), [0.0, 1.0, 2.0, 0.0])
        self.assertEqual(features["HR_sampen_effective_n_24h"].tolist(), [1, 1, 1, 2])
        self.assertTrue(features["HR_sampen_24h"].isna().all())
        self.assertNotIn("HR_sampen_effective_n_24h", pipeline.model_features(features, "enhanced"))
        support = pipeline.measurement_support_rows(features)
        hr_all = next(row for row in support if row["signal"] == "HR" and row["time_stratum"] == "all_hours")
        self.assertEqual(hr_all["median_effective_n"], 1.0)
        self.assertEqual(hr_all["fraction_meeting_sampen_minimum"], 0.0)

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

    def test_average_precision_and_pr_auc_are_named_distinct_estimands(self):
        y = np.array([0, 1, 0, 1])
        p = np.array([0.2, 0.8, 0.7, 0.9])
        metrics = pipeline.discrimination_metrics(y, p)
        self.assertAlmostEqual(metrics["average_precision"], pipeline.average_precision_score(y, p))
        self.assertIn("trapezoidal_pr_auc", metrics)
        self.assertEqual(metrics["xgboost_training_eval_metric"], "logloss")

    def test_dca_uses_patient_actions_and_counts_false_alerts(self):
        frame = pd.DataFrame({
            "Patient_ID": ["A:p1"] * 3 + ["A:p2"] * 3,
            "ICULOS": [1, 2, 3] * 2,
            "SepsisLabel": [1] * 3 + [0] * 3,
            "TrueSepsisOnset_ICULOS": [4.0] * 3 + [math.nan] * 3,
            "probability": [0.1, 0.8, 0.1, 0.9, 0.1, 0.1],
        })
        row = pipeline.decision_curve(frame, "probability", [0.5])[0]
        self.assertEqual((row["tp_patients"], row["fp_patients"]), (1, 1))
        self.assertAlmostEqual(row["model_net_benefit"], 0.0)
        self.assertIn("model_net_benefit_ci_95_low", row)

    def test_alarm_burden_reports_observed_time_and_refractory_episodes(self):
        septic_times = list(range(1, 21))
        frame = pd.DataFrame({
            "Patient_ID": ["A:p1"] * 20 + ["B:p1"] * 2,
            "ICULOS": septic_times + [1, 2],
            "SepsisLabel": [1] * 20 + [0] * 2,
            "TrueSepsisOnset_ICULOS": [20.0] * 20 + [math.nan] * 2,
            "probability": [float(hour in {2, 3, 8, 20}) for hour in septic_times] + [1.0, 1.0],
            "threshold": [0.5] * 22,
        })
        summary = pipeline.early_warning_metrics(frame, "probability", "threshold")["summary"]
        self.assertEqual(summary["n_alarm_episodes"], 4)
        self.assertEqual(summary["time_in_alert_observed_decision_hours"], 6)
        self.assertEqual(summary["repeated_alert_rows_suppressed_by_refractory_policy"], 2)
        self.assertEqual(summary["false_alarm_episodes"], 2)
        self.assertEqual(summary["post_onset_alarm_episodes"], 1)
        self.assertAlmostEqual(summary["time_in_alert_fraction_observed"], 6 / 22)
        self.assertIn("6h refractory", summary["alarm_episode_policy"])
        remote_only = frame.loc[frame["Patient_ID"] == "A:p1"].copy()
        remote_only["probability"] = 0.0
        remote_only.loc[remote_only["ICULOS"] == 1, "probability"] = 1.0
        remote_summary = pipeline.early_warning_metrics(remote_only, "probability", "threshold")["summary"]
        self.assertEqual((remote_summary["tp_patients"], remote_summary["fn_patients"]), (0, 1))
        self.assertTrue(math.isnan(remote_summary["median_lead_time_hours"]))
        self.assertEqual(remote_summary["false_alarm_episodes"], 1)

    def test_paired_lead_time_keeps_detection_denominators(self):
        rows = []
        for patient in ("p1", "p2"):
            for hour in range(1, 5):
                rows.append({"Patient_ID": patient, "ICULOS": hour, "SepsisLabel": int(hour >= 3), "Fold": 0, "TrueSepsisOnset_ICULOS": 5.0, "nested_threshold": 0.5})
        baseline = pd.DataFrame(rows)
        enhanced = baseline.copy()
        baseline["prob_platt"] = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
        enhanced["prob_platt"] = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        comparison = pipeline.paired_early_warning_comparison(baseline, enhanced)
        self.assertEqual((comparison["baseline_detected"], comparison["enhanced_detected"]), (2, 1))
        self.assertEqual((comparison["detected_by_both"], comparison["baseline_only"], comparison["missed_by_both"]), (1, 1, 0))
        self.assertEqual(comparison["median_enhanced_minus_baseline_lead_time_hours_among_both"], -1.0)
        self.assertIn("conditional on detection by both", comparison["interpretation"])

    def test_reporting_writes_metrics_from_supplied_oof(self):
        rows = []
        for patient in range(40):
            septic = patient % 2 == 0
            labels = [0, 0, 1, 1] if septic else [0, 0, 0, 0]
            probabilities = [0.10, 0.20, 0.70, 0.80] if septic else [0.10, 0.20, 0.30, 0.40]
            for hour, (label, probability) in enumerate(zip(labels, probabilities), start=1):
                rows.append({
                    "Patient_ID": f"A:p{patient:03d}", "ICULOS": hour,
                    "SepsisLabel": label, "TrueSepsisOnset_ICULOS": 4.0 if septic else math.nan,
                    "prob_raw": probability, "prob_platt": probability, "nested_threshold": 0.5,
                })
        oof = pd.DataFrame(rows)
        with tempfile.TemporaryDirectory() as directory:
            summary = pipeline.model_summary(oof, "oracle", Path(directory))
            emitted = json.loads((Path(directory) / "oracle_metrics.json").read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as directory:
            pipeline.model_summary(oof, "oracle", Path(directory), persist_artifacts=False)
            self.assertEqual(list(Path(directory).iterdir()), [])
        expected = pipeline.average_precision_score(oof["SepsisLabel"], oof["prob_raw"])
        self.assertAlmostEqual(summary["raw"]["average_precision"], expected)
        self.assertEqual(emitted["raw"]["average_precision"], summary["raw"]["average_precision"])
        self.assertAlmostEqual(summary["prevalence_only_brier_reference"], np.mean(oof["SepsisLabel"]) * (1 - np.mean(oof["SepsisLabel"])))
        composition = summary["positive_label_composition"]
        self.assertEqual((composition["pre_onset_rows"], composition["onset_or_post_onset_rows"]), (20, 20))
        self.assertEqual(composition["pre_onset_rows"] + composition["onset_or_post_onset_rows"] + composition["left_censored_onset_unidentifiable_rows"], summary["n_positive_rows"])
        self.assertIn("not equivalent", composition["interpretation"])
        self.assertIn("fold-specific monotone calibrators", summary["platt_nested"]["discrimination_interpretation"])
        self.assertIn("not a threshold for a final deployable model", summary["operating_policy_interpretation"])

    def test_temporal_strata_and_process_ablation_definitions(self):
        frame = pd.DataFrame({
            "Patient_ID": ["A:p1"] * 4 + ["B:p1"] * 4,
            "ICULOS": [1, 6, 7, 13] * 2,
            "TrueSepsisOnset_ICULOS": [14.0] * 4 + [math.nan] * 4,
            "SepsisLabel": [0, 0, 1, 1] + [0] * 4,
            "probability": [0.1, 0.2, 0.8, 0.9] + [0.1, 0.2, 0.3, 0.4],
        })
        strata = {(row["axis"], row["stratum"]): row for row in pipeline.temporal_stratified_metrics(frame, "probability")}
        self.assertEqual(strata[("time_since_icu_admission", "ICULOS_1_6h")]["n_rows"], 4)
        self.assertEqual(strata[("time_relative_to_true_onset", "useful_window_onset_minus_12_to_1h")]["n_rows"], 3)
        self.assertEqual(strata[("time_relative_to_true_onset", "post_onset_0h_plus")]["n_rows"], 0)
        age_frame = pd.DataFrame({
            "Patient_ID": ["p1", "p2", "p3", "p4", "p5"],
            "Age": [49.9, 50.0, 69.9, 70.0, math.nan],
            "SepsisLabel": [0, 1, 0, 1, 0],
            "probability": [0.1, 0.8, 0.2, 0.9, 0.3],
        })
        age = {row["subgroup"]: row for row in pipeline.age_subgroup_metrics(age_frame, "probability")}
        self.assertEqual([age[group]["n_rows"] for group in ("<50", "50_to_<70", ">=70", "missing")], [1, 2, 1, 1])
        self.assertTrue(all(row["subgroup_schema_version"] == "age_v1_left_closed_50_70" for row in age.values()))
        self.assertTrue(all(row["probability_source"] == "probability" for row in age.values()))

        features = pipeline.feature_patient(patient_frame(), include_hemodynamics=True)
        self.assertFalse(any(column.endswith("_shannon_5h") for column in features.columns))
        process = pipeline.ablation_columns(features, "without_explicit_process")
        physiology = pipeline.ablation_columns(features, "physiology_measurements_only")
        for columns in (process, physiology):
            self.assertFalse(any(column.endswith(("_is_missing", "_observation_age_hours")) for column in columns))
            self.assertTrue({"Unit1", "Unit2", "HospAdmTime", "ICULOS", "Measurement_Count"}.isdisjoint(columns))
            self.assertIn("HR_last_obs", columns)
        self.assertTrue({"Age", "Gender"}.issubset(process))
        self.assertTrue({"Age", "Gender"}.isdisjoint(physiology))

    def test_python_hash_seed_is_exported_before_python_starts(self):
        root = Path(__file__).resolve().parents[1]
        entrypoint = (root / "run.sh").read_text(encoding="utf-8")
        job = (root / "jobs" / "run_experiment.slurm").read_text(encoding="utf-8")
        self.assertLess(entrypoint.index("export PYTHONHASHSEED=20260906"), entrypoint.index("python scripts/source_provenance.py"))
        self.assertLess(entrypoint.index("python -m unittest"), entrypoint.index("python scripts/run_experiment.py --archive"))
        self.assertIn("RESUME_RUN_ID", entrypoint)
        self.assertIn("SCHEDULER_", entrypoint)
        self.assertIn("sacct -n -X", entrypoint)
        self.assertIn("export PYTHONWARNINGS=error", entrypoint)
        self.assertIn("PYTHONHASHSEED=20260906", job)
        self.assertIn("logs/run_ledger.tsv", job)
        self.assertIn("trap '", job)
        self.assertIn("CANCELLED_signal_TERM", job)
        self.assertIn("trap - EXIT", job)
        self.assertNotIn('os.environ["PYTHONHASHSEED"] =', (root / "src" / "scientific_pipeline.py").read_text(encoding="utf-8"))
        production = "\n".join(path.read_text(encoding="utf-8") for directory in (root / "src", root / "scripts") for path in directory.glob("*.py"))
        self.assertNotIn('filterwarnings("ignore")', production)

    def test_ece_definition_is_fixed_equal_width_bins(self):
        y = np.array([0, 1, 1, 0])
        p = np.array([0.1, 0.1, 0.9, 0.9])
        # Two occupied equal-width bins: .5*|.5-.1| + .5*|.5-.9| = .4.
        self.assertAlmostEqual(pipeline.calibration_metrics(y, p, bins=2)["ece_fixed_10_bins"], 0.4)

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
        for metric in ("brier", "ece_fixed_10_bins", "calibration_intercept", "calibration_slope"):
            self.assertLessEqual(report[f"{metric}_ci_95_low"], report[metric])
            self.assertGreaterEqual(report[f"{metric}_ci_95_high"], report[metric])

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
        self.assertNotIn("best_method", inspect.getsource(pipeline.model_summary))
        self.assertNotIn("quantile", inspect.getsource(pipeline.calibration_metrics))

    def test_matched_permutation_control_preserves_fold_margins(self):
        features = pd.DataFrame({
            "Patient_ID": [f"p{i}" for i in range(6)], "SourceSet": ["A"] * 6, "ICULOS": [1] * 6,
            "SepsisLabel": [0, 1, 0, 1, 0, 1], "TrueSepsisOnset_ICULOS": [math.nan] * 6,
            "Age": np.arange(6.0), "Hct_last_obs": np.arange(10.0, 16.0), "HR_cv_8h": np.arange(20.0, 26.0),
        })
        folds = pd.DataFrame({"Patient_ID": features["Patient_ID"], "SepsisLabel": features["SepsisLabel"], "Fold": [0, 0, 0, 1, 1, 1]})
        control, columns = pipeline.matched_permutation_control(features, folds)
        second, second_columns = pipeline.matched_permutation_control(features, folds)
        self.assertEqual(columns, second_columns)
        pd.testing.assert_frame_equal(control, second)
        self.assertEqual(control["Age"].tolist(), features["Age"].tolist())
        for fold in (0, 1):
            patients = folds.loc[folds["Fold"] == fold, "Patient_ID"]
            self.assertEqual(sorted(control.loc[control["Patient_ID"].isin(patients), "HR_cv_8h"]), sorted(features.loc[features["Patient_ID"].isin(patients), "HR_cv_8h"]))

    def test_logistic_robustness_reuses_grouped_folds(self):
        features = pd.DataFrame([
            {"Patient_ID": f"p{i:02d}", "SourceSet": "A", "ICULOS": 1, "SepsisLabel": i % 2, "TrueSepsisOnset_ICULOS": math.nan,
             "Age": 40.0 + i, "Hct_last_obs": 30.0 + i, "HR_cv_8h": float(i % 3)}
            for i in range(20)
        ])
        folds = pd.DataFrame({"Patient_ID": features["Patient_ID"], "SepsisLabel": features["SepsisLabel"], "Fold": [i % 5 for i in range(20)]})
        rows = pipeline.logistic_representation_robustness(features, folds)
        self.assertEqual([row["model_variant"] for row in rows], ["baseline", "enhanced"])
        self.assertEqual(rows[0]["split_hash"], rows[1]["split_hash"])
        self.assertTrue(all(row["classifier"] == "sklearn_SGDClassifier_log_loss_l2" for row in rows))

    def test_stage_checkpoint_resume_is_hash_and_context_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            calls = []
            def producer(output):
                calls.append(1)
                output.write_text("valid\n", encoding="utf-8")
                return {"stage": "features", "artifact_sha256": pipeline.sha256_file(output), "input_sha256": "a" * 64}
            first = pipeline.stage_checkpoint(root, "features", producer, {"input_sha256": "a" * 64})
            second = pipeline.stage_checkpoint(root, "features", producer, {"input_sha256": "a" * 64})
            self.assertEqual((first, len(calls)), (second, 1))
            (root / "features.csv").write_text("tampered\n", encoding="utf-8")
            with self.assertRaises(pipeline.PipelineError):
                pipeline.stage_checkpoint(root, "features", producer, {"input_sha256": "a" * 64})
            (root / "baseline_oof_predictions.csv").write_text("partial\n", encoding="utf-8")
            with self.assertRaises(pipeline.PipelineError):
                pipeline.require_clean_resume_boundary(root)

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
            source = inspect.getsource(pipeline.validate_final_manifest)
            self.assertIn("COMPUTATIONAL_RUN_VALIDATED", source)
            self.assertIn('final_validation"].get("status") != "PASS"', source)

    def test_oof_artifacts_do_not_duplicate_feature_matrix(self):
        self.assertEqual(len(pipeline.OOF_OUTPUT_COLUMNS), len(set(pipeline.OOF_OUTPUT_COLUMNS)))
        self.assertIn("Age", pipeline.OOF_OUTPUT_COLUMNS)
        self.assertNotIn("Hct_last_obs", pipeline.OOF_OUTPUT_COLUMNS)
        self.assertNotIn("model_variant", pipeline.OOF_OUTPUT_COLUMNS)
        self.assertIn("records.append(outer_test[OOF_OUTPUT_COLUMNS])", inspect.getsource(pipeline.outer_oof))
        self.assertIn("if persist_oof:", inspect.getsource(pipeline.outer_oof))
        self.assertIn("if persist_artifacts:", inspect.getsource(pipeline.model_summary))
        validator = inspect.getsource(pipeline.validate_final_manifest)
        self.assertIn('pd.read_csv(run_dir / "features.csv", nrows=0)', validator)
        self.assertIn("list(oof.columns) != OOF_OUTPUT_COLUMNS", validator)

    def test_nested_calibration_and_threshold_provenance(self):
        inner = pd.DataFrame({
            "Patient_ID": ["A:p1"] * 4 + ["A:p2"] * 4,
            "ICULOS": list(range(1, 5)) * 2,
            "SepsisLabel": [0, 0, 1, 1, 0, 0, 1, 1],
            "inner_prob_raw": [0.1, 0.2, 0.8, 0.9, 0.1, 0.2, 0.8, 0.9],
            "InnerFold": [0] * 4 + [1] * 4,
        })
        calibrator = pipeline.fitted_platt(inner)
        inner["inner_prob_platt"] = pipeline.platt_probabilities(calibrator, inner["inner_prob_raw"])
        held_out = pd.DataFrame({"prob_raw": [0.15, 0.85]})
        held_out["prob_platt"] = pipeline.platt_probabilities(calibrator, held_out["prob_raw"])
        self.assertTrue(np.isfinite(held_out["prob_platt"]).all())
        threshold = pipeline.threshold_from_inner_oof(inner, "inner_prob_platt")
        self.assertIn(threshold, pipeline.FEATURE_POLICY["threshold_grid"])
        self.assertNotIn("Fold", inner.columns)  # Outer held-out rows cannot calibrate themselves.

    def test_data_license_notice_distinguishes_local_repackaging(self):
        root = Path(__file__).resolve().parents[1]
        notice = (root / "docs" / "data_license.md").read_text(encoding="utf-8")
        self.assertIn("CC BY 4.0", notice)
        self.assertIn("official PhysioNet archive", notice)
        self.assertIn("repackaging", notice)
        policy = (root / "docs" / "results_policy.md").read_text(encoding="utf-8")
        self.assertIn("tracked, required raw-data dependency", policy)
        self.assertIn("not claimed to recreate this local repackaging", policy)

    def test_historical_result_certifications_are_visibly_withdrawn(self):
        root = Path(__file__).resolve().parents[1]
        marker = "> **WITHDRAWN — HISTORICAL INVALID OUTPUT.**"
        for relative in (
            "code_math_audit_summary.md", "completion_checklist.md", "calibration_audit.md",
            "statistical_methods_notes.md", "statistics_summary.md", "results_snippet.md",
        ):
            self.assertTrue((root / "results" / "statistics" / relative).read_text(encoding="utf-8").startswith(marker))

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
            lineage["result"]["inputs"]["source.csv"] = source_hash
            result.write_text("tampered\n", encoding="utf-8")
            with self.assertRaises(pipeline.PipelineError):
                pipeline.validate_lineage_nodes(root, lineage)

    def test_reporting_traceability_and_fail_closed_source_set(self):
        with self.assertRaises(pipeline.PipelineError):
            pipeline.source_and_patient("Dataset.psv")
        required = {"runtime", "stages", "artifact_sha256", "lineage", "final_validation", "scientific_status"}
        self.assertTrue(required.issuperset({"runtime", "lineage"}))
        self.assertIn("execution_environment", pipeline.runtime_manifest(Path.cwd(), "test", sys.argv, Path(__file__)))


if __name__ == "__main__":
    unittest.main()
