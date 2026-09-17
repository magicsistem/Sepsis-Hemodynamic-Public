"""Independent oracles for the direct-onset Koopman experiment."""

from __future__ import annotations

import math
import inspect
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from src import onset_koopman as koopman
from src import scientific_pipeline as pipeline
from scripts import resource_provenance


def patient(patient_id: str, times, labels, onset, status, hr) -> pd.DataFrame:
    return pd.DataFrame({
        "Patient_ID": patient_id,
        "SourceSet": patient_id.split(":", 1)[0],
        "ICULOS": times,
        "SepsisLabel": labels,
        "TrueSepsisOnset_ICULOS": onset,
        "OnsetReconstructionStatus": status,
        "raw__HR": hr,
        "raw__MAP": np.asarray(hr, dtype=float) / 2,
    })


class OnsetTargetTests(unittest.TestCase):
    def test_target_window_and_exclusions(self):
        septic = patient(
            "A:septic", np.arange(1, 11), [0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
            10.0, "exact_from_shift_transition", np.arange(10.0),
        )
        control = patient(
            "B:control", np.arange(1, 11), np.zeros(10), math.nan,
            "nonseptic", np.arange(10.0),
        )
        left = patient(
            "A:left", np.arange(1, 5), np.ones(4), math.nan,
            "septic_onset_left_censored", np.arange(4.0),
        )
        result = koopman.add_primary_target(pd.concat([septic, control, left], ignore_index=True))
        septic_result = result[result.Patient_ID == "A:septic"]
        self.assertEqual(septic_result[koopman.ELIGIBLE_COLUMN].tolist(), [1] * 9 + [0])
        self.assertEqual(septic_result.loc[septic_result[koopman.ELIGIBLE_COLUMN] == 1, koopman.TARGET_COLUMN].tolist(), [0, 0, 0, 1, 1, 1, 1, 1, 1])
        control_result = result[result.Patient_ID == "B:control"]
        self.assertEqual(control_result[koopman.ELIGIBLE_COLUMN].tolist(), [1] * 4 + [0] * 6)
        self.assertEqual(control_result.loc[control_result[koopman.ELIGIBLE_COLUMN] == 1, koopman.TARGET_COLUMN].tolist(), [0] * 4)
        left_result = result[result.Patient_ID == "A:left"]
        self.assertEqual(left_result[koopman.ELIGIBLE_COLUMN].sum(), 0)
        self.assertTrue(left_result[koopman.TARGET_COLUMN].isna().all())
        self.assertTrue(result.loc[result[koopman.ELIGIBLE_COLUMN] == 0, koopman.TARGET_COLUMN].isna().all())

        inconsistent = septic.copy()
        inconsistent["TrueSepsisOnset_ICULOS"] = 9.0
        with self.assertRaisesRegex(koopman.OnsetKoopmanError, "Challenge shift"):
            koopman.add_primary_target(inconsistent)

    def test_target_is_no_future_stable(self):
        frame = patient(
            "A:septic", np.arange(1, 9), [0, 0, 1, 1, 1, 1, 1, 1],
            9.0, "exact_from_shift_transition", np.arange(8.0),
        )
        expected = koopman.add_primary_target(frame).iloc[:4][[koopman.TARGET_COLUMN, koopman.ELIGIBLE_COLUMN]]
        changed = frame.copy()
        changed.loc[changed.index[-1], "raw__HR"] = 9999
        observed = koopman.add_primary_target(changed).iloc[:4][[koopman.TARGET_COLUMN, koopman.ELIGIBLE_COLUMN]]
        pd.testing.assert_frame_equal(expected, observed)


class KoopmanOracleTests(unittest.TestCase):
    def training_frame(self) -> pd.DataFrame:
        frames = []
        for index in range(3):
            values = np.zeros(8) if index < 2 else np.arange(8.0)
            frames.append(patient(
                f"A:p{index}", np.arange(1, 9), np.zeros(8), math.nan,
                "nonseptic", values,
            ))
        return koopman.add_primary_target(pd.concat(frames, ignore_index=True).sort_values(["Patient_ID", "ICULOS"]).reset_index(drop=True))

    def test_fold_local_support_and_fixed_schema(self):
        train = self.training_frame()
        selected = koopman.select_dynamic_signals(train, ("HR", "MAP"), 0.5, 2, 1)
        self.assertEqual(selected, ("HR",))  # equal coverage, alphabetical tie-break
        fit = koopman.fit_koopman(train, ("HR", "MAP"), "identity", selected_signals=("HR",))
        transformed = koopman.transform_koopman(train, fit)
        self.assertEqual(
            [column for column in transformed if column.startswith("koopman_innovation__")],
            ["koopman_innovation__HR", "koopman_innovation__MAP"],
        )
        self.assertTrue(transformed["koopman_innovation__MAP"].isna().all())
        self.assertEqual(fit.training_patient_hash, koopman._training_patient_hash(train))
        self.assertEqual(fit.training_transition_counts, {"HR": 21})
        with mock.patch.dict(koopman.KOOPMAN_POLICY, {"maximum_training_transitions_per_signal": 3}):
            first = koopman.fit_koopman(train, ("HR",), "identity", selected_signals=("HR",))
            second = koopman.fit_koopman(train, ("HR",), "identity", selected_signals=("HR",))
        self.assertEqual(first.training_transition_counts, {"HR": 3})
        np.testing.assert_array_equal(first.models["HR"].coef_, second.models["HR"].coef_)
        with mock.patch.dict(koopman.os.environ, {"SLURM_CPUS_PER_TASK": "33"}):
            with self.assertRaises(koopman.OnsetKoopmanError):
                koopman.fit_koopman(train, ("HR",), "identity", selected_signals=("HR",))

        imminent = train.copy()
        imminent["OnsetReconstructionStatus"] = "exact_from_shift_transition"
        imminent[koopman.HOURS_TO_ONSET_COLUMN] = 1.0
        fitted = koopman.fit_koopman(imminent, ("HR",), "identity", selected_signals=("HR",))
        self.assertEqual(fitted.training_transition_counts, {"HR": 0})
        self.assertTrue(koopman.transform_koopman(imminent, fitted)["koopman_innovation__HR"].isna().all())

    def test_zero_system_anomaly_and_observed_only_residuals(self):
        train = self.training_frame()
        train = train[train.Patient_ID.isin(["A:p0", "A:p1"])].reset_index(drop=True)
        fit = koopman.fit_koopman(train, ("HR",), "identity", selected_signals=("HR",))
        zero = train[train.Patient_ID == "A:p0"].copy().reset_index(drop=True)
        transformed = koopman.transform_koopman(zero, fit)
        self.assertTrue(math.isnan(transformed.iloc[0]["koopman_innovation__HR"]))
        self.assertTrue(np.allclose(transformed["koopman_innovation__HR"].iloc[1:], 0.0, atol=1e-6))
        anomaly = zero.copy()
        anomaly.loc[anomaly.index[-1], "raw__HR"] = 10.0
        observed = koopman.transform_koopman(anomaly, fit)
        self.assertGreater(observed.loc[observed.index[-1], "koopman_innovation__HR"], 0)
        missing = anomaly.copy()
        missing.loc[missing.index[-1], "raw__HR"] = np.nan
        self.assertTrue(math.isnan(koopman.transform_koopman(missing, fit).iloc[-1]["koopman_innovation__HR"]))

    def test_no_future_identity_quadratic_float32_and_deltas(self):
        train = self.training_frame()
        first = train[train.Patient_ID == "A:p2"].copy().reset_index(drop=True)
        changed = first.copy()
        changed.loc[changed.index[-1], "raw__HR"] = 1000.0
        for lift in koopman.KOOPMAN_POLICY["lifts"]:
            fit = koopman.fit_koopman(train, ("HR",), lift, selected_signals=("HR",))
            before = koopman.transform_koopman(first, fit)
            after = koopman.transform_koopman(changed, fit)
            pd.testing.assert_frame_equal(before.iloc[:-1], after.iloc[:-1])
            self.assertEqual(before["koopman_innovation__HR"].dtype, np.float32)
        delta = koopman.transform_deltas(changed, ("HR", "MAP"), ("HR",))
        self.assertTrue(delta["causal_delta__MAP"].isna().all())
        self.assertEqual(delta.iloc[-1]["causal_delta__HR"], 994.0)
        self.assertEqual(delta.iloc[-1]["causal_slope__HR"], 994.0)


def primary_oof() -> pd.DataFrame:
    frames = []
    for patient_id, septic in (("A:s1", True), ("A:s2", True), ("A:c1", False), ("A:c2", False)):
        times = np.arange(1, 12)
        onset = 10.0 if septic else math.nan
        frame = patient(
            patient_id,
            times,
            [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1] if septic else np.zeros(11),
            onset,
            "exact_from_shift_transition" if septic else "nonseptic",
            np.zeros(11),
        )
        frames.append(frame)
    result = koopman.add_primary_target(pd.concat(frames, ignore_index=True))
    result["prob_raw"] = np.where(result[koopman.TARGET_COLUMN] == 1, 0.8, 0.2)
    result["prob_calibrated"] = result["prob_raw"]
    result["nested_alarm_threshold"] = 0.5
    return result


class NestedPolicyTests(unittest.TestCase):
    def test_calibration_and_threshold_use_inner_oof_contract(self):
        inner = primary_oof()
        calibration = koopman.fit_calibration_policy(inner)
        probability = koopman.apply_calibration(calibration, inner["prob_raw"])
        self.assertTrue(np.isfinite(probability).all())
        inner["prob_calibrated"] = probability
        threshold, alarm = koopman.select_alarm_threshold(
            inner, "prob_calibrated", thresholds=(0.01, 0.5, 0.99)
        )
        self.assertLessEqual(alarm["false_alarm_episodes_per_patient_day"], 0.25)
        self.assertIn(threshold, (0.01, 0.5, 0.99))

        source = inspect.getsource(pipeline.primary_outer_oof)
        self.assertIn("select_inner_primary_model(\n            outer_train", source)
        self.assertIn('fit_calibration_policy(inner_oof, "prob_raw")', source)
        self.assertIn('select_alarm_threshold(inner_oof, "prob_calibrated")', source)
        self.assertLess(source.index("select_inner_primary_model"), source.index("transformed_test"))

    def test_nested_provenance_rejects_transition_count_tampering(self):
        signals = ["HR"]
        candidate = pipeline.MODEL_CANDIDATES[0]
        selection = pd.DataFrame([{
            "outer_fold": fold,
            "selected_candidate": candidate["id"],
            "selected_hyperparameters": json.dumps(candidate, sort_keys=True),
            "selected_lift": "identity",
            "selected_tree_count_from_inner_only": 10,
            "selected_signals_outer_train": json.dumps(signals),
            "selected_signals_hash": pipeline.stable_hash(signals),
            "koopman_training_transition_counts": json.dumps({"HR": 2}),
            "feature_count": 5,
            "feature_column_hash": "a" * 64,
            "calibrator_selected_by_inner_oof_brier": "identity",
            "nested_alarm_threshold_from_inner_oof_only": 0.5,
            "inner_oof_alarm_budget": 0.1,
            "inner_oof_useful_sensitivity": 0.5,
            "outer_train_patient_hash": "b" * 64,
            "outer_test_patient_hash": "c" * 64,
            "outer_train_patient_count": 2,
            "outer_test_patient_count": 2,
        } for fold in range(2)])
        inner = pd.DataFrame([{
            "representation": "C3", "outer_fold": outer, "lift": lift,
            "candidate": model["id"], "inner_fold": inner_fold,
            "patient_balanced_average_precision": 0.5, "best_round": 10,
            "selected_signals": json.dumps(signals),
            "koopman_training_transition_counts": json.dumps({"HR": 2}),
            "fit_patient_hash": "d" * 64, "valid_patient_hash": "e" * 64,
        } for outer in range(2) for lift in koopman.KOOPMAN_POLICY["lifts"]
          for model in pipeline.MODEL_CANDIDATES for inner_fold in range(2)])
        with mock.patch.dict(pipeline.MODEL_POLICY, {"outer_folds": 2, "inner_folds": 2}), \
             mock.patch.dict(pipeline.DATA_POLICY, {"patient_count": 4}):
            pipeline.validate_primary_nested_provenance(selection, inner, "C3")
            tampered = inner.copy()
            tampered.loc[0, "koopman_training_transition_counts"] = json.dumps({"HR": 20001})
            with self.assertRaises(pipeline.PipelineError):
                pipeline.validate_primary_nested_provenance(selection, tampered, "C3")

    def test_primary_feature_schemas_exclude_provenance_and_outcomes(self):
        base = self.training_features()
        for representation in koopman.REPRESENTATIONS:
            frame = base.copy()
            if representation == "C2":
                frame = pd.concat([
                    frame,
                    koopman.transform_deltas(frame, pipeline.DYNAMIC_COLUMNS, ("HR",)),
                ], axis=1)
            elif representation == "C3":
                for signal in pipeline.DYNAMIC_COLUMNS:
                    frame[koopman.innovation_column(signal)] = np.nan
                for name in (
                    "koopman_energy", "koopman_energy_mean_8h", "koopman_energy_max_8h",
                    "koopman_innovation_count",
                ):
                    frame[name] = np.nan
            columns = pipeline.primary_model_features(frame, representation)
            self.assertTrue({"SourceSet", koopman.TARGET_COLUMN, "SepsisLabel"}.isdisjoint(columns))

    def test_representation_fit_excludes_left_censored_patients(self):
        frame = pd.concat([
            patient(
                "A:control", [1, 2], [0, 0], math.nan, "nonseptic", [60.0, 61.0]
            ),
            patient(
                "A:left", [1, 2], [1, 1], math.nan,
                "septic_onset_left_censored", [999.0, 999.0],
            ),
        ], ignore_index=True)
        frame = koopman.add_primary_target(frame)
        with mock.patch.object(koopman, "select_dynamic_signals", return_value=("HR",)) as select:
            pipeline.fit_primary_representation(frame, "C2")
        fitted_population = select.call_args.args[0]
        self.assertEqual(set(fitted_population["Patient_ID"]), {"A:control"})

    @staticmethod
    def training_features() -> pd.DataFrame:
        row = {
            "Patient_ID": "A:p", "SourceSet": "A", "ICULOS": 1, "Age": 60,
            "SepsisLabel": 0, "TrueSepsisOnset_ICULOS": math.nan,
            "OnsetReconstructionStatus": "nonseptic", koopman.TARGET_COLUMN: 0,
            koopman.ELIGIBLE_COLUMN: 1, koopman.HOURS_TO_ONSET_COLUMN: math.nan,
            "Hct_last_obs": 30.0,
        }
        for signal in pipeline.DYNAMIC_COLUMNS:
            row[f"{signal}_last_obs"] = 1.0
            row[f"{signal}_is_missing"] = 0
            row[f"{signal}_observation_age_hours"] = 0.0
            row[koopman.raw_column(signal)] = 1.0
        for signal in pipeline.HEMODYNAMIC_COLUMNS:
            row[f"{signal}_cv_8h"] = 0.0
            row[f"{signal}_iqr_8h"] = 0.0
            row[f"{signal}_sampen_24h"] = 0.0
            row[f"{signal}_sampen_24h_zero_match"] = 0
        return pd.DataFrame([row])


class InferenceTransportTests(unittest.TestCase):
    def test_patient_balanced_ap_and_paired_patient_bootstrap(self):
        candidate = primary_oof()
        comparator = candidate.copy()
        comparator["prob_calibrated"] = 0.5
        decisions = koopman.primary_decisions(candidate)
        expected = pipeline.average_precision_score(
            decisions[koopman.TARGET_COLUMN],
            decisions["prob_calibrated"],
            sample_weight=koopman.equal_patient_weights(decisions),
        )
        self.assertAlmostEqual(
            koopman.patient_balanced_average_precision(candidate, "prob_calibrated"), expected
        )
        rows = koopman.paired_patient_bootstrap(
            comparator, candidate, repeats=20, seed=7
        )
        self.assertEqual({row["inference_unit"] for row in rows}, {"patient"})
        self.assertGreater(
            next(row for row in rows if row["metric"] == "patient_balanced_average_precision")["candidate_minus_comparator"],
            0,
        )

    def test_alarm_window_budget_and_dca(self):
        frame = primary_oof()
        alarm = koopman.alarm_metrics(frame, "prob_calibrated", 0.5)
        self.assertEqual(alarm["tp_patients"], 2)
        self.assertEqual(alarm["false_alarm_episodes"], 0)
        self.assertEqual(alarm["useful_sensitivity"], 1.0)
        dca = koopman.decision_curve(frame, "prob_calibrated", (0.25,), repeats=20, seed=9)
        self.assertEqual(dca[0]["uncertainty_unit"], "patient-cluster bootstrap")
        self.assertEqual(dca[0]["outcome_estimand"], "true onset in 1--6 hours")

    def test_transport_contract_never_fits_destination_labels(self):
        source = inspect.getsource(pipeline.fit_primary_source_transport)
        self.assertIn("select_inner_primary_model(\n        train", source)
        self.assertIn('"source_set_is_predictor": False', source)
        self.assertIn('"destination_labels_used_for_fitting": False', source)
        self.assertNotIn("select_inner_primary_model(\n        test", source)

    def test_direct_onset_lineage_is_transitive_and_tamper_evident(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            raw = run_dir / "archive.zip"
            names = ["harmonized.csv", "features.csv", "folds.csv"]
            for representation in koopman.REPRESENTATIONS:
                names.extend([
                    f"{representation}_nested_selection.csv",
                    f"{representation}_inner_selection.csv",
                    f"{representation}_oof_predictions.csv",
                    f"{representation}_metrics.json",
                    f"{representation}_dca.csv",
                ])
            names.extend([
                "transport.csv", "inference.csv", "metrics.json",
                "scientific_gate_status.json", "probast_ai_status.json",
            ])
            raw.write_bytes(b"raw")
            for name in names:
                (run_dir / name).write_text(name, encoding="utf-8")
            runtime = {"git_commit": "a" * 40, "data_archive_path": str(raw)}
            lineage = pipeline.direct_onset_lineage(run_dir, runtime)
            self.assertEqual(pipeline.validate_lineage_nodes(run_dir, lineage), 29)
            (run_dir / "C3_oof_predictions.csv").write_text("tampered", encoding="utf-8")
            with self.assertRaises(pipeline.PipelineError):
                pipeline.validate_lineage_nodes(run_dir, lineage)


class ResourceOrchestrationTests(unittest.TestCase):
    def test_failed_gnu_time_diagnostic_is_parsed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "time.txt"
            path.write_text(
                "Command exited with non-zero status 1\n"
                "elapsed_seconds=2\nuser_seconds=1\nsystem_seconds=0.5\n"
                "max_rss_kb=1024\nexit_status=1\n",
                encoding="utf-8",
            )
            self.assertEqual(resource_provenance.parse_time(path)["exit_status"], 1.0)

    def test_profile_selection_caps_efficiency_and_memory_margin(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profiles = [
                ("cpu8", 8, 0, 10.0, 0.90),
                ("cpu16", 16, 0, 8.0, 0.85),
                ("cpu32", 32, 0, 7.8, 0.82),
                ("gpu8", 8, 1, 7.2, 0.50),
                ("gpu16", 16, 1, 7.0, 0.50),
                ("gpu32", 32, 1, 7.1, 0.50),
            ]
            for name, cpus, gpus, elapsed, efficiency in profiles:
                payload = {
                    "stage": name, "status": "PASS", "hostname": "compute-0-2",
                    "slurm_job_id": name,
                    "requested": {"cpus": cpus, "memory_gb": 32, "gpus": gpus},
                    "measured": {
                        "elapsed_seconds": elapsed, "cpu_efficiency": efficiency,
                        "max_rss_gb": 8.0,
                    },
                    "benchmark": {"estimated_full_peak_gb": 10.0},
                }
                (root / f"{name}.json").write_text(json.dumps(payload), encoding="utf-8")
            output = root / "selection.json"
            selected = resource_provenance.select_profile(Namespace(profile_dir=root, output=output))
            self.assertEqual(selected["selected"], {"cpus": 8, "memory_gb": 12, "gpus": 1})
            self.assertLessEqual(selected["selected"]["cpus"], 32)
            self.assertLessEqual(selected["selected"]["memory_gb"], 64)

    def test_one_entrypoint_compute_node_and_fail_closed_dependencies(self):
        root = Path(__file__).resolve().parents[1]
        entrypoint = (root / "run.sh").read_text(encoding="utf-8")
        job = (root / "jobs" / "run_experiment.slurm").read_text(encoding="utf-8")
        self.assertIn('TEST_RUN_DIR="$ROOT/runs/${RUN_ID}-tests-${TEST_SUFFIX}"', entrypoint)
        self.assertIn('stage_run_dir=${7:-$RUN_DIR}', entrypoint)
        self.assertIn('--dependency="afterok:$dependency"', entrypoint)
        self.assertIn('--partition="$partition"', entrypoint)
        self.assertIn('partition=cpu', entrypoint)
        self.assertIn('partition=gpu', entrypoint)
        self.assertIn("RESUME_RUN_ID", entrypoint)
        self.assertIn("verify-stage", entrypoint)
        self.assertIn("--nodelist=compute-0-2", job)
        self.assertNotIn("compute-0-1", job)
        self.assertIn("set -euo pipefail", entrypoint)
        self.assertIn("set -euo pipefail", job)
        self.assertIn("host_python=unavailable", job)
        self.assertIn('"$RUNTIME" exec', job)
        self.assertEqual(resource_provenance.CPU_CAP, 32)
        self.assertEqual(resource_provenance.MEMORY_CAP_GB, 64)
        self.assertEqual(resource_provenance.GPU_CAP, 1)


class MethodologyDocumentTests(unittest.TestCase):
    def test_plan_is_current_reproducible_and_not_manuscript(self):
        root = Path(__file__).resolve().parents[1]
        path = root / "docs" / "PLAN_METODOLOGICO_PREDICCION_SEPSIS.md"
        self.assertTrue(path.is_file())
        text = path.read_text(encoding="utf-8")
        for required in (
            "Y_{i,t}", "C0", "C1", "C2", "C3", "Koopman", "EDMD",
            "nested", "A → B", "B → A", "BLOCKED_EXTERNAL_DATA",
            "Zahibi", "Zabihi", "IEEE", "2021–2026", "consulta reproducible",
            "No es el manuscrito",
        ):
            self.assertIn(required, text)
        self.assertNotIn("EXPECTED_AUROC", text)
        self.assertGreaterEqual(text.count("doi.org/"), 8)


if __name__ == "__main__":
    unittest.main()
