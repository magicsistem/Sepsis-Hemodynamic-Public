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
from scripts import profile_resources, resource_provenance


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

    def test_previous_state_expires_values_older_than_policy(self):
        frame = patient(
            "A:stale", [1, 26, 27, 28], [0, 0, 0, 0], math.nan,
            "nonseptic", [10.0, math.nan, 20.0, math.nan],
        )
        state, previous = koopman._previous_state(frame, ("HR",))
        self.assertTrue(previous.tolist() == [False, True, True, True])
        self.assertTrue(math.isnan(state[1, 0]))
        self.assertEqual(state[1, 1], koopman.KOOPMAN_POLICY["maximum_observation_age_hours"])
        self.assertTrue(math.isnan(state[2, 0]))
        self.assertEqual((state[3, 0], state[3, 1]), (20.0, 1.0))

    def test_vectorized_multisignal_state_and_delta_oracle(self):
        frame = patient(
            "A:multi", [1, 2, 4, 5], [0, 0, 0, 0], math.nan,
            "nonseptic", [10.0, math.nan, 16.0, 19.0],
        )
        frame["raw__MAP"] = [5.0, 7.0, math.nan, 11.0]
        state, previous = koopman._previous_state(frame, ("HR", "MAP"))
        np.testing.assert_allclose(
            state,
            [
                [math.nan, math.nan, math.nan, math.nan],
                [10.0, 1.0, 5.0, 1.0],
                [10.0, 3.0, 7.0, 2.0],
                [16.0, 1.0, 7.0, 3.0],
            ],
            equal_nan=True,
        )
        self.assertEqual(previous.tolist(), [False, True, True, True])
        transformed = koopman.transform_deltas(
            frame, ("HR", "MAP"), ("HR", "MAP")
        )
        np.testing.assert_allclose(
            transformed["causal_delta__HR"],
            [math.nan, math.nan, 6.0, 3.0], equal_nan=True,
        )
        np.testing.assert_allclose(
            transformed["causal_slope__HR"],
            [math.nan, math.nan, 2.0, 3.0], equal_nan=True,
        )
        np.testing.assert_allclose(
            transformed["causal_delta__MAP"],
            [math.nan, 2.0, math.nan, 4.0], equal_nan=True,
        )
        np.testing.assert_allclose(
            transformed["causal_slope__MAP"],
            [math.nan, 2.0, math.nan, 4.0 / 3.0], equal_nan=True,
        )
        rolling_frame = frame.iloc[:3].copy()
        rolling_frame["ICULOS"] = [1, 8, 9]
        mean, maximum = koopman._rolling_energy(
            rolling_frame, np.asarray([1.0, 2.0, 3.0]), 8
        )
        np.testing.assert_allclose(mean, [1.0, 1.5, 2.5])
        np.testing.assert_allclose(maximum, [1.0, 2.0, 3.0])


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
    def test_inner_selection_executes_patient_partitions(self):
        class Model:
            best_iteration = 0
            n_estimators = 1

            def fit(self, *_args, **_kwargs):
                return self

            def predict_proba(self, values):
                probability = np.asarray(values[:, 0], dtype=float)
                return np.column_stack([1.0 - probability, probability])

        rows = []
        for index in range(12):
            target = index % 2
            rows.append({
                "Patient_ID": f"A:p{index:02d}", "SourceSet": "A", "ICULOS": 1,
                "Age": 0.8 if target else 0.2, "SepsisLabel": target,
                "TrueSepsisOnset_ICULOS": 2.0 if target else math.nan,
                "OnsetReconstructionStatus": "exact_from_shift_transition" if target else "nonseptic",
                koopman.TARGET_COLUMN: target, koopman.ELIGIBLE_COLUMN: 1,
            })
        train = pd.DataFrame(rows)
        patient_rows = train[["Patient_ID", koopman.TARGET_COLUMN]].copy()
        splits = []
        indices = np.arange(len(patient_rows))
        for start in range(0, len(patient_rows), 4):
            valid = indices[start:start + 4]
            splits.append((np.setdiff1d(indices, valid), valid))
        with (
            mock.patch.object(pipeline, "primary_patient_splits", return_value=(patient_rows, splits)),
            mock.patch.object(pipeline, "primary_model_features", return_value=["Age"]),
            mock.patch.object(pipeline, "xgb_model", side_effect=lambda *_args, **_kwargs: Model()),
        ):
            _, _, _, inner_oof, detail = pipeline.select_inner_primary_model(
                train, "C0", {"available": False}, outer_fold=0
            )
        self.assertEqual(set(inner_oof["Patient_ID"]), set(train["Patient_ID"]))
        self.assertFalse(inner_oof.duplicated(["Patient_ID", "ICULOS"]).any())
        self.assertEqual(len(detail), len(pipeline.MODEL_CANDIDATES) * pipeline.MODEL_POLICY["inner_folds"])

    def test_parallel_worker_budget_and_order_are_fail_closed(self):
        cpu = {"available": False}
        with mock.patch.dict(
            pipeline.os.environ,
            {
                "SLURM_CPUS_PER_TASK": "64",
                "SEPSIS_FIT_THREADS": "16",
                "SEPSIS_PARALLEL_WORKERS": "2",
                "SEPSIS_PARALLEL_CANDIDATES": "2",
            },
            clear=True,
        ):
            self.assertEqual(pipeline.allocated_total_cpu_count(), 64)
            self.assertEqual(pipeline.allocated_cpu_count(), 16)
            self.assertEqual(pipeline.parallel_candidate_workers(cpu, 2), 2)
            self.assertEqual(pipeline.parallel_model_workers(cpu, 5), 2)
            self.assertEqual(
                pipeline.ordered_parallel_map(lambda value: value * 2, [3, 1, 2], 2),
                [6, 2, 4],
            )
            pipeline.os.environ["SEPSIS_PARALLEL_WORKERS"] = "3"
            with self.assertRaises(pipeline.PipelineError):
                pipeline.parallel_model_workers(cpu, 5)
            pipeline.os.environ["SEPSIS_PARALLEL_WORKERS"] = "2"
            with self.assertRaises(pipeline.PipelineError):
                pipeline.parallel_model_workers({"available": True}, 5)

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
        inner_source = inspect.getsource(pipeline.select_inner_primary_model)
        self.assertEqual(inner_source.count("fit_xgb("), 1)
        self.assertIn("candidate_probabilities[(winner_lift, winner_id)]", inner_source)

    def test_seed_and_balance_sensitivities_report_every_configuration(self):
        configurations = pipeline.robustness_configurations()
        self.assertEqual(len(configurations), 5)
        self.assertEqual(len(set(configurations)), 5)
        self.assertEqual(
            {seed for seed, balance, _ in configurations if balance == "equal_patient"},
            set(pipeline.ROBUSTNESS_POLICY["training_seed_bases"]),
        )
        self.assertEqual(
            {balance for seed, balance, _ in configurations if seed == pipeline.SEED},
            set(pipeline.ROBUSTNESS_POLICY["balance_policies"]),
        )
        wide = pd.DataFrame({
            "Patient_ID": ["A:p0", "B:p1"],
            "SourceSet": ["A", "B"],
            "ICULOS": [1, 1],
            "Fold": [0, 1],
            koopman.TARGET_COLUMN: [0, 1],
        })
        for representation in ("C0", "C3"):
            for seed, balance, _ in configurations:
                wide[pipeline.robustness_probability_column(
                    representation, seed, balance
                )] = [0.1, 0.9] if representation == "C3" else [0.5, 0.5]
        summary, differences = pipeline.robustness_summary(wide)
        self.assertEqual((len(summary), len(differences)), (10, 5))
        self.assertFalse(summary["configuration_selected"].any())
        self.assertFalse(differences["configuration_selected"].any())
        self.assertTrue(
            (differences["C3_minus_C0_patient_balanced_average_precision"] >= 0).all()
        )
        identity = pipeline.ROBUSTNESS_IDENTITY_COLUMNS
        c0_columns = identity + [column for column in wide if "prob_raw__C0__" in column]
        c3_columns = identity + [column for column in wide if "prob_raw__C3__" in column]
        combined = pipeline.combine_robustness_oof({
            "C0": wide[c0_columns], "C3": wide[c3_columns],
        })
        self.assertEqual(len(combined), 2)
        self.assertEqual(list(combined.columns), list(wide.columns))
        mismatched = wide[c3_columns].copy()
        mismatched.loc[0, "ICULOS"] = 2
        with self.assertRaises(pipeline.PipelineError):
            pipeline.combine_robustness_oof({
                "C0": wide[c0_columns], "C3": mismatched,
            })

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
        observed_difference = next(
            row for row in rows if row["metric"] == "patient_balanced_average_precision"
        )["candidate_minus_comparator"]
        comparator_decisions = koopman.primary_decisions(comparator)
        expected_difference = expected - pipeline.average_precision_score(
            comparator_decisions[koopman.TARGET_COLUMN],
            comparator_decisions["prob_calibrated"],
            sample_weight=koopman.equal_patient_weights(comparator_decisions),
        )
        self.assertAlmostEqual(observed_difference, expected_difference)
        self.assertGreater(observed_difference, 0)

    def test_alarm_window_budget_and_dca(self):
        frame = primary_oof()
        alarm = koopman.alarm_metrics(frame, "prob_calibrated", 0.5)
        self.assertEqual(alarm["tp_patients"], 2)
        self.assertEqual(alarm["fn_patients"], 0)
        self.assertEqual(alarm["false_alarm_episodes"], 0)
        self.assertEqual(alarm["useful_sensitivity"], 1.0)
        events = pd.DataFrame(koopman.alarm_event_rows(frame, "prob_calibrated", 0.5))
        self.assertEqual(len(events), 4)
        self.assertEqual(events["tp_patient"].sum(), 2)
        self.assertTrue(events.loc[events["tp_patient"] == 1, "first_eligible_alert_iculos"].notna().all())
        self.assertEqual(events["post_onset_alarm_episodes"].sum(), 0)
        post = frame.copy()
        post["prob_calibrated"] = 0.0
        post.loc[
            (post["Patient_ID"].str.startswith("A:s"))
            & (post["ICULOS"] >= post["TrueSepsisOnset_ICULOS"]),
            "prob_calibrated",
        ] = 1.0
        post_alarm = koopman.alarm_metrics(post, "prob_calibrated", 0.5)
        self.assertEqual(post_alarm["post_onset_alarm_episodes"], 2)
        self.assertEqual((post_alarm["tp_patients"], post_alarm["fn_patients"]), (0, 2))
        reliability = pd.DataFrame(koopman.primary_reliability_rows(
            frame, "prob_calibrated", "C3", bins=10
        ))
        self.assertEqual(len(reliability), 10)
        self.assertEqual(reliability["n_decision_hours"].sum(), len(koopman.primary_decisions(frame)))
        self.assertAlmostEqual(reliability["patient_weight_mass"].sum(), 4.0)
        decisions = koopman.primary_decisions(frame)
        calibration = pipeline.calibration_metrics(
            decisions[koopman.TARGET_COLUMN].to_numpy(dtype=int),
            decisions["prob_calibrated"].to_numpy(dtype=float),
            sample_weight=koopman.equal_patient_weights(decisions),
        )
        reliability_ece = sum(
            row.patient_weight_mass / reliability["patient_weight_mass"].sum()
            * abs(row.observed_frequency - row.mean_prediction)
            for row in reliability.dropna(subset=["mean_prediction"]).itertuples()
        )
        self.assertAlmostEqual(reliability_ece, calibration["ece_fixed_10_bins"])
        persistent = frame.copy()
        persistent["prob_calibrated"] = 1.0
        persistent_alarm = koopman.alarm_metrics(persistent, "prob_calibrated", 0.5)
        self.assertEqual((persistent_alarm["tp_patients"], persistent_alarm["fn_patients"]), (0, 2))
        self.assertEqual(persistent_alarm["false_alarm_episodes"], 4)
        fast_alarm = koopman._alarm_metrics_for_threshold(
            persistent, "prob_calibrated", 0.5
        )
        for name, value in persistent_alarm.items():
            if isinstance(value, float):
                self.assertTrue(np.isclose(value, fast_alarm[name], equal_nan=True))
            else:
                self.assertEqual(value, fast_alarm[name])
        left = koopman.add_primary_target(patient(
            "A:left", [1, 2], [1, 1], math.nan,
            "septic_onset_left_censored", [0.0, 0.0],
        ))
        left["prob_calibrated"] = 1.0
        left_alarm = koopman.alarm_metrics(
            pd.concat([frame, left], ignore_index=True), "prob_calibrated", 0.5
        )
        self.assertEqual(left_alarm["left_censored_unclassified_alarm_episodes"], 1)
        self.assertEqual(left_alarm["n_patients_with_predictions"], 5)
        self.assertEqual(left_alarm["n_monitored_patients"], 4)
        dca = koopman.decision_curve(frame, "prob_calibrated", (0.25,), repeats=20, seed=9)
        self.assertEqual(dca[0]["uncertainty_unit"], "patient-cluster bootstrap")
        self.assertEqual(dca[0]["outcome_estimand"], "true onset in 1--6 hours")
        comparator = frame.copy()
        comparator["prob_calibrated"] = 0.5
        paired_dca = koopman.paired_decision_curve_difference(
            comparator, frame, (0.25,), repeats=20, seed=9
        )
        self.assertGreater(paired_dca[0]["C3_minus_C0_net_benefit"], 0)
        self.assertEqual(
            paired_dca[0]["uncertainty_unit"], "paired patient-cluster bootstrap"
        )

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
            names = [
                "runtime_manifest.json", "harmonized.csv", "features.csv", "folds.csv",
                "model_runtime_manifest.json",
            ]
            for representation in koopman.REPRESENTATIONS:
                names.extend([
                    f"{representation}_nested_selection.csv",
                    f"{representation}_inner_selection.csv",
                    f"{representation}_oof_predictions.csv",
                    f"{representation}_metrics.json",
                    f"{representation}_dca.csv",
                    f"{representation}_reliability.csv",
                    f"{representation}_alarm_events.csv",
                ])
            names.extend([
                "transport.csv", "transport_inner_selection.csv", "inference.csv",
                "dca_inference.csv", "metrics.json",
                "robustness_oof.csv", "robustness_summary.csv",
                "robustness_differences.csv", "ablation_summary.csv",
                "master_results.csv",
                "scientific_gate_status.json", "probast_ai_status.json",
            ])
            raw.write_bytes(b"raw")
            for name in names:
                (run_dir / name).write_text(name, encoding="utf-8")
            runtime = {"git_commit": "a" * 40, "data_archive_path": str(raw)}
            lineage = pipeline.direct_onset_lineage(run_dir, runtime)
            self.assertEqual(pipeline.validate_lineage_nodes(run_dir, lineage), 46)
            (run_dir / "C3_oof_predictions.csv").write_text("tampered", encoding="utf-8")
            with self.assertRaises(pipeline.PipelineError):
                pipeline.validate_lineage_nodes(run_dir, lineage)


class ResourceOrchestrationTests(unittest.TestCase):
    def test_profile_subset_balances_to_the_rarest_eligible_stratum(self):
        rows = []
        for source, outcome, count in (("A", 0, 4), ("A", 1, 3), ("B", 0, 5), ("B", 1, 2)):
            for index in range(count):
                rows.append({
                    "Patient_ID": f"{source}:{outcome}:{index}", "SourceSet": source,
                    koopman.TARGET_COLUMN: outcome, koopman.ELIGIBLE_COLUMN: 1,
                })
        frame = pd.DataFrame(rows)
        selected = profile_resources.fixed_patients(
            frame, maximum_per_stratum=3, minimum_per_stratum=1
        )
        strata = frame.loc[frame["Patient_ID"].isin(selected)].groupby(
            ["SourceSet", koopman.TARGET_COLUMN]
        ).size()
        self.assertEqual(strata.to_dict(), {("A", 0): 2, ("A", 1): 2, ("B", 0): 2, ("B", 1): 2})
        with self.assertRaisesRegex(pipeline.PipelineError, "fewer than 3"):
            profile_resources.fixed_patients(
                frame, maximum_per_stratum=3, minimum_per_stratum=3
            )

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

    def test_gpu_profile_uses_active_one_second_samples(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gpu.csv"
            path.write_text(
                "0, 10, 40960\n60, 1000, 40960\n80, 1200, 40960\n",
                encoding="utf-8",
            )
            measured = resource_provenance.parse_gpu(path)
            self.assertEqual((measured["samples"], measured["active_samples"]), (3, 2))
            self.assertEqual(measured["mean_active_utilization_percent"], 70.0)
            self.assertEqual(measured["max_memory_used_mib"], 1200.0)

    def test_profile_workload_represents_full_model_mix(self):
        self.assertEqual(
            profile_resources.estimated_full_peak_gb(2.0, 4.0, 0.5, 0.25),
            24.75,
        )
        outer = pipeline.MODEL_POLICY["outer_folds"]
        inner = pipeline.MODEL_POLICY["inner_folds"]
        candidates = len(pipeline.MODEL_CANDIDATES)
        lifts = len(koopman.KOOPMAN_POLICY["lifts"])
        primary_xgb = outer * sum(
            inner * candidates * (lifts if representation == "C3" else 1) + 1
            for representation in koopman.REPRESENTATIONS
        )
        transport_xgb = 2 * sum(
            inner * candidates * (lifts if representation == "C3" else 1) + 1
            for representation in koopman.REPRESENTATIONS
        )
        robustness_xgb = outer * len(pipeline.ROBUSTNESS_POLICY["representations"]) * (
            len(pipeline.robustness_configurations()) - 1
        )
        self.assertEqual(
            pipeline.MODEL_POLICY["planned_full_xgboost_fits"],
            primary_xgb + transport_xgb + robustness_xgb,
        )
        planned_koopman = outer * (inner * lifts + 1) + 2 * (inner * lifts + 1)
        self.assertEqual(
            pipeline.MODEL_POLICY["planned_full_koopman_fits"], planned_koopman
        )
        benchmark_ratio = (
            profile_resources.BENCHMARK_XGBOOST_FITS
            / len(koopman.KOOPMAN_POLICY["lifts"])
        )
        planned_ratio = (
            pipeline.MODEL_POLICY["planned_full_xgboost_fits"]
            / pipeline.MODEL_POLICY["planned_full_koopman_fits"]
        )
        self.assertLess(abs(benchmark_ratio - planned_ratio) / planned_ratio, 0.15)

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
                    "partition": "gpu" if gpus else "cpu",
                    "slurm_job_id": name,
                    "run_id": "run", "source_git_commit": "a" * 40,
                    "source_inventory_sha256": "b" * 64,
                    "requested": {"cpus": cpus, "memory_gb": 32, "gpus": gpus},
                    "measured": {
                        "elapsed_seconds": elapsed, "cpu_efficiency": efficiency,
                        "max_rss_gb": 8.0,
                        "gpu": {
                            "active_samples": 5 if gpus else 0,
                            "mean_active_utilization_percent": 80.0 if gpus else None,
                        },
                    },
                    "benchmark": {
                        "status": "PASS", "estimated_full_peak_gb": 25.5,
                        "active_cpu_efficiency": efficiency,
                    },
                }
                (root / f"{name}.json").write_text(json.dumps(payload), encoding="utf-8")
            output = root / "selection.json"
            selected = resource_provenance.select_profile(Namespace(
                profile_dir=root, output=output, run_id="run",
                git_commit="a" * 40, source_inventory="b" * 64,
            ))
            self.assertEqual(selected["selected"], {"cpus": 8, "memory_gb": 32, "gpus": 1})
            self.assertEqual(selected["fit_profile"], {"cpus": 8, "memory_gb": 32, "gpus": 1})
            self.assertEqual(selected["parallel_workers"], 1)
            self.assertEqual(selected["parallel_candidates"], 1)
            self.assertEqual(selected["selected_profile_active_gpu_samples"], 5)
            self.assertLessEqual(selected["selected"]["cpus"], 64)
            self.assertLessEqual(selected["selected"]["memory_gb"], 64)

    def test_cpu_profile_parallelism_is_bounded_by_measured_memory(self):
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
                    "partition": "gpu" if gpus else "cpu", "slurm_job_id": name,
                    "run_id": "run", "source_git_commit": "a" * 40,
                    "source_inventory_sha256": "b" * 64,
                    "requested": {"cpus": cpus, "memory_gb": 32, "gpus": gpus},
                    "measured": {
                        "elapsed_seconds": elapsed, "cpu_efficiency": efficiency,
                        "max_rss_gb": 8.0,
                        "gpu": {
                            "active_samples": 2 if gpus else 0,
                            "mean_active_utilization_percent": 40.0 if gpus else None,
                        },
                    },
                    "benchmark": {
                        "status": "PASS", "estimated_full_peak_gb": 25.5,
                        "active_cpu_efficiency": efficiency,
                    },
                }
                (root / f"{name}.json").write_text(json.dumps(payload), encoding="utf-8")
            selected = resource_provenance.select_profile(Namespace(
                profile_dir=root, output=root / "selection.json", run_id="run",
                git_commit="a" * 40, source_inventory="b" * 64,
            ))
            self.assertEqual(selected["fit_profile"], {"cpus": 16, "memory_gb": 32, "gpus": 0})
            self.assertEqual(selected["parallel_workers"], 2)
            self.assertEqual(selected["parallel_candidates"], 2)
            self.assertEqual(selected["selected"], {"cpus": 64, "memory_gb": 64, "gpus": 0})
            self.assertEqual(selected["finalize_memory_gb"], 32)

    def test_selected_parallel_profile_requires_measured_majority_cpu_use(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profiles = root / "profiles"
            profiles.mkdir()
            selection = {
                "status": "PASS", "run_id": "run",
                "source_git_commit": "a" * 40,
                "source_inventory_sha256": "b" * 64,
                "selected": {"cpus": 64, "memory_gb": 64, "gpus": 0},
                "fit_profile": {"cpus": 16, "memory_gb": 32, "gpus": 0},
                "parallel_workers": 2, "parallel_candidates": 2,
            }
            selection_path = root / "selection.json"
            selection_path.write_text(json.dumps(selection), encoding="utf-8")
            benchmark = {
                "status": "PASS", "fit_threads": 16,
                "parallel_workers": 2, "parallel_candidates": 2,
                "concurrent_xgboost_fits": 4,
                "xgboost_active_cpu_efficiency": 0.55,
            }
            (profiles / "selected-model-benchmark.json").write_text(
                json.dumps(benchmark), encoding="utf-8"
            )
            profile = {
                "stage": "selected-model", "status": "PASS",
                "hostname": "compute-0-2", "partition": "cpu",
                "run_id": "run", "source_git_commit": "a" * 40,
                "source_inventory_sha256": "b" * 64,
                "requested": selection["selected"],
                "measured": {"max_rss_gb": 40.0, "gpu": {}},
                "benchmark": benchmark,
            }
            profile_path = profiles / "selected-model.json"
            profile_path.write_text(json.dumps(profile), encoding="utf-8")
            args = Namespace(profile_dir=profiles, selection=selection_path)
            self.assertEqual(
                resource_provenance.verify_selected_profile(args)["status"], "PASS"
            )
            profile["benchmark"]["xgboost_active_cpu_efficiency"] = 0.50
            (profiles / "selected-model-benchmark.json").write_text(
                json.dumps(profile["benchmark"]), encoding="utf-8"
            )
            profile_path.write_text(json.dumps(profile), encoding="utf-8")
            with self.assertRaises(resource_provenance.ResourceError):
                resource_provenance.verify_selected_profile(args)

    def test_resource_aggregate_binds_selected_profile_and_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            resources = root / "resources"
            resources.mkdir()
            selected = {"cpus": 8, "memory_gb": 12, "gpus": 1}
            requests = {
                "prepare": {"cpus": 1, "memory_gb": 10, "gpus": 0},
                "model": selected,
                "finalize": {"cpus": 1, "memory_gb": 12, "gpus": 0},
            }
            for stage, requested in requests.items():
                (resources / f"{stage}.json").write_text(json.dumps({
                    "stage": stage, "status": "PASS", "hostname": "compute-0-2",
                    "partition": "gpu" if requested["gpus"] else "cpu",
                    "requested": requested, "run_id": "run", "source_git_commit": "a" * 40,
                    "source_inventory_sha256": "b" * 64,
                }), encoding="utf-8")
            selection = root / "selection.json"
            selection.write_text(json.dumps({
                "status": "PASS", "selected": selected, "run_id": "run",
                "finalize_memory_gb": 12,
                "source_git_commit": "a" * 40,
                "source_inventory_sha256": "b" * 64,
            }), encoding="utf-8")
            profiles = root / "profiles"
            profiles.mkdir()
            benchmark = {"status": "PASS"}
            (profiles / "selected-model-benchmark.json").write_text(
                json.dumps(benchmark), encoding="utf-8"
            )
            (profiles / "selected-model.json").write_text(json.dumps({
                "stage": "selected-model", "status": "PASS",
                "hostname": "compute-0-2", "partition": "gpu",
                "requested": selected, "run_id": "run",
                "source_git_commit": "a" * 40,
                "source_inventory_sha256": "b" * 64,
                "benchmark": benchmark,
            }), encoding="utf-8")
            args = Namespace(
                resources_dir=resources, profile_selection=selection,
                output=root / "resource_manifest.json",
            )
            self.assertEqual(resource_provenance.aggregate(args)["status"], "PASS")
            model = json.loads((resources / "model.json").read_text(encoding="utf-8"))
            model["requested"]["cpus"] = 16
            (resources / "model.json").write_text(json.dumps(model), encoding="utf-8")
            with self.assertRaises(resource_provenance.ResourceError):
                resource_provenance.aggregate(args)

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
        self.assertNotIn("json.load(open(", entrypoint)
        self.assertIn('with open(sys.argv[1], encoding="utf-8") as handle:', entrypoint)
        self.assertIn("HOST_PYTHON=${HOST_PYTHON:-python3}", entrypoint)
        self.assertIn("--nodelist=compute-0-2", job)
        self.assertNotIn("compute-0-1", job)
        self.assertIn("set -euo pipefail", entrypoint)
        self.assertIn("set -euo pipefail", job)
        self.assertIn("host_python=unavailable", job)
        self.assertIn('"$RUNTIME" exec', job)
        self.assertIn('export SEPSIS_FIT_THREADS="$MODEL_FIT_THREADS"', entrypoint)
        self.assertIn('export SEPSIS_PARALLEL_WORKERS="$MODEL_WORKERS"', entrypoint)
        self.assertIn('export SEPSIS_PARALLEL_CANDIDATES="$MODEL_CANDIDATES"', entrypoint)
        self.assertIn("verify-selected-profile", entrypoint)
        self.assertIn("selected-model", job)
        self.assertIn('FIT_THREADS * PARALLEL_WORKERS * PARALLEL_CANDIDATES <= REQUESTED_CPUS', job)
        self.assertEqual(resource_provenance.CPU_CAP, 64)
        self.assertEqual(resource_provenance.MEMORY_CAP_GB, 64)
        self.assertEqual(resource_provenance.GPU_CAP, 1)
        self.assertEqual(resource_provenance.MIN_ACTIVE_CPU_EFFICIENCY, 0.50)
        self.assertEqual(resource_provenance.MIN_ACTIVE_GPU_UTILIZATION_PERCENT, 50.0)


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
            "No es el manuscrito", "Ablaciones, semillas y balance",
            "DEFERRED_TO_MANUSCRIPT_PHASE", "4,000 pacientes", ">50 %",
        ):
            self.assertIn(required, text)
        self.assertNotIn("EXPECTED_AUROC", text)
        self.assertGreaterEqual(text.count("doi.org/"), 8)


if __name__ == "__main__":
    unittest.main()
