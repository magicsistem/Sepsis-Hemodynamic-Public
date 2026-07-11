# Statistics Summary

## Input Paths

- Baseline results: `results/baseline`
- Enhanced results: `results/enhanced`
- Output tables: `results/statistics`

## Primary Metrics

| metric | baseline | enhanced | delta |
|---|---:|---:|---:|
| AUROC raw | 0.931497 | 0.934295 | 0.002797 |
| AUPRC raw | 0.306045 | 0.326413 | 0.020367 |
| Utility raw best | 0.505665 | 0.501011 | -0.004654 |
| n features | 287 | 305 | 18 |

These values are read from the final run `summary_metrics.json` / `metrics.json` files. The enhanced model improves raw AUROC and raw AUPRC. Enhanced raw Utility best is lower than baseline; report Utility as threshold-dependent and do not claim uniform operational utility improvement.

## Statistical Tests

- Delta_AUROC_raw: delta=0.002797, 95% CI [0.001679, 0.003855], p<0.001, BH p<0.001 (paired_patient_bootstrap)
- Delta_AUPRC_raw: delta=0.020367, 95% CI [0.015957, 0.024697], p<0.001, BH p<0.001 (paired_patient_bootstrap)
- Delta_Brier_raw: delta=-0.007508, 95% CI [-0.007789, -0.007203], p<0.001, BH p<0.001 (paired_patient_bootstrap)
- Delta_Utility_raw_best: delta=-0.004654, 95% CI [NA, NA], NA, BH NA (descriptive_summary_delta)
- McNemar_time_step_threshold_0.5: delta=14537.000000, 95% CI [NA, NA], p<1e-300, BH p<1e-300 (mcnemar_time_step_correctness;base_only=17316;enhanced_only=31853)
- McNemar_patient_threshold_0.5: delta=1418.000000, 95% CI [NA, NA], p<0.001, BH p<0.001 (mcnemar_patient_correctness;base_only=918;enhanced_only=2336)
- McNemar_time_step_utility_optimal: delta=-1627.000000, 95% CI [NA, NA], p<0.001, BH p<0.001 (mcnemar_time_step_correctness;base_only=29507;enhanced_only=27880)
- McNemar_patient_utility_optimal: delta=-177.000000, 95% CI [NA, NA], p=0.002, BH p=0.002 (mcnemar_patient_correctness;base_only=1663;enhanced_only=1486)
- Delta_AUROC_raw_DeLong: delta=0.002797, 95% CI [NA, NA], p<0.001, BH p<0.001 (paired_delong;z=9.74773;variance=8.23494e-08)

Paired AUROC/AUPRC/Brier tests use patient-level bootstrap resampling over OOF predictions. DeLong status: **CONDITIONAL**. Utility bootstrap status: **PASS**.

## Statistics Completion Status

- ready_for_methodology_review: **CONDITIONAL**
- explanation: Only conditional statistical/reporting items remain: DeLong complete/conditional
- calibration status: **PASS** - Training code audit confirms fold-wise OOF cross-fitting via crossfit_calibration(fold_ids): train folds fit calibration, held-out fold receives calibrated probabilities.

## Remaining Unresolved Items

- DeLong complete/conditional: CONDITIONAL - Paired DeLong AUROC completed and validated against sklearn AUROC, but patient bootstrap remains primary because OOF rows contain repeated time steps per patient.
- ready_for_methodology_review: CONDITIONAL - Only conditional statistical/reporting items remain: DeLong complete/conditional

## Hemodynamic Feature Importance Checks

- Hemodynamic complexity features found in enhanced importance table: 18 of 18 expected features.
- Top-ranked hemodynamic features: HeartRate_sampen_24h, O2Sat_sampen_24h, MeanBP_sampen_24h, SysBP_sampen_24h, RespRate_sampen_24h, DiaBP_sampen_24h, O2Sat_cv_8h, SysBP_cv_8h, RespRate_iqr_8h, MeanBP_cv_8h.
- SampEn features present: DiaBP_sampen_24h, HeartRate_sampen_24h, MeanBP_sampen_24h, O2Sat_sampen_24h, RespRate_sampen_24h, SysBP_sampen_24h.

## Fold Delta Summary

- Delta AUROC mean/std: 0.002598 / 0.001432
- Delta AUPRC mean/std: 0.020864 / 0.005697

## Allowed Paper Claims

- Enhanced improves discrimination under patient-grouped internal cross-validation on the public PhysioNet/CinC 2019 training split.
- AUPRC gain is statistically supported by paired patient bootstrap if `statistical_tests.csv` confirms significance.
- Hemodynamic complexity features add complementary ranking information.
- Utility is threshold-dependent and should be reported with caution.
- Independent/prospective validation is required before clinical deployment.

## Do Not Write In Paper

- Do not claim clinical deployment.
- Do not claim near-perfect calibration.
- Do not claim external validation.
- Do not claim hidden-test performance.
- Do not claim SOTA superiority over Zabihi.
- Do not claim all calibration improved, because raw calibration worsened if ECE_raw or Brier_raw increased.

## Interpretation Warnings

- No hidden test set was used in this reporting package.
- SourceSet A/B supports traceability and internal source-wise checks; it is not external validation.
- Any comparison with Zabihi et al. must be like-for-like in cohort, preprocessing, horizon, metrics, and utility setup.
- Raw Utility Score is threshold-dependent and must be interpreted cautiously; it does not necessarily improve when discrimination improves.
- Threshold 0.5 can have low patient-level sensitivity; lower thresholds may increase false alarms.
- Patient-level and time-step-level metrics answer different questions and must not be mixed in paper text.
- Raw calibration worsened if ECE_raw or Brier_raw increased; do not claim all calibration improved.
- subgroup_metrics.csv contains n_rows; verify paper text does not double-count time steps as patients.

## Generated Tables

- `comparison_summary.csv`: Primary baseline vs enhanced metric deltas.
- `statistical_tests.csv`: Patient-level paired bootstrap and McNemar tests.
- `threshold_operating_points.csv`: Selected threshold operating points.
- `patient_level_metrics.csv`: Patient-level sensitivity, specificity, and lead-time summaries.
- `calibration_table.csv`: Raw, Platt, and isotonic calibration metrics.
- `subgroup_table.csv`: Baseline vs enhanced subgroup comparisons.
- `feature_importance_table.csv`: Feature importance with hemodynamic feature annotations.
- `fold_comparison.csv`: Fold-wise AUROC/AUPRC comparison.
- `statistics_summary.md`: Human-readable statistics report.
- `completion_checklist.md`: Completion checklist for statistics readiness.
- `completion_checklist.csv`: Machine-readable completion checklist.
- `utility_statistical_tests.csv`: Patient-bootstrap Utility Score inference.
- `patient_confusion_matrices.csv`: Patient-level confusion matrices at selected thresholds.
- `subgroup_patient_table.csv`: Patient-level subgroup operating metrics from OOF predictions.
- `feature_count_audit.csv`: Feature count and paper wording audit.
- `calibration_audit.md`: Calibration interpretation audit.
- `statistical_methods_notes.md`: Statistical methods notes for the statistics package.
- `code_math_audit_summary.md`: Static code/math audit summary for statistics.
- `validation_checks.csv`: Input artifact and OOF alignment validation checks.
- `validation_checks.md`: Human-readable input validation report.
- `results_snippet.md`: Short cautious results snippet for later local drafting.
- `manifest.json`: Reproducibility manifest with command, inputs, outputs, and checksums.

## Pending Paper Checks

- Correct feature count statements.
- Correct whether SampEn features are described as five or six signals.
- Correct subgroup text and avoid row/time-step double counting.
- Explain calibration and distinguish raw, Platt, and isotonic outputs.
- Explain patient-level vs time-step-level metrics.
- Explain PhysioNet Utility Score and threshold dependence.
- Update reproducibility and ethics statements.
