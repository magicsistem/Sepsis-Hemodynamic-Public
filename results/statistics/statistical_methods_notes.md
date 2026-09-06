> **WITHDRAWN — HISTORICAL INVALID OUTPUT.** Preserved only as audit evidence; do not use for scientific claims, metrics, thresholds, validation, or reporting. Current outputs exist only in a hash-validated `runs/<run-id>/result_manifest.json`.

# Statistical Methods Notes

## Scope

All analyses use out-of-fold predictions from patient-grouped internal cross-validation on the public PhysioNet/CinC 2019 training split. No independent external validation or official hidden-test evaluation was performed; therefore, all claims are limited to internal cross-validation on the public PhysioNet/CinC 2019 training split.

## Out-Of-Fold Predictions And Grouped CV

Baseline and enhanced models are compared on paired OOF predictions aligned by `Patient_ID`, `TimeStep`, and `SepsisLabel`. Cross-validation is grouped by patient to avoid splitting a patient across train and validation folds.

## Paired Patient-Level Bootstrap

Bootstrap inference resamples `Patient_ID` values with replacement and keeps all time steps for each sampled patient. This preserves within-patient temporal dependence and avoids the anti-conservative behavior of row-level bootstrap over repeated time steps.

## McNemar Tests

McNemar tests compare paired correctness indicators from aligned baseline and enhanced OOF predictions. The script reports both `mcnemar_time_step_correctness` and `mcnemar_patient_correctness`; patient-level rows should be preferred for patient-level paper claims. Numeric `p_value_two_sided` and display-safe `p_value_display` are both emitted so paper text does not show `p=0.000000`.

## Benjamini-Hochberg Correction

Primary statistical rows are adjusted with Benjamini-Hochberg FDR correction and include `p_value_bh` plus `significant_bh_0.05`.

## Calibration Analysis

Raw, Platt, and isotonic probabilities are summarized separately. Platt and isotonic metrics are internal cross-fitted calibration analyses derived from OOF folds; they must not be described as independent validation.

## Threshold And Lead-Time Analysis

Threshold operating points include the conventional raw 0.5 threshold, raw max-Utility threshold, raw max-F1 threshold, and calibrated selected thresholds when available. Lead-time summaries are patient-level descriptive summaries at selected thresholds.

## DeLong Status

- Status: CONDITIONAL
- Explanation: Paired DeLong AUROC completed and validated against sklearn AUROC, but patient bootstrap remains primary because OOF rows contain repeated time steps per patient.

## Utility Bootstrap Status

- Status: PASS
- Explanation: Utility bootstrap completed after reproducing summary metrics.

## SourceSet A/B

`SourceSet` supports traceability and possible internal source-wise analyses. It is not external validation.
