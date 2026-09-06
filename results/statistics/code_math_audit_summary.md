> **WITHDRAWN — HISTORICAL INVALID OUTPUT.** Preserved only as audit evidence; do not use for scientific claims, metrics, thresholds, validation, or reporting. Current outputs exist only in a hash-validated `runs/<run-id>/result_manifest.json`.

# Code Math Audit Summary

This generated summary accompanies the statistics tables. It is a reporting artifact only; it does not modify training, data harmonization, feature engineering, or experiment outputs.

## Verified Definitions From Code Inspection

- Data harmonization loads PSV files preferentially and skips aggregate CSV/TSV files when PSV files exist, avoiding `Dataset.csv` duplication.
- `Patient_ID` is preserved as string, `TimeStep` is sorted/preserved or causally reconstructed, `SourceSet` is retained, and `SepsisLabel` is cast to int8.
- Feature engineering uses patient-grouped causal rolling/expanding transforms.
- Hemodynamic complexity features are expected as 18 columns: six signals times CV 8h, IQR 8h, and SampEn 24h.
- Training uses patient-level fold assignments via `StratifiedGroupKFold` and excludes `Patient_ID`, `TimeStep`, `SourceSet`, and `SepsisLabel` from model features.
- Calibration in training is cross-fitted by fold IDs in `crossfit_calibration`; reporting still flags paper wording caution until the author confirms this is the intended procedure to describe.
- Reporting distinguishes time-step discrimination metrics, patient-level confusion matrices, patient-level subgroup tables, and Utility Score inference.

## Statistical Status

- DeLong status: **CONDITIONAL** - Paired DeLong AUROC completed and validated against sklearn AUROC, but patient bootstrap remains primary because OOF rows contain repeated time steps per patient.
- Utility bootstrap status: **PASS** - Utility bootstrap completed after reproducing summary metrics.
- Calibration audit status: **PASS** - Training code audit confirms fold-wise OOF cross-fitting via crossfit_calibration(fold_ids): train folds fit calibration, held-out fold receives calibrated probabilities.

## Known Paper Risks To Fix Before Local Manuscript Editing

- Old feature counts such as 233/251, 71/77, and duplicated-row counts must be replaced by final PSV-only values.
- Hidden-test, external-validation, and direct SOTA-superiority claims must not be made from internal CV results.
- Raw and calibrated calibration metrics must not be collapsed into a blanket claim that calibration improved.
- Patient-level metrics and time-step metrics must be labeled separately.

## Recommendation

Statistics readiness is determined by `completion_checklist.md`. If only calibration wording remains conditional, the next review step can proceed with explicit author confirmation; if Utility or DeLong validation fails on real artifacts, keep the claim tied to paired patient bootstrap instead.
