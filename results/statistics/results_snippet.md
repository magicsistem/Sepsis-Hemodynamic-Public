> **WITHDRAWN — HISTORICAL INVALID OUTPUT.** Preserved only as audit evidence; do not use for scientific claims, metrics, thresholds, validation, or reporting. Current outputs exist only in a hash-validated `runs/<run-id>/result_manifest.json`.

# Final Results Snippet

In patient-grouped out-of-fold internal cross-validation on the public PhysioNet/CinC 2019 training split, the enhanced hemodynamic-complexity model improved discrimination relative to the baseline. Raw AUROC increased from 0.931497 to 0.934295, and raw AUPRC increased from 0.306045 to 0.326413. Paired patient-level bootstrap supported the improvement in discrimination after BH correction.

Raw Utility Score was threshold-dependent and did not improve under the enhanced model (0.505665 baseline vs 0.501011 enhanced); operational benefit should be interpreted cautiously.

Patient-level sensitivity at the conventional 0.5 threshold remained low in both models (0.041951 baseline and 0.051842 enhanced), emphasizing the need for threshold selection and independent/prospective validation before any clinical deployment.

No independent external validation or official hidden-test evaluation was performed; all claims are limited to internal cross-validation on the public PhysioNet/CinC 2019 training split.
