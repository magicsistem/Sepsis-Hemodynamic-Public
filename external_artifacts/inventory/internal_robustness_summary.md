# Internal Robustness Summary

This report uses existing final out-of-fold predictions and statistics reporting artifacts only.
It is an internal robustness analysis under patient-grouped cross-validation, not an independent cohort study.

## Validation Checks

- PASS: 22
- FAIL: 0
- CONDITIONAL: 0
- NOT_AVAILABLE: 1

## Analyses Generated

- Source-wise rows: 6
- Subgroup rows: 26
- Threshold robustness rows: 114
- Calibration robustness rows: 6
- Ablation/window inventory rows: 1

## Interpretation Guardrails

- Patient-level and lead-time best-method summary values are checked from final JSON summaries or comparison_summary.csv.
- Operating-point patient tables remain descriptive and are not used for the best-method summary check.
- Utility is threshold-dependent.
- Enhanced did not uniformly improve Utility in the final internal run.
- Baseline had slightly higher best raw Utility in the final internal run.
- Threshold 0.5 is not an optimized clinical operating point.
- No official hidden-test evaluation, MIMIC-IV, eICU, or prospective validation was performed.
- Claims should remain restricted to patient-grouped internal cross-validation.
