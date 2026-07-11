# Calibration Audit

## Raw Calibration

- Baseline ECE_raw: 0.15007940
- Enhanced ECE_raw: 0.13012155
- Delta ECE_raw enhanced-baseline: -0.01995785
- Baseline Brier_raw: 0.07010657
- Enhanced Brier_raw: 0.06259856
- Delta Brier_raw enhanced-baseline: -0.00750801

Raw calibration worsened if these deltas are positive. Do not claim that all calibration improved.

## Calibrated Metrics

- Platt and isotonic AUROC/AUPRC/Brier/ECE are summarized in `calibration_table.csv`.
- If calibrated metrics improve, describe them as calibrated-output improvements, not raw-probability calibration improvements.
- Isotonic ECE may be extremely low and must be described cautiously.

## Independence of Calibration Fit

Status: **CODE AUDIT CONFIRMED INTERNAL CROSS-FIT**.

`src/training/train_zabihi_cudf.py` uses `crossfit_calibration(y_true, y_prob, fold_ids, method=...)`, fitting calibration on `fold_ids != fold` and predicting calibrated probabilities on `fold_ids == fold`. This supports internal OOF calibrated metrics, not external/prospective calibration. Do not write "near-perfect calibration"; describe the result as cross-fitted internal calibration and keep raw calibration caveats visible.
