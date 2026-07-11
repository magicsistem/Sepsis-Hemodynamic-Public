# Internal Robustness Checklist

| check_id | status | details | evidence |
| --- | --- | --- | --- |
| baseline_dir_exists | PASS | baseline_dir exists. | results/baseline |
| enhanced_dir_exists | PASS | enhanced_dir exists. | results/enhanced |
| statistics_dir_exists | PASS | statistics_dir exists. | results/statistics |
| no_independent_cohort_read | PASS | Script uses final OOF predictions and harmonized traceability only. | CLI inputs |
| methodology_audit_pass_count | NOT_AVAILABLE | methodology audit CSV was not found. | external_artifacts/inventory/no_leakage_audit.csv |
| baseline_oof_columns | PASS | baseline OOF required columns present. | results/baseline/oof_predictions.csv |
| baseline_no_duplicate_patient_timestep | PASS | Duplicate Patient_ID+TimeStep rows: 0. | results/baseline/oof_predictions.csv |
| baseline_binary_labels | PASS | Observed labels: ['0', '1']. | results/baseline/oof_predictions.csv |
| baseline_raw_probability_range | PASS | prob_raw range: [6.28898e-05, 0.9990569]. | results/baseline/oof_predictions.csv |
| baseline_platt_probability_range | PASS | prob_platt range: [0.0013510255, 0.42423233]. | results/baseline/oof_predictions.csv |
| baseline_isotonic_probability_range | PASS | prob_isotonic range: [0.0, 1.0]. | results/baseline/oof_predictions.csv |
| enhanced_oof_columns | PASS | enhanced OOF required columns present. | results/enhanced/oof_predictions.csv |
| enhanced_no_duplicate_patient_timestep | PASS | Duplicate Patient_ID+TimeStep rows: 0. | results/enhanced/oof_predictions.csv |
| enhanced_binary_labels | PASS | Observed labels: ['0', '1']. | results/enhanced/oof_predictions.csv |
| enhanced_raw_probability_range | PASS | prob_raw range: [3.0886079e-06, 0.99950075]. | results/enhanced/oof_predictions.csv |
| enhanced_platt_probability_range | PASS | prob_platt range: [0.0016877659, 0.44148934]. | results/enhanced/oof_predictions.csv |
| enhanced_isotonic_probability_range | PASS | prob_isotonic range: [0.0, 1.0]. | results/enhanced/oof_predictions.csv |
| baseline_enhanced_oof_alignment | PASS | OOF rows are aligned by Patient_ID, TimeStep, SepsisLabel, and fold. |  |
| baseline_final_metric_match | PASS | n_features: observed=287, expected=287, diff=0; AUROC_raw: observed=0.93149745, expected=0.93149745, diff=0; AUPRC_raw: observed=0.306045364, expected=0.306045364, diff=0; Utility_raw_at_0.5: observed=0.501492271, expected=0.501492271, diff=0; Utility_raw_best: observed=0.505664826, expected=0.505664826, diff=0; Utility_raw_best_threshold: observed=0.449999988, expected=0.449999988, diff=0 | results/baseline/summary_metrics.json |
| enhanced_final_metric_match | PASS | n_features: observed=305, expected=305, diff=0; AUROC_raw: observed=0.934294715, expected=0.934294715, diff=0; AUPRC_raw: observed=0.326412513, expected=0.326412513, diff=0; Utility_raw_at_0.5: observed=0.495466658, expected=0.495466658, diff=0; Utility_raw_best: observed=0.50101102, expected=0.50101102, diff=0; Utility_raw_best_threshold: observed=0.409999996, expected=0.409999996, diff=0 | results/enhanced/summary_metrics.json |
| baseline_patient_level_and_lead_time_metrics_match | PASS | Patient_sensitivity_best_method_at_0.5: observed=0.0419508868, expected=0.0419508868, diff=0, column=Patient_sensitivity_best_method_at_0.5; Patient_specificity_best_method_at_0.5: observed=0.993476634, expected=0.993476634, diff=0, column=Patient_specificity_best_method_at_0.5; Lead_time_median_best_method_at_0.5: observed=24, expected=24, diff=0, column=Lead_time_median_best_method_at_0.5 | results/baseline/summary_metrics.json |
| enhanced_patient_level_and_lead_time_metrics_match | PASS | Patient_sensitivity_best_method_at_0.5: observed=0.0518417462, expected=0.0518417462, diff=0, column=Patient_sensitivity_best_method_at_0.5; Patient_specificity_best_method_at_0.5: observed=0.991230884, expected=0.991230884, diff=0, column=Patient_specificity_best_method_at_0.5; Lead_time_median_best_method_at_0.5: observed=23.5, expected=23.5, diff=0, column=Lead_time_median_best_method_at_0.5 | results/enhanced/summary_metrics.json |
| sourceset_available | PASS | SourceSet present in OOF predictions. | OOF |
