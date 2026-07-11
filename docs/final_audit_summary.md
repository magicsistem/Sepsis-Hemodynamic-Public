# Final Audit Summary

The public workflow is limited to internal patient-grouped cross-validation on the PhysioNet/CinC 2019 public training split. It does not include an official hidden-test evaluation, an independent external cohort, prospective validation, or clinical deployment validation.

## Supported Public Claims

- The baseline and enhanced workflows use patient-grouped folds.
- The harmonization step prefers PSV files and skips aggregate CSV/TSV files when PSV files are present.
- `SourceSet` is retained for internal traceability when training set paths are present in the ZIP.
- The enhanced model adds hemodynamic complexity features to the baseline clinical feature set.
- Final compact tables and figures are generated from documented result artifacts.

## Guardrails

- Do not describe SourceSet analysis as external validation.
- Do not claim clinical readiness, deployment readiness, mechanistic proof, or direct superiority over external leaderboard results.
- Keep Utility Score language threshold-specific.
- Keep patient-level and time-step-level metrics separate.

## Local Traceability

Historical notes and manuscript-editing audits may remain locally, but public documentation is consolidated into this summary and the reproducibility docs.
