# Sepsis Hemodynamic: corrected public experiment

This repository runs one corrected, fail-closed experiment on the public
PhysioNet/Computing in Cardiology Challenge 2019 training data. It does not
attempt to reproduce any historical result, table, threshold, feature count,
or p-value.

## Run on CEDIA

From the clean, committed project checkout on CEDIA:

```bash
bash run.sh
```

`run.sh` is the only top-level entrypoint. It submits one Slurm job, waits for
it, and returns nonzero unless tests, every pipeline stage, and final manifest
validation pass. The job runs the test suite and then creates one immutable
`runs/<run-id>/` directory. Existing run directories are never reused or
overwritten.

## Scientific scope

- Input is `data/raw/archive.zip`, verified as an inventory of official
  patient-level PSV files. CSV/TSV fallback is prohibited.
- Schema is the 40 official predictors plus `SepsisLabel`; `Hct` is required.
- `SepsisLabel` remains explicitly the Challenge's shifted persistent label.
  Reconstructed onset is `first positive ICULOS + 6 h` and is used only for
  pre-specified early-warning analyses.
- Model selection, Platt calibration, and operating threshold selection occur
  only inside each outer training partition.
- Challenge Utility calls the pinned official PhysioNet scorer. Average
  Precision and trapezoidal PR-AUC are named separately.
- SourceSet A/B transport is internal public-data transport, not external
  validation. MIMIC/eICU external validation remains unavailable without the
  required credentialed cohorts.

Historical files under `results/` and `external_artifacts/` are retained as
evidence only. The corrected pipeline never reads them.
