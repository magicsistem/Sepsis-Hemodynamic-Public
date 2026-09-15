# Sepsis Hemodynamic: corrected public experiment

This repository runs one corrected, fail-closed experiment on the public
PhysioNet/Computing in Cardiology Challenge 2019 training data. It does not
attempt to reproduce any historical result, table, threshold, feature count,
or p-value.

## Run on CEDIA

From the source tree synchronized from a clean laptop commit and validated by
its exact source sidecar on CEDIA:

```bash
bash run.sh
```

`run.sh` is the only top-level entrypoint. It submits one Slurm job, waits for
it, and returns nonzero unless tests, every pipeline stage, and final manifest
validation pass. The job runs the test suite and then creates one immutable
`runs/<run-id>/` directory. A failed directory is preserved; an explicit
`RESUME_RUN_ID` can reuse only its hash- and runtime-compatible upstream
checkpoints after downstream partials have been removed deliberately.

## Scientific scope

- Input is `data/raw/archive.zip`, pinned at SHA-256
  `1a0eb8040c76fdab84ee6c7dd6afdab4ad457a33d363cb7e4e200af713345897`
  and verified as 40,336 official patient-level PSV files, 40,336 patients and
  1,552,210 rows. CSV/TSV fallback is prohibited.
- Schema is the 40 official predictors plus `SepsisLabel`; `Hct` is required.
- `SepsisLabel` remains explicitly the Challenge's shifted persistent label.
  Reconstructed onset is `first positive ICULOS + 6 h` and is used only for
  fixed-policy early-warning analyses.
- Model selection, persistent-label sigmoid calibration, the Utility operating
  threshold, and the separate six-hour DCA sigmoid calibration occur only
  inside each outer training partition.
- Challenge Utility calls the pinned official PhysioNet scorer. Average
  Precision and trapezoidal PR-AUC are named separately.
- SourceSet A/B transport is internal public-data transport, not external
  validation. MIMIC/eICU external validation remains unavailable without the
  required credentialed cohorts.

Historical files under `results/` and `external_artifacts/` are retained as
evidence only. The corrected pipeline never reads them.
