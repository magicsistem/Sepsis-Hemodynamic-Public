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

`run.sh` is the only top-level entrypoint. It submits parameterized Slurm jobs
for tests, preparation, resource profiling, modeling, finalization, and
promotion, waits for each dependency, and returns nonzero unless every gate
passes. All scientific work is pinned to `compute-0-2`. A failed immutable
`runs/<run-id>/` directory is preserved; `RESUME_RUN_ID` reuses only complete,
hash- and source-bound stage checkpoints and refuses incompatible partials.

## Scientific scope

- Input is `data/raw/archive.zip`, pinned at SHA-256
  `1a0eb8040c76fdab84ee6c7dd6afdab4ad457a33d363cb7e4e200af713345897`
  and verified as 40,336 official patient-level PSV files, 40,336 patients and
  1,552,210 rows. CSV/TSV fallback is prohibited.
- Schema is the 40 official predictors plus `SepsisLabel`; `Hct` is required.
- The primary outcome is true reconstructed onset in 1--6 hours. Onset/post-
  onset rows, left-censored onset records, and the final six control hours are
  excluded from that estimand. The shifted persistent `SepsisLabel` is retained
  only for a clearly labelled secondary official-Utility analysis.
- C0 is baseline plus causal 8 h CV; C1 is the causal state; C2 adds simple
  observed deltas/slopes; C3 adds fold-local EDMD/Koopman innovations. Model,
  representation, tree count, calibration identity, and alarm threshold are
  selected without access to the outer fold.
- Useful alarms occur from onset−6 h through onset−1 h, with a six-hour
  refractory rising-edge policy and a maximum inner-OOF burden of 0.25 false
  episodes per eligible patient-day.
- Challenge Utility calls the pinned official PhysioNet scorer. Average
  Precision and trapezoidal PR-AUC are named separately.
- SourceSet A/B transport is internal public-data transport, not external
  validation. MIMIC/eICU external validation remains unavailable without the
  required credentialed cohorts.
- CPU/GPU profiles are measured before the model stage; selection is bounded
  to 32 CPU, 64 GB RAM, and one A100 40 GB, with no artificial memory fill.

Historical files under `results/` and `external_artifacts/` are retained as
evidence only. The corrected pipeline never reads them.
