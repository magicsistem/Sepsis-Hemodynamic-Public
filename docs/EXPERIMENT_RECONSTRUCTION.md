# Experimental reconstruction record

Status: **IN PROGRESS**. This is an evidence map, not manuscript text. Run
`20260911T185228Z-bb3e353` (Slurm 25832) completed but was superseded when the
independent audit found N036 in its DCA probability estimand. Job 26121 passed
41/41 tests for that correction; job 26122 was then cancelled and preserved
when the continuing audit found N037. Numerical results remain blocked until
the next corrected run and its independent audit pass.

## Evidence authority

When sources disagree, this reconstruction applies the following non-substitutable roles:

1. `Memory.updated.md` and `PONYTAIL_AUDIT.md`: historical forensic findings and required corrections.
2. Current tracked code at the scientific run commit: exact implementation.
3. Immutable `runs/<run-id>/` artifacts plus `result_manifest.json`: what actually executed and the only numerical source.
4. `GATES.md`, checked against code, hashes, Slurm, logs and artifacts: acceptance status.
5. Git: chronology and supersession.
6. `README.md`, `MANIFEST.md`, and `docs/*.md`: current explanatory documentation; these do not override code or run evidence.

The request's “sources” means the project's source files as a whole; no literal `sources/` directory exists in the current checkout.

## Document classification

| Class | Evidence | Current use |
|---|---|---|
| CANONICAL CURRENT | `run.sh`; `jobs/run_experiment.slurm`; `scripts/run_experiment.py`; `scripts/source_provenance.py`; `src/scientific_pipeline.py`; pinned `vendor/physionet2019`; `tests/test_scientific_pipeline.py` | Defines the active fail-closed experiment. |
| CANONICAL CURRENT | `GATES.md`; current-run `runtime_manifest.json`, stage manifests and final `result_manifest.json` | Acceptance and provenance, conditional on independent verification. |
| AUDIT/CORRECTION | `/home/miguel/Documents/Memory.updated.md`; `/home/miguel/Documents/PONYTAIL_AUDIT.md`; ignored `EXPERIMENT_PROGRESS.md`; Git commits after `c96f15f` | Defines defects, remediation history and unresolved human/external items. The ignored ledger is operational evidence, never a public result. |
| DOCUMENTO INFORMATIVO | `README.md`, `MANIFEST.md`, `docs/reproducibility.md`, `docs/data_harmonization.md`, `docs/data_license.md`, `docs/results_policy.md`, `docs/artifact_inventory.md`, `docs/final_audit_summary.md` | Explains the current boundary; must agree with code/run evidence. |
| CANÓNICO HISTÓRICO/SUPERADO | Removed legacy trainers/reporters and ten Slurm entrypoints at `c96f15f`; every tracked file under `results/` and `external_artifacts/` | Audit history only. Their 287/305 features, isotonic calibration, thresholds, metrics, tables, figures and PASS labels are invalid for current claims. |
| CANÓNICO HISTÓRICO/SUPERADO | Run `20260908T145928Z-0dea39c`, job 25587 | Complete historical corrected run that passed its contemporary gates, but superseded after N011–N015. It may be used to understand chronology, never as the final numerical source. |

## Git and execution chronology

- `c96f15f`: audited public state with legacy parallel trainers/reporters, duplicated metrics/Utility, ten Slurm scripts and hard-coded historical reporting.
- `1c804a7`: replaced the active legacy path with one fail-closed pipeline and one root `run.sh`.
- Subsequent commits corrected source provenance, official Utility oracles, GPU detection/backend selection, chronology/schema, nested calibration, grouped split stability, container provenance, alarm burden, temporal/process ablations, lineage, left-censored onset and compact OOF artifacts.
- `0dea39c`: removed outer `Fold` from every model feature policy. Job 25587 then completed the whole contemporary pipeline.
- `47e38e7`: reopened the experiment after the manuscript reconstruction audit found N011–N015; corrected early-warning denominators/CI naming, added paired logistic inference, add-one-family ablations and exact runtime window policy.
- Job 25746 was cancelled after N016–N032 were found during whole-source line review; it is historical and cannot provide final numbers.
- `bb3e353`: stabilized calibration inference; job 25832 completed but its DCA is superseded by N036.
- `eeaea78`: added nested six-hour-outcome calibration for DCA; job 26121 passed 41/41 tests.
- `8e84bee`: records the N036 test gate; job 26122 was cancelled after N037 was found.

All cancelled/failed chains remain historical evidence in `logs/run_ledger.tsv`; no historical run is promoted merely because an intermediate job or test job says PASS.

## Gate chronology

| Gate/evidence family | Latest classification | Basis |
|---|---|---|
| Legacy `results/**` and `external_artifacts/**` PASS/checklists | HISTORICAL PASS — INVALIDATED | They validate the defective c96f15f pipeline and historical constants; six prominent certifications carry an explicit WITHDRAWN banner. |
| Test jobs before 25587 | HISTORICAL PASS — SUPERSEDED | Each was a preflight for an older source commit and did not produce a complete final run. |
| Failed/cancelled full jobs 25321, 25423, 25457, 25459, 25463, 25498 | HISTORICAL RUN — FAILED/CANCELLED | The ledger and Slurm retain terminal state; no final manifest was promoted. |
| G1 | CURRENT PASS | Exhaustive private ledger remains locally excluded from Git. |
| G2 | CURRENT PASS | Job 26121 passed 41/41 for scientific code commit `eeaea78`; `8e84bee` changes gate evidence only. |
| G3–G5 | PENDING | Run 26122 is cancelled historical evidence; no final result is accepted before a new manifest validation and independent audit. |
| G6–G11 | IN PROGRESS | Whole-source audit, reconstruction, reporting inventory and final traceability remain open. |

## Implemented pipeline inventory

| Stage/branch | Implementation | Required run evidence | Current state |
|---|---|---|---|
| Environment/provenance | exact source inventory including raw archive; pinned archive, container, utility scorer, versions, hash seed and validated GPU; resume context includes dependencies/backend | runtime manifest and Slurm log | TESTED; CORRECTED RERUN REQUIRED |
| Data gate | pinned ZIP PSV-only; exact cohort counts; official 40 predictors plus label; explicit Hct aliases; source-qualified patient IDs; strict numeric/chronology/label checks | harmonized stage manifest | IMPLEMENTED; FULL RERUN REQUIRED |
| Outcome | shifted persistent Challenge label; exact onset = first 0→1 transition ICULOS + 6 h; first-row-positive cases are left-censored | harmonized data/cohort flow | IMPLEMENTED; FULL RERUN REQUIRED |
| Causal features | static predictors unchanged; current missingness, observation age, 24 h bounded last observation; current dynamic measurement count | feature manifest | IMPLEMENTED; FULL RERUN REQUIRED |
| Enhanced families | six hemodynamic signals; 8 h CV, 8 h IQR, observed-only 24 h SampEn over comparable starts and zero-match indicator | feature policy/hash and measurement support | IMPLEMENTED; FULL RERUN REQUIRED |
| Outer validation | five stratified patient-grouped folds with exact cohort/outcome provenance | folds manifest | IMPLEMENTED; FULL RERUN REQUIRED |
| Inner development | three grouped folds; depth-3/depth-5 XGBoost candidates; AP selection; inner-only early stopping/tree count | nested-selection CSVs | NOT YET EXECUTED |
| Calibration/policy | fixed persistent-label sigmoid on patient-balanced inner OOF; separate six-hour-onset sigmoid on eligible inner-OOF decision hours; Utility grid threshold from inner OOF only | selection and OOF artifacts | TESTED; CORRECTED RERUN REQUIRED |
| Primary OOF | baseline and enhanced outer-held-out raw/calibrated probabilities | two OOF files and metrics | NOT YET EXECUTED |
| Primary inference | patient-cluster paired randomization for AUROC/AP/Brier; BH family | inference CSV | NOT YET EXECUTED |
| Calibration | Brier, 10-bin ECE, intercept, slope, reliability rows and 300-draw patient bootstrap CIs | metric/reliability artifacts | NOT YET EXECUTED |
| Challenge Utility | unchanged official scorer at raw/calibrated 0.5 and nested fold thresholds | metric artifacts and scorer hash | NOT YET EXECUTED |
| Early warning | onset-anchored [−12 h, −1 h], 6 h refractory episodes, detection, paired timing and burden | model summaries, patient artifacts, paired comparison | NOT YET EXECUTED |
| Temporal/subgroup | ICU-time and onset-relative descriptive strata; age groups <50, [50,70), >=70 | temporal and age CSVs | NOT YET EXECUTED |
| DCA | assessment now for reconstructed onset in the next 6 h at observed pre-onset decision hours using its outcome-specific nested probability; thresholds 0.05–0.50; model/all/none with patient-cluster bootstrap | DCA CSVs | TESTED; CORRECTED RERUN REQUIRED |
| Stability | seeds 20260906, 20261007 and 20261108 | stability fold manifests and summary | NOT YET EXECUTED |
| Transport | train A→test B and train B→test A, separately for both feature variants | transport CSV | NOT YET EXECUTED |
| Classifier robustness | L2 logistic SGD baseline/enhanced on identical grouped folds plus paired inference | robustness and inference CSVs | NOT YET EXECUTED |
| Process sensitivities | physiology-measurements-only; without explicit process; without ICULOS; without HospAdmTime; without explicit missingness | ablations CSV | NOT YET EXECUTED |
| Feature-family controls | without CV/IQR/SampEn; baseline+CV/IQR/SampEn; fold-wise matched permutation of all enhanced-only columns | ablations CSV | NOT YET EXECUTED |
| Risk-of-bias support | machine-readable status only | PROBAST+AI status JSON | NOT YET EXECUTED; human appraisal remains required |
| External validation | credentialed independent cohort | unavailable | BLOCKED_EXTERNAL_DATA |

## Unresolved non-computational boundary

The experiment cannot establish hidden-test, prospective, temporal, hospital-C or independent external validation. MIMIC-IV/eICU full cohorts require access not present here. Author contribution, funding/in-kind support, AI-use disclosure, Zenodo DOI, software-license choice, claims of “prespecification,” and the completed PROBAST+AI signalling appraisal remain author actions. None will be invented or converted to PASS.
