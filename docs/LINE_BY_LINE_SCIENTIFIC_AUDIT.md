# Whole-source line-by-line scientific audit

Status: **IMPLEMENTED, EXECUTION EVIDENCE PENDING**. This is technical audit
evidence, not manuscript text. Line ranges refer to the source state prepared
after cancelling Slurm 25746; final commit and run hashes will be added only
after the corresponding gates pass.

## Reviewed active files

| File | Lines reviewed | Role and result |
|---|---:|---|
| `run.sh` | 1–90 | Sole orchestration path reviewed end to end: source gate, tests-before-run, immutable run ID, controlled resume, Slurm wait, terminal-state failure. No second top-level entrypoint. |
| `jobs/run_experiment.slurm` | 1–62 | Allocation and trap path reviewed: hard-pinned `compute-0-2`, one A100, environment capture, append-only ledger and container execution. |
| `scripts/run_experiment.py` | 1–37 | Thin run/validate adapter reviewed; it delegates every scientific definition to the canonical module and returns nonzero on `PipelineError`. |
| `scripts/source_provenance.py` | complete file | Every branch reviewed. The sidecar now hashes the raw ZIP and rejects missing, changed, path-traversing, or extra active runtime files on CEDIA. |
| `src/scientific_pipeline.py` | 1–end | Every function and caller reviewed from raw ZIP through final validation. Findings N016–N032 were corrected centrally and are listed below. |
| `tests/test_scientific_pipeline.py` | 1–end | Every oracle reviewed for independence from invalid historical numbers. New hand-computable or interface tests cover each executable correction. |
| `vendor/physionet2019/evaluate_sepsis_score.py` | 1–485 | Read completely and left unmodified. SHA-256 `26b8b26267ed32e8b7a7a27e45201cfc8c6640e717ba4cdc1f452b32f12b99e5` is enforced before scoring. |
| `vendor/physionet2019/__init__.py` | complete | Empty package marker only; no second scorer. |
| `vendor/physionet2019/README.md` | complete | Pin/source notice reviewed against the enforced scorer identity. |
| `vendor/physionet2019/LICENSE.txt` | complete | Vendored scorer license reviewed; no executable behavior. |

The tracked raw ZIP is binary input, not code; it is audited by exact SHA-256,
member paths, schema, numeric content, chronology, labels and exact cohort
counts. Tracked `results/**` and `external_artifacts/**` are withdrawn historical
evidence. A reachability scan found no active Python or shell reader for either
tree; both are excluded from current source/result manifests and cannot be
promoted by the pipeline.

## Cross-file interfaces traced

1. `run.sh` → source sidecar → Slurm job → `run.sh --inside-slurm` → tests →
   `scripts/run_experiment.py` → `run_scientific_pipeline` → standalone final
   validation. Any nonzero result prevents the completion marker.
2. Raw archive → exact schema/numeric/chronology gate → harmonized checkpoint →
   causal feature checkpoint → exact patient outcome folds → nested inner
   selection/calibration/threshold → held-out OOF rows.
3. OOF rows → one official Utility wrapper, discrimination, calibration,
   onset-anchored alarms, six-hour DCA, patient-cluster inference, transport,
   robustness and ablations → transitive lineage and exact artifact inventory.
4. `result_manifest.json` is accepted only if policies, cohort identity, fold
   hashes, OOF identities, nested-selection rows, reliability, DCA, inference,
   transport and ablation schemas agree with the generating source.

## New findings from this pass

| ID | Defect | Central correction | Oracle/evidence state |
|---|---|---|---|
| N016 | BH adjustment accumulated in the wrong direction. | Canonical reverse cumulative minimum. | Hand-computable three-p-value oracle; CEDIA pending. |
| N017 | Corrupt/infinite observed values could become missing. | Shared fail-closed numeric parser at raw, harmonized and model boundaries. | Corrupt and infinity oracles; CEDIA pending. |
| N018 | SampEn `m` count included an incompatible terminal template. | Both counts use the same `N-m` starts. | Manual B/A count oracles; CEDIA pending. |
| N019 | A bootstrap test incorrectly required a percentile CI to contain its point estimate. | Test now checks deterministic ordered percentile bounds. | CEDIA pending. |
| N020 | Dead `patient_metric_values` duplicated unused logic. | Deleted. | AST caller scan: zero callers. |
| N021 | Several derived products omitted probability/threshold/model/feature provenance. | Added exact provenance fields and feature hashes at their shared producers. | Artifact-schema/full-run validation pending. |
| N022 | The final manifest accepted unlisted extra run artifacts. | Exact observed-versus-declared inventory equality. | Unexpected-file oracle; CEDIA pending. |
| N023 | Raw archive bytes and exact cohort counts were recorded but not enforced. | Pinned archive SHA and exact files/patients/rows/A/B counts. | Local SHA and cohort mismatch oracle; full run pending. |
| N024 | Fold artifact could contain extra patients or changed patient outcomes. | Exact patient set and outcome equality. | Tampered-label oracle; CEDIA pending. |
| N025 | Technical docs overstated Git checkout and no-resume behavior. | Documents now describe sidecar execution and controlled compatible resume. | Documentation consistency scan pending. |
| N026 | Persistent positive state could reopen as a useful episode after refractory time. | Episodes require a new negative-to-positive crossing. | Persistent-remote alert oracle; CEDIA pending. |
| N027 | DCA treated eventual sepsis as the action outcome regardless of horizon. | Observed pre-onset decision-hour outcome is onset within six hours. | Two-patient manual net-benefit oracle; CEDIA pending. |
| N028 | “Calibration intercept” conflated CITL with the joint calibration intercept. | Separate fixed-slope CITL and joint intercept/slope. | Balanced calibration oracle; CEDIA pending. |
| N029 | ECE accepted arbitrary bins while always claiming ten. | One policy-defined equal-width ten-bin estimator. | Manual ECE oracle; CEDIA pending. |
| N030 | Sidecar allowed an extra executable file on CEDIA. | Exact active runtime inventory comparison. | Extra-source oracle; CEDIA pending. |
| N031 | Resume context omitted container, dependencies and backend. | Exact scientific runtime-context equality. | Context mutation oracle; CEDIA pending. |
| N032 | Final validation mostly checked hashes, not scientific product contracts. | Added cohort, fold, OOF, selection, DCA, reliability, inference, transport and ablation gates. | Full-run validation pending. |

No item in this table is `CLOSED` until the updated suite, complete run and
independent post-run audit pass on `compute-0-2`.
