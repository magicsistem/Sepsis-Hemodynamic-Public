# Whole-source line-by-line scientific audit

Status: **2026-09-24 PRE-RUN SOURCE REVIEW COMPLETE; EXACT SUITE/RUN PENDING**.
This is technical audit evidence, not manuscript text. Historical runs,
including jobs 26224, 27093 and 27295, remain preserved evidence and are not
promoted. The next full run must start from raw preparation and enter model
development at C0 only after the exact committed suite passes.

## Reviewed active files

| File | Lines reviewed | Role and result |
|---|---:|---|
| `run.sh` | 1–225 | Sole orchestration path: clean-source gate, tests, prepare, six profiles, selected-layout benchmark, C0-first model stage, finalize/promotion, resume and scheduler failure handling. |
| `jobs/run_experiment.slurm` | 1–133 | Hard-pinned `compute-0-2`, bounded stage/model threads, no oversubscription, environment capture, timing, append-only ledger, resource aggregation and promotion. |
| `scripts/run_experiment.py` | 1–50 | Thin stage adapter; all scientific definitions remain canonical and `PipelineError` returns nonzero. |
| `scripts/source_provenance.py` | 1–122 | Sidecar hashes every tracked source and raw input; rejects dirty, missing, changed, path-traversing or extra active files. |
| `scripts/profile_resources.py` | 1–230 | Fixed-stratum benchmark, timing boundaries, full-memory estimator and exact selected-layout benchmark reviewed. |
| `scripts/resource_provenance.py` | 1–524 | Profile eligibility/selection, 64 CPU/64 GB/one-GPU caps, exact 16-CPU prepare and 4-CPU finalize requests, selected benchmark binding and stage aggregation reviewed. |
| `src/onset_koopman.py` | 1–1016 | Target, fold-local signal selection, EDMD/Ridge, causal transforms, calibration, alarms, patient-balanced metrics, bootstraps and DCA reviewed function by function and caller by caller. |
| `src/scientific_pipeline.py` | 1–3341 | Raw archive through final promotion reviewed line by line. Nested selection, transport, C0–C3 reporting, lineage, exact recomputation and new deterministic stage pools were traced end to end. |
| `tests/test_scientific_pipeline.py` | 1–677 | Scientific oracles reviewed; includes serial/parallel prepare parity without historical expected metrics. |
| `tests/test_onset_koopman.py` | 1–957 | Target/Koopman/nested/inference/resource/document oracles reviewed; exact container execution remains the pre-run gate. |
| `vendor/physionet2019/evaluate_sepsis_score.py` | 1–485 | Read completely and left unmodified. SHA-256 `26b8b26267ed32e8b7a7a27e45201cfc8c6640e717ba4cdc1f452b32f12b99e5` is enforced before scoring. |
| `vendor/physionet2019/__init__.py` | complete | Empty package marker only; no second scorer. |
| `vendor/physionet2019/README.md` | complete | Pin/source notice reviewed against the enforced scorer identity. |
| `vendor/physionet2019/LICENSE.txt` | complete | Vendored scorer license reviewed; no executable behavior. |
| `README.md`, `MANIFEST.md`, `.gitignore`, `GATES.md`, `docs/**` | complete files | Operational/scientific claims checked against current code. No paper file is present or modified. Stale resource text was corrected. |

The tracked raw ZIP is binary input, not code; it is audited by exact SHA-256,
member paths, schema, numeric content, chronology, labels and exact cohort
counts. Tracked `results/**` and `external_artifacts/**` are withdrawn historical
evidence. A reachability scan found no active Python or shell reader for either
tree; both are excluded from current source/result manifests and cannot be
promoted by the pipeline.

## Cross-file interfaces traced

1. `run.sh` → source sidecar → Slurm wrapper → `run.sh --inside-slurm` → tests →
   stage adapter → prepare/profile/model/finalize → resource aggregation →
   independent promotion validation. Any nonzero result prevents completion.
2. Raw archive → exact schema/numeric/chronology gate → harmonized checkpoint →
   causal feature checkpoint → exact patient outcome folds → nested inner
   selection/outcome-specific calibration/threshold → held-out OOF rows.
3. OOF rows → one official Utility wrapper, discrimination, calibration,
   onset-anchored alarms, six-hour DCA, patient-cluster inference, transport,
   robustness and ablations → transitive lineage and exact artifact inventory.
4. `result_manifest.json` is accepted only if policies, cohort identity, fold
   hashes, OOF identities, nested-selection rows, reliability, DCA, inference,
   transport and ablation schemas agree with the generating source.

## New findings from this pass

| ID | Defect | Central correction | Oracle/evidence state |
|---|---|---|---|
| N016 | BH adjustment accumulated in the wrong direction. | Canonical reverse cumulative minimum. | Hand-computable oracle passed in job 26121. |
| N017 | Corrupt/infinite observed values could become missing. | Shared fail-closed numeric parser at raw, harmonized and model boundaries. | Corrupt and infinity oracles passed in job 26121. |
| N018 | SampEn `m` count included an incompatible terminal template. | Both counts use the same `N-m` starts. | Manual B/A count oracles passed in job 26121. |
| N019 | A bootstrap test incorrectly required a percentile CI to contain its point estimate. | Test now checks deterministic ordered percentile bounds. | Corrected oracle passed in job 26121. |
| N020 | Dead `patient_metric_values` duplicated unused logic. | Deleted. | AST caller scan: zero callers. |
| N021 | Several derived products omitted probability/threshold/model/feature provenance. | Added exact provenance fields and feature hashes at their shared producers. | Schema oracles passed; final artifact audit pending. |
| N022 | The final manifest accepted unlisted extra run artifacts. | Exact observed-versus-declared inventory equality. | Unexpected-file oracle passed in job 26121. |
| N023 | Raw archive bytes and exact cohort counts were recorded but not enforced. | Pinned archive SHA and exact files/patients/rows/A/B counts. | Cohort mismatch oracle passed; full run pending. |
| N024 | Fold artifact could contain extra patients or changed patient outcomes. | Exact patient set and outcome equality. | Tampered-label oracle passed in job 26121. |
| N025 | Technical docs overstated Git checkout and no-resume behavior. | Documents now describe sidecar execution and controlled compatible resume. | Consistency scan passed locally. |
| N026 | Persistent positive state could reopen as a useful episode after refractory time. | Episodes require a new negative-to-positive crossing. | Persistent-remote alert oracle passed in job 26121. |
| N027 | DCA treated eventual sepsis as the action outcome regardless of horizon. | Observed pre-onset decision-hour outcome is onset within six hours. | Manual net-benefit oracle passed in job 26121. |
| N028 | “Calibration intercept” conflated CITL with the joint calibration intercept. | Separate fixed-slope CITL and joint intercept/slope. | Balanced calibration oracle passed in job 26121. |
| N029 | ECE accepted arbitrary bins while always claiming ten. | One policy-defined equal-width ten-bin estimator. | Manual ECE oracle passed in job 26121. |
| N030 | Sidecar allowed an extra executable file on CEDIA. | Exact active runtime inventory comparison. | Extra-source oracle passed in job 26121. |
| N031 | Resume context omitted container, dependencies and backend. | Exact scientific runtime-context equality. | Context mutation oracle passed in job 26121. |
| N032 | Final validation mostly checked hashes, not scientific product contracts. | Added cohort, fold, OOF, selection, DCA, reliability, inference, transport and ablation gates. | Full-run validation pending. |
| N033 | Two test fixtures encoded incorrect assumptions. | Correct hand-ranked AP data and canonical feature construction. | Corrected suite passed in jobs 25762 onward. |
| N034 | pandas 1.5 retained `object` dtype after block `.loc` assignment. | Assign the complete named block so numeric dtypes materialize. | String-input oracle passed in jobs 25765 onward. |
| N035 | Unscaled calibration regression was ill-conditioned in a bootstrap draw. | Standardize weighted logits for fitting and transform coefficients back. | Numerical/non-identifiability oracles passed in jobs 25831 and 26121. |
| N036 | DCA used persistent-label probabilities for the different six-hour-onset outcome. | Fit a separate sigmoid on eligible inner-OOF six-hour outcomes and apply it only to the outer fold. | Eligibility, onset, target, probability-source and nested-isolation oracles passed in job 26121; full run pending. |
| N037 | Provenance called analyses predeclared/prespecified without preregistration evidence. | Use “declared computational family” and “fixed feature-family ablations”; remove the unreachable fallback claim. | Source oracle passed in jobs 26123/26128; full run pending. |
| N038 | Alarm rates called their denominator observed decision hours but used elapsed ICULOS span. | Divide by the actual count of observed decision rows. | Irregular-ICULOS oracle passed in job 26128; full run pending. |
| N039 | The DCA threshold range was duplicated in generation and validation but absent from the hashed policy. | Put the fixed 0.05–0.50 grid in `FEATURE_POLICY` and consume it in both paths. | Exact policy/grid oracle passed in job 26128; full run pending. |
| N040 | Left-censored and late pre-onset alarm episodes could remain uncategorized in summary output. | Report them separately without relabelling unidentifiable episodes as false or useful. | Exhaustive category oracle passed in job 26128; full run pending. |
| N041 | The Slurm job requested 16 CPU and 128 GB although XGBoost uses at most 8 threads and the prior complete run peaked at 19.50 GB. | Request 8 CPU and 32 GB; retain one required A100 and the fail-closed 48-hour limit. | `sacct` evidence from complete job 25832 and cancelled job 26130; shell syntax check required. |
| N042 | Effective model/calibration constants were source-controlled but not emitted together as a hashed model policy; calibration used the undocumented approximation `C=1e6`. | Centralize and hash the exact model policy, use the same values at every caller, use supported unpenalized logistic calibration, and carry the exact source-inventory hash into the run manifest. | Static policy/caller/provenance oracles added; accumulated CEDIA test required before a new run. |
| N043 | A test opened withdrawn files under `results/` even though that historical tree is intentionally excluded from source provenance and laptop-to-CEDIA synchronization. It passed only while stale remote copies existed. | Delete the obsolete historical-file test; withdrawal remains a documented policy, while every executable test now depends only on the validated active source inventory. | Negative reachability scan and clean-source CEDIA suite required. |
| N044 | Fold provenance converted patient outcomes to integer before equality and did not require integer, policy-complete fold IDs; values such as label/fold `0.5` could cross the boundary incorrectly. | Validate exact binary patient outcomes and integer fold IDs covering `0..outer_folds-1` before merging. | Fractional-label, fractional-fold and missing-fold negative oracles added; accumulated CEDIA test required. |
| N045 | Final validation checked the presence of nested-selection columns but not whether their values matched the hashed model policy. | Validate candidates and exact hyperparameters, tree-count bounds, both calibrator identities, threshold grid, inner-fold hashes, patient counts and GPU state before promotion. | Valid compact artifact plus invalid-threshold negative oracle added; accumulated CEDIA test required. |
| N046 | The shared discrimination helper attached XGBoost training metadata to transport, stability and even the separate logistic-robustness classifier. | Keep the shared helper metric-only and record XGBoost `logloss` solely in the hashed XGBoost model policy. | AP/PR identity oracle now rejects trainer metadata in the generic metric result; accumulated CEDIA test required. |
| N047 | Split-stability rows did not identify their calibrated probability/outcome or the between-fold ranking limitation; classifier-robustness rows omitted their uncalibrated probability identity. | Add explicit probability kind, outcome estimand and calibrated-pooling interpretation at the shared producers, and require stability provenance at final promotion. | Logistic provenance oracle and final stability schema gate added; accumulated CEDIA test required. |
| N048 | Utility-threshold selection used `np.isclose`, so merely near-equal values could be treated as ties and bias selection toward a lower threshold. | Select the exact deterministic maximum; choose the lowest threshold only among exactly equal maxima, and record the tie policy in the hashed configuration. | Near-equal two-threshold oracle added; accumulated CEDIA test required. |
| N049 | Calibration clipped probabilities before Brier and ECE, changing canonical metric values at exact 0/1; several metric paths lacked one shared range gate. | Validate finite one-dimensional `[0,1]` probabilities once, use original probabilities for Brier/ECE/reliability/discrimination, and clip only for logit regression. | Perfect 0/1 Brier/ECE oracle plus out-of-range negative oracle added; accumulated CEDIA test required. |
| N050 | Utility, six-hour DCA and early-warning paths still accepted finite scores outside `[0,1]`, and Utility threshold parsing was not fail-closed. | Reuse the shared probability/outcome gates at every decision boundary and require a finite Utility threshold in `[0,1]`. | Four negative decision-path oracles added; accumulated CEDIA test required. |
| N051 | Hash validation did not prove OOF identity/folds against source artifacts or reproduce primary report values from the serialized prediction artifact. | Re-read serialized OOF before reporting; compare identity/folds/thresholds to features, folds and nested selection; reproduce AUROC/AP/PR-AUC/Brier before final promotion. | Final-validator interface oracles added; accumulated CEDIA test required. |
| N052 | Several outcome paths converted to integer before validation, and longitudinal decision helpers did not share an independent chronology/persistence gate. | Validate before conversion in calibration, folds, summaries, transport and inference; share one exact binary/time/persistence gate across Utility, six-hour decisions and alarms. | Fractional target/score, onset chronology and nonpersistent-label oracles added; accumulated CEDIA test required. |
| N053 | Two focused unit fixtures used two folds without declaring that synthetic policy, so the new production five-fold gate correctly rejected them. | Scope `outer_folds=2` only inside those two fixtures; production policy and assertions remain five. | Job 26221 preserved as failed evidence; corrected job 26223 passed 40/40. |
| N054 | pandas 1.5 returned `read_csv(usecols=...)` in file order, while final OOF identity comparison expected the requested list order; semantically identical job-26224 rows therefore failed `DataFrame.equals`. | Reindex the selected feature identity explicitly to the canonical identity-column order before comparison. | Local compile/shell/diff checks pass; recovered artifacts pass 52/52 declared hashes and exact 1,552,210-row features/folds/baseline/enhanced OOF identity. No replacement run by explicit user direction. |
| N078 | `prepare` parsed 40,336 archive members and engineered 40,336 patients serially despite independent patient boundaries. | Reuse one ordered stdlib pool: 16 archive batches, each with its own `ZipFile`, and 16 contiguous patient-complete feature batches; stable sort restores canonical order and worker exceptions propagate. | Serial/parallel harmonization and feature parity oracle added; exact container suite pending. |
| N079 | C0–C3 report generation and the same independent artifact checks ran serially, leaving finalize/validation on one CPU. | Reuse the ordered pool across the four independent representations with four workers and one native math thread each. | Syntax and source audit pass; exact container suite and final-run resource evidence pending. |
| N080 | Finalize RAM was coupled to the selected XGBoost per-fit memory although reporting has a different workload. | Remove the derived `finalize_memory_gb`; use explicit 4 CPU/32 GB finalize and 16 CPU/32 GB prepare contracts, while 64 CPU remains exclusive to measured model concurrency. | Resource aggregation/orchestration oracles updated; exact container suite pending. |

Run-dependent items remain open until the updated suite and final validator
execute on `compute-0-2`; preserved historical outputs are evidence, not a
promoted final result.
