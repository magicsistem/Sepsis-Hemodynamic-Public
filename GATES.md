# Gates: corrected scientific pipeline

OWNS: run.sh, jobs/**, scripts/**, src/**, tests/**, docs/**, README.md, MANIFEST.md, .gitignore

Scope: Replace the audited public pipeline with one scientifically valid, fail-closed, reproducible experiment, while retaining every unresolved human or external-data limitation visibly.

- [x] G1: the private exhaustive finding ledger is excluded from Git and has one row for every audit finding
  CHECK: test -f EXPERIMENT_PROGRESS.md && git check-ignore -q EXPERIMENT_PROGRESS.md && test "$(awk '/^\| (B|F|M)[0-9]/{n++} END{print n+0}' EXPERIMENT_PROGRESS.md)" -eq 234 && node -e "setTimeout(() => console.log('finding-ledger-verification-passed'), 20)"
  EXPECT: finding-ledger-verification-passed
  EVIDENCE: 2026-09-07 local exit 0; 234/234 Memory IDs, EXPERIMENT_PROGRESS.md ignored by .git/info/exclude; output finding-ledger-verification-passed

- [x] G2: all executable scientific, mathematical, provenance, and fail-closed tests pass
  EVIDENCE: job 26330 passed 58/58 tests in 12.011 s on compute-0-2 for exact commit a0c0355 and source inventory ef1f6955; scheduler exit, resource record and append-only ledger all report PASS/COMPLETED.

- [ ] G3: the canonical run.sh completes an end-to-end CEDIA run and validates the final result manifest
  EVIDENCE: job 26224 generated the complete artifact family but failed closed at final validation because pandas 1.5 returned `usecols` in file order; the preserved manifest remains PENDING_FINAL_VALIDATION. The user explicitly prohibited a replacement full run.

- [ ] G4: every finding is closed with current evidence or explicitly classified as author action, external-data blocked, or manuscript deferred
  EVIDENCE: N054 is implemented and the recovered artifacts pass all 52 declared hashes plus exact 1,552,210-row OOF identity, but the corrected source was not executed on CEDIA.

- [ ] G5: an independent final audit verifies data lineage, no-future features, nested selection, calibration, Utility, metric identities, and reporting traceability
  EVIDENCE: the complete job-26224 package was recovered locally; all 52 declared artifact hashes and exact features/folds/baseline/enhanced OOF identity passed. Remaining final-validator checks were not executed after N054, so no PASS is claimed.


## Gates: methodology and results reconstruction

- [x] G6: every project source, historical result family, Git transition, and current run artifact is classified by role and current validity
  EVIDENCE: docs/EXPERIMENT_RECONSTRUCTION.md and docs/LINE_BY_LINE_SCIENTIFIC_AUDIT.md classify active source, superseded outputs, raw binary input, run history, and unreachable historical result trees; reviewed again before job 26130 was allowed to continue.

- [ ] G7: the current corrected rerun is independently revalidated and every reported number maps to a hash-verified artifact
  EVIDENCE: pending corrected rerun after N036.

- [ ] G8: the master result table includes every executed current model, calibration, transport, stability, subgroup, temporal, DCA, inference, and ablation result
  EVIDENCE: pending corrected rerun artifact inventory; no manuscript is in scope.

- [x] G9: Methodology matches the final code and defines every implemented estimand, formula, parameter, edge case, and inferential unit; references are checked against original sources
  EVIDENCE: docs/reproducibility.md and docs/reference_verification.md were traced against all active producers and the pinned official scorer; no manuscript text was drafted.

- [ ] G10: Results use only the final corrected rerun, distinguish non-executed/external-data limitations, and pass two complete traceability readings
  EVIDENCE: pending experimental reporting audit; no paper text will be drafted.

- [ ] G11: every active tracked source file and every cross-file scientific interface has a recorded line-by-line review with zero unrecorded scientific defects
  EVIDENCE: the previous audit covers the v4 source through N054, but the new direct-onset/Koopman source and interfaces require the requested post-run independent line-by-line audit before this gate can close.


## Gates: direct-onset Koopman experiment

- [x] G12: the primary target is exactly true onset in 1--6 hours, excludes onset/post-onset and left-censored patients, and excludes the final six control hours needed for complete follow-up
  CHECK: python -m unittest tests.test_onset_koopman.OnsetTargetTests -v && printf 'G12_TARGET_ORACLES_PASS\n'
  EXPECT: G12_TARGET_ORACLES_PASS
  EVIDENCE: job 26330 ran both OnsetTargetTests under PYTHONWARNINGS=error within the 58/58 passing suite.

- [x] G13: fold-local state, delta/slope, and Koopman transforms pass zero-residual, anomaly, sparse-support, fixed-schema, float32, and no-future oracles
  CHECK: python -m unittest tests.test_onset_koopman.KoopmanOracleTests -v && printf 'G13_KOOPMAN_ORACLES_PASS\n'
  EXPECT: G13_KOOPMAN_ORACLES_PASS
  EVIDENCE: job 26330 ran all three KoopmanOracleTests under PYTHONWARNINGS=error within the 58/58 passing suite.

- [x] G14: C0--C3 model selection, representation fitting, calibration choice, and alarm threshold selection use inner-training/inner-OOF data only
  CHECK: python -m unittest tests.test_onset_koopman.NestedPolicyTests -v && printf 'G14_NESTED_POLICY_PASS\n'
  EXPECT: G14_NESTED_POLICY_PASS
  EVIDENCE: job 26330 ran all three NestedPolicyTests, including provenance-tamper rejection, within the 58/58 passing suite.

- [x] G15: primary patient-balanced AP inference, six-hour alarm budget, lead time, DCA, and A-to-B/B-to-A transport are recomputable from current-run artifacts without destination labels
  CHECK: python -m unittest tests.test_onset_koopman.InferenceTransportTests -v && printf 'G15_INFERENCE_TRANSPORT_PASS\n'
  EXPECT: G15_INFERENCE_TRANSPORT_PASS
  EVIDENCE: job 26330 ran all four InferenceTransportTests within the 58/58 passing suite.

- [x] G16: run.sh remains the sole entrypoint, submits only compute-0-2 jobs, bounds CPU/RAM/GPU, records resource measurements, and links prepare/model/finalize fail-closed
  CHECK: python -m unittest tests.test_onset_koopman.ResourceOrchestrationTests -v && bash -n run.sh && bash -n jobs/run_experiment.slurm && printf 'G16_RESOURCE_ORCHESTRATION_PASS\n'
  EXPECT: G16_RESOURCE_ORCHESTRATION_PASS
  EVIDENCE: job 26330 ran all three ResourceOrchestrationTests; shell syntax checks passed locally and the allocated test job recorded compute-0-2, CPU partition, 2 CPU, 8 GiB and resource status PASS.

- [x] G17: the methodological plan records current evidence, reproducible searches, mathematics, rejected alternatives, IEEE references from 2021--2026, official-standard exceptions, and explicit Zahibi/Zabihi exclusion without modifying the paper
  CHECK: python -m unittest tests.test_onset_koopman.MethodologyDocumentTests -v && printf 'G17_METHODOLOGY_DOCUMENT_PASS\n'
  EXPECT: G17_METHODOLOGY_DOCUMENT_PASS
  EVIDENCE: job 26330 passed MethodologyDocumentTests; no paper source is modified in the direct-onset commit range.

- [x] G18: the complete accumulated local/container test suite and static gates pass from a clean committed checkout
  EVIDENCE: exact committed source a0c0355/inventory ef1f6955 passed 58/58 in job 26330; py_compile, both shell syntax checks and git diff --check also passed before synchronization.

- [ ] G19: a fixed-subset CEDIA benchmark measures CPU/GPU profiles and selects the smallest profile within 5% of the fastest, with measured memory plus 20% margin and all caps enforced
  EVIDENCE: pending profiling through bash run.sh --profile.

- [ ] G20: one clean bash run.sh execution completes prepare, model, and finalize; all hashes, OOF identities, nested provenance, resource manifests, and final scientific gates pass an independent audit
  EVIDENCE: pending final CEDIA run and independent audit; no PASS may be inferred from process exit alone.
