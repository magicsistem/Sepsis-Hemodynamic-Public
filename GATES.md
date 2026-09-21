# Gates: corrected scientific pipeline

OWNS: run.sh, jobs/**, scripts/**, src/**, tests/**, docs/**, README.md, MANIFEST.md, .gitignore

Scope: Replace the audited public pipeline with one scientifically valid, fail-closed, reproducible experiment, while retaining every unresolved human or external-data limitation visibly.

- [x] G1: the private exhaustive finding ledger is excluded from Git and has one row for every audit finding
  CHECK: test -f EXPERIMENT_PROGRESS.md && git check-ignore -q EXPERIMENT_PROGRESS.md && test "$(awk '/^\| (B|F|M)[0-9]/{n++} END{print n+0}' EXPERIMENT_PROGRESS.md)" -eq 234 && node -e "setTimeout(() => console.log('finding-ledger-verification-passed'), 20)"
  EXPECT: finding-ledger-verification-passed
  EVIDENCE: 2026-09-07 local exit 0; 234/234 Memory IDs, EXPERIMENT_PROGRESS.md ignored by .git/info/exclude; output finding-ledger-verification-passed

- [ ] G2: all executable scientific, mathematical, provenance, and fail-closed tests pass
  EVIDENCE: historical job 26808 passed 48/48 for commit 78dc8d7; the current v6 robustness/resource source requires a new clean committed CEDIA suite before this gate can close.

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

- [ ] G9: Methodology matches the final code and defines every implemented estimand, formula, parameter, edge case, and inferential unit; references are checked against original sources
  EVIDENCE: v6 adds prespecified seed/balance sensitivity and active-resource selection; final post-run code-to-method trace remains pending. No manuscript text is drafted.

- [ ] G10: Results use only the final corrected rerun, distinguish non-executed/external-data limitations, and pass two complete traceability readings
  EVIDENCE: pending experimental reporting audit; no paper text will be drafted.

- [ ] G11: every active tracked source file and every cross-file scientific interface has a recorded line-by-line review with zero unrecorded scientific defects
  EVIDENCE: the previous audit covers the v4 source through N054, but the new direct-onset/Koopman source and interfaces require the requested post-run independent line-by-line audit before this gate can close.


## Gates: direct-onset Koopman experiment

- [ ] G12: the primary target is exactly true onset in 1--6 hours, excludes onset/post-onset and left-censored patients, and excludes the final six control hours needed for complete follow-up
  CHECK: python -m unittest tests.test_onset_koopman.OnsetTargetTests -v && printf 'G12_TARGET_ORACLES_PASS\n'
  EXPECT: G12_TARGET_ORACLES_PASS
  EVIDENCE: historical oracle evidence exists; current v6 exact-commit CEDIA suite pending.

- [ ] G13: fold-local state, delta/slope, and Koopman transforms pass zero-residual, anomaly, sparse-support, fixed-schema, float32, and no-future oracles
  CHECK: python -m unittest tests.test_onset_koopman.KoopmanOracleTests -v && printf 'G13_KOOPMAN_ORACLES_PASS\n'
  EXPECT: G13_KOOPMAN_ORACLES_PASS
  EVIDENCE: historical oracle evidence exists; current v6 exact-commit CEDIA suite pending.

- [ ] G14: C0--C3 model selection, representation fitting, calibration choice, and alarm threshold selection use inner-training/inner-OOF data only
  CHECK: python -m unittest tests.test_onset_koopman.NestedPolicyTests -v && printf 'G14_NESTED_POLICY_PASS\n'
  EXPECT: G14_NESTED_POLICY_PASS
  EVIDENCE: inner held-out predictions now remove redundant refits and seed/balance sensitivities reuse only outer-train-selected settings; current v6 exact-commit CEDIA suite pending.

- [ ] G15: primary patient-balanced AP inference, six-hour alarm budget, lead time, DCA, and A-to-B/B-to-A transport are recomputable from current-run artifacts without destination labels
  CHECK: python -m unittest tests.test_onset_koopman.InferenceTransportTests -v && printf 'G15_INFERENCE_TRANSPORT_PASS\n'
  EXPECT: G15_INFERENCE_TRANSPORT_PASS
  EVIDENCE: historical oracle evidence exists; current v6 exact-commit CEDIA suite and final artifact recomputation pending.

- [ ] G16: run.sh remains the sole entrypoint, submits only compute-0-2 jobs, bounds CPU/RAM/GPU, records resource measurements, and links prepare/model/finalize fail-closed
  CHECK: python -m unittest tests.test_onset_koopman.ResourceOrchestrationTests -v && bash -n run.sh && bash -n jobs/run_experiment.slurm && printf 'G16_RESOURCE_ORCHESTRATION_PASS\n'
  EXPECT: G16_RESOURCE_ORCHESTRATION_PASS
  EVIDENCE: v6 changes login-node Python, active CPU/GPU eligibility, one-second GPU sampling, and measured prepare resources; current exact-commit CEDIA suite/profile pending.

- [ ] G17: the methodological plan records current evidence, reproducible searches, mathematics, rejected alternatives, IEEE references from 2021--2026, official-standard exceptions, and explicit Zahibi/Zabihi exclusion without modifying the paper
  CHECK: python -m unittest tests.test_onset_koopman.MethodologyDocumentTests -v && printf 'G17_METHODOLOGY_DOCUMENT_PASS\n'
  EXPECT: G17_METHODOLOGY_DOCUMENT_PASS
  EVIDENCE: plan now includes reviewer-requested ablations, seeds, balance and resource policy; current exact-commit test plus final paper-file diff audit pending.

- [ ] G18: the complete accumulated local/container test suite and static gates pass from a clean committed checkout
  EVIDENCE: local v6 static syntax/diff checks pass; clean committed CEDIA suite pending.

- [ ] G19: a fixed-subset CEDIA benchmark measures CPU/GPU profiles and selects the smallest profile within 5% of the fastest, with measured memory plus 20% margin and all caps enforced
  EVIDENCE: pending profiling through bash run.sh --profile.

- [ ] G20: one clean bash run.sh execution completes prepare, model, and finalize; all hashes, OOF identities, nested provenance, resource manifests, and final scientific gates pass an independent audit
  EVIDENCE: pending final CEDIA run and independent audit; no PASS may be inferred from process exit alone.

- [ ] G21: the 2026-09-21 reviewer recommendations are separated into executable evidence and manuscript-only actions without changing the paper
  CHECK: python -m unittest tests.test_onset_koopman.NestedPolicyTests tests.test_onset_koopman.MethodologyDocumentTests -v
  EVIDENCE: C0--C3 ablation plus three seeds and three training-balance policies are implemented as non-selective sensitivity artifacts. Journal citations/title/Abstract/contributions/Discussion/Conclusion/future-work/special-issue alignment remain explicitly DEFERRED_TO_MANUSCRIPT_PHASE. Current exact-commit CEDIA suite and final artifacts pending.
