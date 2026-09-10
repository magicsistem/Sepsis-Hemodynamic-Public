# Gates: corrected scientific pipeline

OWNS: run.sh, jobs/**, scripts/**, src/**, tests/**, docs/**, README.md, MANIFEST.md, .gitignore

Scope: Replace the audited public pipeline with one scientifically valid, fail-closed, reproducible experiment, while retaining every unresolved human or external-data limitation visibly.

- [x] G1: the private exhaustive finding ledger is excluded from Git and has one row for every audit finding
  CHECK: test -f EXPERIMENT_PROGRESS.md && git check-ignore -q EXPERIMENT_PROGRESS.md && test "$(awk '/^\| (B|F|M)[0-9]/{n++} END{print n+0}' EXPERIMENT_PROGRESS.md)" -eq 234 && node -e "setTimeout(() => console.log('finding-ledger-verification-passed'), 20)"
  EXPECT: finding-ledger-verification-passed
  EVIDENCE: 2026-09-07 local exit 0; 234/234 Memory IDs, EXPERIMENT_PROGRESS.md ignored by .git/info/exclude; output finding-ledger-verification-passed

- [ ] G2: all executable scientific, mathematical, provenance, and fail-closed tests pass
  EVIDENCE: job 25761 failed closed on two incorrect fixtures (N033); corrected suite has not yet passed on CEDIA.

- [ ] G3: the canonical run.sh completes an end-to-end CEDIA run and validates the final result manifest
  EVIDENCE: run 25587 is historical; run 25746 was intentionally cancelled after N016 was found and is retained as a superseded attempt.

- [ ] G4: every finding is closed with current evidence or explicitly classified as author action, external-data blocked, or manuscript deferred
  EVIDENCE: B09, M23, M175, and N011-N033 are reopened until corrected test and run evidence exists.

- [ ] G5: an independent final audit verifies data lineage, no-future features, nested selection, calibration, Utility, metric identities, and reporting traceability
  EVIDENCE: prior independent audit is historical after N011-N032; repeat after a corrected full run.


## Gates: methodology and results reconstruction

- [ ] G6: every project source, historical result family, Git transition, and current run artifact is classified by role and current validity
  EVIDENCE: pending line-by-line source and reachability audit.

- [ ] G7: the current corrected rerun is independently revalidated and every reported number maps to a hash-verified artifact
  EVIDENCE: pending corrected rerun after N016-N032.

- [ ] G8: the master result table includes every executed current model, calibration, transport, stability, subgroup, temporal, DCA, inference, and ablation result
  EVIDENCE: pending corrected rerun artifact inventory; no manuscript is in scope.

- [ ] G9: Methodology matches the final code and defines every implemented estimand, formula, parameter, edge case, and inferential unit; references are checked against original sources
  EVIDENCE: pending technical code-to-definition audit; no paper text will be drafted.

- [ ] G10: Results use only the final corrected rerun, distinguish non-executed/external-data limitations, and pass two complete traceability readings
  EVIDENCE: pending experimental reporting audit; no paper text will be drafted.

- [ ] G11: every active tracked source file and every cross-file scientific interface has a recorded line-by-line review with zero unrecorded defects
  EVIDENCE: audit in progress; binary raw data and withdrawn historical outputs are checked for isolation and reachability rather than interpreted as source code.
