# Gates: corrected scientific pipeline

OWNS: run.sh, jobs/**, scripts/**, src/**, tests/**, docs/**, README.md, MANIFEST.md, .gitignore

Scope: Replace the audited public pipeline with one scientifically valid, fail-closed, reproducible experiment, while retaining every unresolved human or external-data limitation visibly.

- [x] G1: the private exhaustive finding ledger is excluded from Git and has one row for every audit finding
  CHECK: test -f EXPERIMENT_PROGRESS.md && git check-ignore -q EXPERIMENT_PROGRESS.md && test "$(awk '/^\| (B|F|M)[0-9]/{n++} END{print n+0}' EXPERIMENT_PROGRESS.md)" -eq 234 && node -e "setTimeout(() => console.log('finding-ledger-verification-passed'), 20)"
  EXPECT: finding-ledger-verification-passed
  EVIDENCE: 2026-09-07 local exit 0; 234/234 Memory IDs, EXPERIMENT_PROGRESS.md ignored by .git/info/exclude; output finding-ledger-verification-passed

- [x] G2: all executable scientific, mathematical, provenance, and fail-closed tests pass
  EVIDENCE: CEDIA job 25745 tested commit 47e38e7 through run.sh on compute-0-2; exit 0. The prior job 25587 evidence is historical after N011-N015.

- [ ] G3: the canonical run.sh completes an end-to-end CEDIA run and validates the final result manifest
  EVIDENCE: run 25587 was a historical PASS superseded by N011-N015; corrected run 25746 is in progress.

- [ ] G4: every finding is closed with current evidence or explicitly classified as author action, external-data blocked, or manuscript deferred
  EVIDENCE: M23 and M175 plus N011-N015 are reopened until corrected run evidence exists.

- [ ] G5: an independent final audit verifies data lineage, no-future features, nested selection, calibration, Utility, metric identities, and reporting traceability
  EVIDENCE: run 25587 independent audit is historical after N011-N015; repeat after corrected run 25746.


## Gates: methodology and results reconstruction

- [ ] G6: every project source, historical result family, Git transition, and current run artifact is classified by role and current validity
  CHECK: manuscript traceability audit has no unclassified evidence item

- [ ] G7: the corrected rerun after N011-N015 is independently revalidated and every reported number maps to a hash-verified artifact
  CHECK: numerical reconstruction and result-manifest validation both exit 0

- [ ] G8: the master result table includes every executed current model, calibration, transport, stability, subgroup, temporal, DCA, inference, and ablation result
  CHECK: manuscript traceability audit has no missing executed branch or metric

- [ ] G9: Methodology matches the final code and defines every implemented estimand, formula, parameter, edge case, and inferential unit; references are checked against original sources
  CHECK: code-to-method and citation audits have no unsupported statement

- [ ] G10: Results use only the final corrected rerun, distinguish non-executed/external-data limitations, and pass two complete traceability readings
  CHECK: manuscript traceability audit exits 0 twice from a clean checkout
