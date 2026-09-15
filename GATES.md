# Gates: corrected scientific pipeline

OWNS: run.sh, jobs/**, scripts/**, src/**, tests/**, docs/**, README.md, MANIFEST.md, .gitignore

Scope: Replace the audited public pipeline with one scientifically valid, fail-closed, reproducible experiment, while retaining every unresolved human or external-data limitation visibly.

- [x] G1: the private exhaustive finding ledger is excluded from Git and has one row for every audit finding
  CHECK: test -f EXPERIMENT_PROGRESS.md && git check-ignore -q EXPERIMENT_PROGRESS.md && test "$(awk '/^\| (B|F|M)[0-9]/{n++} END{print n+0}' EXPERIMENT_PROGRESS.md)" -eq 234 && node -e "setTimeout(() => console.log('finding-ledger-verification-passed'), 20)"
  EXPECT: finding-ledger-verification-passed
  EVIDENCE: 2026-09-07 local exit 0; 234/234 Memory IDs, EXPERIMENT_PROGRESS.md ignored by .git/info/exclude; output finding-ledger-verification-passed

- [ ] G2: all executable scientific, mathematical, provenance, and fail-closed tests pass
  EVIDENCE: job 26221 ran 40 tests and failed only two undeclared two-fold fixtures after the strengthened production fold gate; N053 fixture correction has local compile/diff checks pending one accumulated rerun.

- [ ] G3: the canonical run.sh completes an end-to-end CEDIA run and validates the final result manifest
  EVIDENCE: runs 26122/26124 were cancelled after N037/N038 and 26130 was cancelled at explicit user direction before final source corrections; run 25832 remains superseded by N036.

- [ ] G4: every finding is closed with current evidence or explicitly classified as author action, external-data blocked, or manuscript deferred
  EVIDENCE: N036-N040 are tested; N041-N053 require the corrected accumulated preflight suite, then all affected items require one complete rerun plus independent artifact validation before closure.

- [ ] G5: an independent final audit verifies data lineage, no-future features, nested selection, calibration, Utility, metric identities, and reporting traceability
  EVIDENCE: independent audit of run 25832 verified Utility delegation and downloaded artifact hashes but found N036; repeat the complete audit after the corrected full run.


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

- [x] G11: every active tracked source file and every cross-file scientific interface has a recorded line-by-line review with zero unrecorded scientific defects
  EVIDENCE: docs/LINE_BY_LINE_SCIENTIFIC_AUDIT.md records the complete active-source ranges and cross-file traces through N053; the final pre-compute reread corrected N041-N052 and the accumulated suite exposed/corrected N053 before any replacement full run. Binary raw data and withdrawn historical outputs are checked by identity/isolation rather than treated as active scientific inputs.
