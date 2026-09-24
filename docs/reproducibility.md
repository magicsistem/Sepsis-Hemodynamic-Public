# Reproducibility contract

The final scientific run is launched only with `bash run.sh` from CEDIA. It
requires an exact sidecar-validated copy of a clean laptop commit, the pinned
raw archive, Slurm, and the existing CEDIA container. `PYTHONHASHSEED=20260906`
is exported before Python starts. No package installation or update occurs.

Each run records UTC time, host, Git SHA and dirty state, exact source inventory,
command, archive and ZIP-inventory hashes, schema, feature/model/target/Koopman/
calibration/alarm policy hashes, folds, seeds, dependency and CUDA versions,
validated GPU state, the official Utility source hash, stage resources, and
every scientific artifact hash. The lineage is:

```text
raw archive -> harmonized rows -> causal features and target -> patient folds
-> fold-local representations and nested models/OOF scores
-> nested calibration and alarm thresholds
-> metrics, reliability, event analysis, paired inference, transport and DCA
-> resource evidence -> validated manifest
```

The archive loader accepts source-qualified patient PSV files only. It checks
the official 40-predictor schema, required `Hct`, binary persistent labels,
source identity, exact cohort counts, unique patient/hour rows, archive hash,
and strictly increasing `ICULOS`. It fails closed on any violation.

The primary target is `Y(i,t)=1` exactly when reconstructed true onset is 1--6
hours ahead. Onset/post-onset rows and left-censored onsets are ineligible.
Controls require six complete future hours. The shifted persistent Challenge
label is a separate secondary estimand used only with official Utility.

Features retain the 34 dynamic raw observations only as transform inputs,
causal last observations with a 24-hour maximum age, variable-specific
observation ages and missingness, current measurement count, the five static
predictors, and six observed-sample 8 h CV descriptors. Static predictors are
not transformed. SourceSet is provenance only. SampEn and Shannon remain
independent mathematical oracles but are not production predictors.

C0 is baseline plus CV, C1 is the causal state, C2 adds observed deltas/slopes,
and C3 adds fold-local identity/quadratic ridge EDMD/Koopman innovations plus
8 h innovation-energy summaries. Signal support, median/IQR normalization,
state fill, operator, XGBoost candidate/tree count, calibrator, and alarm
threshold are learned only inside the relevant training partition. The outer
fold never selects a feature policy, model, calibrator, or threshold.

Only a negative-to-positive crossing opens an alarm episode. Persistent
positivity does not rearm after the six-hour refractory period. Useful alarms
fall in `[onset-6h, onset-1h]`; remote false, late pre-onset, post-onset,
right-censored, and left-censored-unclassifiable episodes are separate. False
burden divides by actual eligible decision rows. Thresholds maximize inner-OOF
sensitivity under 0.25 false episodes per eligible patient-day.

Primary discrimination reports patient-balanced AUROC and sklearn Average
Precision. The generic oracle keeps trapezoidal PR-AUC distinct from AP.
Calibration reports Brier, fixed equal-width 10-bin ECE, calibration-in-the-
large, joint intercept/slope, reliability rows, and patient-cluster percentile
intervals. C3-minus-C0 AP/Brier and net-benefit differences use paired patient
cluster bootstraps. DCA evaluates action now for onset in 1--6 hours against
C0, treat-all, and treat-none. No bootstrap sign proportion is called a formal
null test.

C0--C3 are the representation ablation. Prespecified C0/C3 seed and training-
balance sensitivities are stored as one paired wide OOF table (one row identity,
ten probability columns), summarized without selecting a winner, and linked in
the final lineage.

Challenge Utility delegates to the unchanged pinned official scorer. A/B
transport fits every representation, model, calibrator, and threshold in the
source cohort and evaluates once in the destination. It is public-source
transport, not independent external validation.

`run.sh` runs the complete suite, prepares validated data, benchmarks six
feasible CPU/GPU profiles on a fixed 4,000-patient subset with two Koopman fits
and ten 200-tree XGBoost fits (a 5:1 mix versus the planned 278:49 mix),
selects the smallest eligible profile within 5% of the
fastest, then runs independent outer folds, model candidates, and transports
concurrently within the measured per-fit RAM plus 20% and the total job caps.
Before the full model stage, the selected concurrent layout is benchmarked
again with its exact fit/fold/candidate concurrency. The run aborts unless the
active XGBoost phase exceeds 50% of the CPU allocation and remains within the
selected memory request.
Preparation is separately bounded to 16 CPU/32 GB: archive members and
patient-complete feature batches are processed concurrently, then restored to
stable source/patient/time order. Final reporting and its independent C0--C3
revalidation use 4 CPU/32 GB. Both pools set native BLAS/OpenMP threads to one,
so their worker count is the total CPU bound rather than a multiplier. These
non-model stages never request the 64 CPU model ceiling.
It links model/finalize jobs with `afterok`. CPU eligibility requires
more than 50% active compute efficiency. GPU eligibility requires more than
50% mean utilization across at least three active one-second samples and more
than 5% total speedup over the best eligible CPU profile. The bounds are 32
CPU per fit, 64 CPU and 64 GB RAM per job, and one A100 40 GB; RAM is requested from measured/estimated
peak plus 20%. Every stage records measured CPU, RAM, GPU, run ID, commit, and
source inventory. Promotion
independently reconciles those records and all scientific products, rejects
unlisted/tampered artifacts, and never turns successful execution alone into a
scientific PASS.
