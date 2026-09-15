# Reproducibility contract

The final scientific run is launched only with `bash run.sh` from CEDIA. It
requires an exact sidecar-validated copy of a clean laptop commit, the pinned
raw archive, Slurm, and the existing CEDIA container. `PYTHONHASHSEED=20260906`
is exported before Python starts. No package installation or update occurs.

Each run records UTC time, host, Git SHA and dirty state, the exact source
inventory hash, command, archive and input-inventory hashes, schema, feature
and model policy hashes, fold hash, seed,
dependency versions, GPU runtime validation, official Utility source hash, and
hashes for every scientific artifact. The lineage is:

```text
raw archive -> harmonized PSV rows -> causal features -> patient folds
-> nested models/OOF scores -> nested outcome-specific sigmoid calibration/thresholds
-> metrics, event analysis, inference, transport, ablations, DCA -> manifest
```

The archive loader accepts source-qualified patient PSV files only. It checks
the official schema, required `Hct`, binary/persistent labels, source identity,
unique patient/hour rows, and strictly increasing `ICULOS`. It fails closed on
any violation.

Features use raw observations for rolling variability and entropy; last
observation has a 24-hour maximum age. SampEn is canonical `m=2`, `r=0.2 SD`;
the `m` and `m+1` counts use the same `N-m` starting positions, fewer than four
observations are undefined, and a zero `(m+1)` match is recorded explicitly.
Shannon entropy uses non-negative count probabilities. Static
predictors are not transformed. SourceSet is provenance only, never a model
feature.

The Challenge Utility evaluator is the unchanged official scorer, pinned in
`vendor/physionet2019`. It evaluates the shifted persistent Challenge labels;
it is never labelled a fixed-horizon outcome. Fixed early-warning events use
the reconstructed onset and the policy-fixed useful window `[onset-12h,
onset-1h]`. Only a negative-to-positive crossing opens an alarm episode;
continuous persistence cannot become a later useful warning after the six-hour
refractory interval. Alarm rates divide by actual observed decision rows, not
the elapsed ICULOS span; useful, remote-false, late-pre-onset, post-onset, and
left-censored-unclassifiable episodes remain separate categories.

Reported discrimination includes AUROC, sklearn Average Precision, and
trapezoidal PR-AUC as distinct estimands. Row-time calibration includes Brier,
fixed equal-width 10-bin ECE, calibration-in-the-large with slope fixed at one,
a joint intercept/slope fit, reliability rows, and patient-cluster bootstrap
uncertainty. DCA evaluates assessment now for true reconstructed onset in the
next six hours at observed pre-onset decision hours. Its probability is a
separate sigmoid fit on eligible inner-OOF six-hour outcomes, then applied to
the held-out outer fold; DCA uncertainty uses a patient-cluster bootstrap.
Cluster-respecting paired patient permutation tests define the
AUROC/AP/Brier family and use canonical Benjamini-Hochberg reverse cumulative
minima; no bootstrap sign proportion is presented as a null test.

The hashed model policy records five outer folds, three inner folds, the two
XGBoost candidates, Average Precision selection, the 600-tree early-stopping
ceiling, 30-round patience, eight threads, and both unpenalized `lbfgs`
sigmoid calibrators, plus the exact L2-logistic robustness settings. These are
run configuration, not historical expected results.
