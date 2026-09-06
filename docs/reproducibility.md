# Reproducibility contract

The final scientific run is launched only with `bash run.sh` from CEDIA. It
requires a clean committed checkout, the pinned raw archive, Slurm, and the
existing CEDIA container. `PYTHONHASHSEED=20260906` is exported before Python
starts. No package installation or update occurs.

Each run records UTC time, host, Git SHA and dirty state, command, archive and
input-inventory hashes, schema and feature policy hashes, fold hash, seed,
dependency versions, GPU runtime validation, official Utility source hash, and
hashes for every scientific artifact. The lineage is:

```text
raw archive -> harmonized PSV rows -> causal features -> patient folds
-> nested models/OOF scores -> nested Platt calibration/thresholds
-> metrics, event analysis, inference, transport, ablations, DCA -> manifest
```

The archive loader accepts source-qualified patient PSV files only. It checks
the official schema, required `Hct`, binary/persistent labels, source identity,
unique patient/hour rows, and strictly increasing `ICULOS`. It fails closed on
any violation.

Features use raw observations for rolling variability and entropy; last
observation has a 24-hour maximum age. SampEn is canonical `m=2`, `r=0.2 SD`,
with fewer than four observations undefined and a zero `(m+1)` match recorded
explicitly. Shannon entropy uses non-negative count probabilities. Static
predictors are not transformed. SourceSet is provenance only, never a model
feature.

The Challenge Utility evaluator is the unchanged official scorer, pinned in
`vendor/physionet2019`. It evaluates the shifted persistent Challenge labels;
it is never labelled a fixed-horizon outcome. Fixed early-warning events use
the reconstructed onset and the pre-specified useful window `[onset-12h,
onset-1h]` with a six-hour refractory alarm policy.

Reported discrimination includes AUROC, sklearn Average Precision, and
trapezoidal PR-AUC as distinct estimands. Calibration includes Brier, fixed
10-bin ECE, intercept, slope, and reliability rows. Cluster-respecting paired
patient permutation tests predefine the AUROC/AP/Brier family; no bootstrap
sign proportion is presented as a null test.
