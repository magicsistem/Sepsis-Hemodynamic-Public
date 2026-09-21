# Results policy

Only artifacts emitted under an immutable corrected `runs/<run-id>/` directory
and accepted by its `result_manifest.json` are current scientific outputs.
They are intentionally ignored by Git because they are large, reproducible run
products.

`results/` and `external_artifacts/` contain historical evidence. They are not
accepted as caches, inputs, regression expectations, or sources for reporting.
No report has a hard-coded metric, threshold, feature count, ECE, or p-value.

The experimental report data include OOF predictions with fold provenance,
nested selection records, C0--C3 representation comparisons, metrics,
reliability rows, patient event/alarm rows, DCA, paired DCA inference,
SourceSet transport, prespecified seed/training-balance OOF sensitivities,
inference, resource evidence, and the manifest chain.
Missing or hash-mismatched artifacts cause validation failure, not a partial
statistics package.

`data/raw/archive.zip` is the tracked, required raw-data dependency for this
workflow. Its exact-byte reproduction path is that tracked file plus the
SHA-256 recorded by the run. The official PhysioNet source can provide the
underlying records, but it is not claimed to recreate this local repackaging
byte-for-byte.
