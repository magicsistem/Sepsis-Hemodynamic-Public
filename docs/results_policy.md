# Results policy

Only artifacts emitted under an immutable corrected `runs/<run-id>/` directory
and accepted by its `result_manifest.json` are current scientific outputs.
They are intentionally ignored by Git because they are large, reproducible run
products.

`results/` and `external_artifacts/` contain historical evidence. They are not
accepted as caches, inputs, regression expectations, or sources for reporting.
No report has a hard-coded metric, threshold, feature count, ECE, or p-value.

The experimental report data include OOF predictions with fold provenance,
nested selection records, metrics, reliability rows, patient event/alarm rows,
DCA, SourceSet transport, ablations, inference, and the manifest chain.
Missing or hash-mismatched artifacts cause validation failure, not a partial
statistics package.
