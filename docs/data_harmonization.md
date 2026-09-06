# Data contract

`data/raw/archive.zip` must contain official patient PSV files at
`training_setA/training/p*.psv` or `training_setB/training_setB/p*.psv`. The loader
does not inspect or fall back to aggregate `Dataset.csv`, CSV, or TSV files.

Each PSV must contain exactly the official 40 predictors and `SepsisLabel`.
Only explicit `HCT`/`Hematocrit` aliases are normalized to the required
canonical `Hct`; unrecognized fields, duplicate normalized fields, or missing
fields abort the run. Patient identity is source-qualified (`A:p000001`) to
prevent collisions between source sets.

The run manifest records archive SHA-256, ZIP inventory hash, file count, row
count, patient count, and per-source patient counts. These properties are
validated before feature construction.
