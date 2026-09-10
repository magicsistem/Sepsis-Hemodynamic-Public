# Data contract

`data/raw/archive.zip` must contain official patient PSV files at
`training_setA/training/p*.psv` or `training_setB/training_setB/p*.psv`. The loader
does not inspect or fall back to aggregate `Dataset.csv`, CSV, or TSV files.

Each PSV must contain exactly the official 40 predictors and `SepsisLabel`.
Only explicit `HCT`/`Hematocrit` aliases are normalized to the required
canonical `Hct`; unrecognized fields, duplicate normalized fields, or missing
fields abort the run. Patient identity is source-qualified (`A:p000001`) to
prevent collisions between source sets.

The accepted archive SHA-256 is
`1a0eb8040c76fdab84ee6c7dd6afdab4ad457a33d363cb7e4e200af713345897`.
Before feature construction the loader requires exactly 40,336 PSV files,
40,336 source-qualified patients (A: 20,336; B: 20,000), and 1,552,210 rows.
The run manifest also records the ZIP inventory hash. Observed nonnumeric or
infinite values are rejected; only genuine missing observations remain missing.
