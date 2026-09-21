# Active manifest

| Path | Role |
|---|---|
| `run.sh` | Sole CEDIA entrypoint and Slurm supervisor. |
| `jobs/run_experiment.slurm` | Single parameterized allocated execution job. |
| `scripts/run_experiment.py` | Runs or validates the current immutable experiment directory. |
| `scripts/profile_resources.py` | Fixed-subset CPU/GPU benchmark. |
| `scripts/resource_provenance.py` | Resource measurement, selection, and source-context gates. |
| `scripts/source_provenance.py` | Clean laptop commit and per-file source sidecar. |
| `src/onset_koopman.py` | Primary target, causal EDMD/Koopman transforms, calibration, alarms, inference, and DCA. |
| `src/scientific_pipeline.py` | Schema, chronology, features, nested training, transport, reporting products, lineage, and final validation. |
| `vendor/physionet2019/` | Pinned unmodified official BSD-2 Challenge scorer; sole Utility implementation. |
| `tests/test_scientific_pipeline.py` | Mathematical, provenance, chronology, and fail-closed oracle suite. |
| `tests/test_onset_koopman.py` | Direct-onset, Koopman, nested-policy, event, DCA, transport, and resource oracles. |
| `docs/reproducibility.md` | Definitions and reproducibility contract. |

No legacy training, reporting, cache, or Slurm entrypoint is active. Historical
result directories are not inputs to the active pipeline.
