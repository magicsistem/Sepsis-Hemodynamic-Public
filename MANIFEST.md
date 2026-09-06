# Active manifest

| Path | Role |
|---|---|
| `run.sh` | Sole CEDIA entrypoint and Slurm supervisor. |
| `jobs/run_experiment.slurm` | Single parameterized allocated execution job. |
| `scripts/run_experiment.py` | Runs or validates the current immutable experiment directory. |
| `src/scientific_pipeline.py` | Schema gate, chronology, features, nested validation, metrics, inference, reporting data, and manifests. |
| `vendor/physionet2019/` | Pinned unmodified official BSD-2 Challenge scorer; sole Utility implementation. |
| `tests/test_scientific_pipeline.py` | Mathematical, provenance, chronology, and fail-closed oracle suite. |
| `docs/reproducibility.md` | Definitions and reproducibility contract. |

No legacy training, reporting, cache, or Slurm entrypoint is active. Historical
result directories are not inputs to the active pipeline.
