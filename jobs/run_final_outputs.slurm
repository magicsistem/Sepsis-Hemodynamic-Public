#!/bin/bash
#SBATCH --job-name=final_outputs_assets
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/home/miguel.benavides__yachaytech.edu.ec/Sepsis-Hemodynamic/logs/final_outputs_assets_%j.out
#SBATCH --error=/home/miguel.benavides__yachaytech.edu.ec/Sepsis-Hemodynamic/logs/final_outputs_assets_%j.err

set -euo pipefail

PROJECT_DIR=/home/miguel.benavides__yachaytech.edu.ec/Sepsis-Hemodynamic
CONTAINER=${FINAL_OUTPUTS_CONTAINER:-/home/miguel.benavides__yachaytech.edu.ec/pytorch_24.01-py3.sif}

mkdir -p "$PROJECT_DIR/logs"

echo "[$(date -Is)] Starting final public table/figure generation"
echo "hostname=$(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-NA}"
echo "PROJECT_DIR=$PROJECT_DIR"
echo "CONTAINER=$CONTAINER"

cd "$PROJECT_DIR"

if [ ! -f "$CONTAINER" ]; then
  echo "ERROR: container not found: $CONTAINER"
  exit 1
fi

if command -v singularity >/dev/null 2>&1; then
  CONTAINER_RUNTIME="singularity"
elif command -v apptainer >/dev/null 2>&1; then
  CONTAINER_RUNTIME="apptainer"
else
  echo "ERROR: neither singularity nor apptainer is available on this node"
  exit 1
fi

echo "Using container runtime: $CONTAINER_RUNTIME"

"$CONTAINER_RUNTIME" exec "$CONTAINER" bash -lc "
  set -euo pipefail
  cd '$PROJECT_DIR'

  echo 'Container python:'
  which python
  python --version

  python - <<'PY'
import sys
print('sys.version:', sys.version)
try:
    import pandas as pd
    import numpy as np
    import sklearn
    import matplotlib
    print('pandas:', pd.__version__)
    print('numpy:', np.__version__)
    print('sklearn:', sklearn.__version__)
    print('matplotlib:', matplotlib.__version__)
except Exception as e:
    print('IMPORT_ERROR:', repr(e))
    raise
PY

  python -m py_compile src/reporting/build_final_outputs.py scripts/build_final_outputs.py

  python scripts/build_final_outputs.py \
    --statistics-dir results/statistics \
    --internal-robustness-dir results/internal_robustness \
    --baseline-dir results/baseline \
    --enhanced-dir results/enhanced \
    --tables-dir results/final_tables \
    --figures-dir results/final_figures
"

echo "[$(date -Is)] Final public table/figure generation complete"
