#!/bin/bash
#SBATCH --job-name=methodology_audit
#SBATCH --partition=cpu-dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/home/miguel.benavides__yachaytech.edu.ec/Sepsis-Hemodynamic/logs/methodology_audit_%j.out
#SBATCH --error=/home/miguel.benavides__yachaytech.edu.ec/Sepsis-Hemodynamic/logs/methodology_audit_%j.err

set -euo pipefail

PROJECT_DIR=/home/miguel.benavides__yachaytech.edu.ec/Sepsis-Hemodynamic
CONTAINER=${METHODOLOGY_CONTAINER:-/home/miguel.benavides__yachaytech.edu.ec/pytorch_24.01-py3.sif}

mkdir -p "$PROJECT_DIR/logs"

echo "[$(date -Is)] Starting no-leakage/methodology audit"
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

  python -m py_compile scripts/audit_methodology.py

  python scripts/audit_methodology.py \
    --baseline-dir results/baseline \
    --enhanced-dir results/enhanced \
    --statistics-dir results/statistics \
    --output-dir external_artifacts/inventory
"

echo "[$(date -Is)] No-leakage/methodology audit complete"
