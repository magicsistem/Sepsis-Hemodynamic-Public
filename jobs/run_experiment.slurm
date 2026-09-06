#!/usr/bin/env bash
#SBATCH --job-name=sepsis_scientific_v2
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:a100-sxm4-40gb:1
#SBATCH --time=2-00:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail
: "${PROJECT_DIR:?PROJECT_DIR must be supplied by run.sh}"
: "${RUN_ID:?RUN_ID must be supplied by run.sh}"
: "${RUN_DIR:?RUN_DIR must be supplied by run.sh}"
CONTAINER=${SEPSIS_CONTAINER:-/home/miguel.benavides__yachaytech.edu.ec/pytorch_24.01-py3.sif}
[[ -f "$CONTAINER" ]] || { echo "FAIL: container not found: $CONTAINER" >&2; exit 1; }
mkdir -p "$PROJECT_DIR/logs"
cd "$PROJECT_DIR"

if command -v singularity >/dev/null 2>&1; then
    RUNTIME=singularity
elif command -v apptainer >/dev/null 2>&1; then
    RUNTIME=apptainer
else
    echo "FAIL: no Singularity/Apptainer runtime" >&2
    exit 1
fi

"$RUNTIME" exec --nv "$CONTAINER" env \
    PYTHONHASHSEED=20260906 \
    PROJECT_DIR="$PROJECT_DIR" \
    RUN_ID="$RUN_ID" \
    RUN_DIR="$RUN_DIR" \
    RUN_TESTS_ONLY="${RUN_TESTS_ONLY:-false}" \
    SOURCE_GIT_COMMIT="$SOURCE_GIT_COMMIT" \
    SOURCE_GIT_DIRTY="$SOURCE_GIT_DIRTY" \
    bash run.sh --inside-slurm
