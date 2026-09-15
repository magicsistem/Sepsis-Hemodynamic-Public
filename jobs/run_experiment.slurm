#!/usr/bin/env bash
#SBATCH --job-name=sepsis_scientific_v2
#SBATCH --partition=gpu
#SBATCH --nodelist=compute-0-2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100-sxm4-40gb:1
#SBATCH --time=2-00:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail
: "${PROJECT_DIR:?PROJECT_DIR must be supplied by run.sh}"
: "${RUN_ID:?RUN_ID must be supplied by run.sh}"
: "${RUN_DIR:?RUN_DIR must be supplied by run.sh}"
CONTAINER=${SEPSIS_CONTAINER:-/home/miguel.benavides__yachaytech.edu.ec/pytorch_24.01-py3.sif}
mkdir -p "$PROJECT_DIR/logs"
LEDGER="$PROJECT_DIR/logs/run_ledger.tsv"
ledger_event() { printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$RUN_ID" "${SLURM_JOB_ID:-unknown}" "$SOURCE_GIT_COMMIT" "$(hostname)" "$1" >> "$LEDGER"; }
ledger_event STARTED
trap 'ledger_event CANCELLED_signal_TERM; trap - EXIT; exit 143' TERM INT
trap 'status=$?; if [[ $status -eq 0 ]]; then ledger_event COMPLETED; else ledger_event "FAILED_exit_${status}"; fi' EXIT
[[ -f "$CONTAINER" ]] || { echo "FAIL: container not found: $CONTAINER" >&2; exit 1; }
CONTAINER_SHA256=$(sha256sum "$CONTAINER" | awk '{print $1}')
cd "$PROJECT_DIR"

if command -v singularity >/dev/null 2>&1; then
    RUNTIME=singularity
elif command -v apptainer >/dev/null 2>&1; then
    RUNTIME=apptainer
else
    echo "FAIL: no Singularity/Apptainer runtime" >&2
    exit 1
fi

printf 'CEDIA_ENVIRONMENT_BEGIN\n'
hostname
uname -a
if command -v python >/dev/null 2>&1; then python --version; else echo "HOST_PYTHON_UNAVAILABLE"; fi
nvidia-smi
command -v python || true
command -v sbatch
command -v squeue
git --version
"$RUNTIME" --version
printf 'CEDIA_ENVIRONMENT_END\n'

"$RUNTIME" exec --nv "$CONTAINER" env \
    PYTHONHASHSEED=20260906 \
    PROJECT_DIR="$PROJECT_DIR" \
    RUN_ID="$RUN_ID" \
    RUN_DIR="$RUN_DIR" \
    RUN_TESTS_ONLY="${RUN_TESTS_ONLY:-false}" \
    REQUIRE_GPU=true \
    SOURCE_GIT_COMMIT="$SOURCE_GIT_COMMIT" \
    SOURCE_GIT_DIRTY="$SOURCE_GIT_DIRTY" \
    SOURCE_INVENTORY_SHA256="$SOURCE_INVENTORY_SHA256" \
    SOURCE_CONTAINER_PATH="$CONTAINER" \
    SOURCE_CONTAINER_SHA256="$CONTAINER_SHA256" \
    SOURCE_CONTAINER_RUNTIME="$RUNTIME" \
    bash run.sh --inside-slurm
