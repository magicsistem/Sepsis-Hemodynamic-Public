#!/usr/bin/env bash
#SBATCH --job-name=sepsis_direct_onset
#SBATCH --nodelist=compute-0-2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=2-00:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail
: "${PROJECT_DIR:?PROJECT_DIR must be supplied by run.sh}"
: "${RUN_ID:?RUN_ID must be supplied by run.sh}"
: "${RUN_DIR:?RUN_DIR must be supplied by run.sh}"
: "${PIPELINE_STAGE:?PIPELINE_STAGE must be supplied by run.sh}"
: "${REQUESTED_CPUS:?REQUESTED_CPUS must be supplied by run.sh}"
: "${REQUESTED_MEMORY_GB:?REQUESTED_MEMORY_GB must be supplied by run.sh}"
: "${REQUESTED_GPUS:?REQUESTED_GPUS must be supplied by run.sh}"
CONTAINER=${SEPSIS_CONTAINER:-/home/miguel.benavides__yachaytech.edu.ec/pytorch_24.01-py3.sif}
NODE_NAME=$(hostname -s)
[[ "$NODE_NAME" == compute-0-2 ]] || { echo "FAIL: scientific work is restricted to compute-0-2" >&2; exit 1; }
[[ -f "$CONTAINER" ]] || { echo "FAIL: container not found: $CONTAINER" >&2; exit 1; }
mkdir -p "$PROJECT_DIR/logs" "$RUN_DIR/resources" "$RUN_DIR/profiles"
LEDGER="$PROJECT_DIR/logs/run_ledger.tsv"
ledger_event() { printf '%s\t%s\t%s\t%s\t%s\t%s_%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$RUN_ID" "${SLURM_JOB_ID:-unknown}" "$SOURCE_GIT_COMMIT" "$NODE_NAME" "$PIPELINE_STAGE" "$1" >> "$LEDGER"; }
ledger_event STARTED
GPU_SAMPLER_PID=
cleanup_sampler() { [[ -z "$GPU_SAMPLER_PID" ]] || kill "$GPU_SAMPLER_PID" 2>/dev/null || true; }
trap 'cleanup_sampler; ledger_event CANCELLED_signal_TERM; trap - EXIT; exit 143' TERM INT
trap 'status=$?; cleanup_sampler; if [[ $status -eq 0 ]]; then ledger_event COMPLETED; else ledger_event "FAILED_exit_${status}"; fi' EXIT

if command -v singularity >/dev/null 2>&1; then
    RUNTIME=singularity
elif command -v apptainer >/dev/null 2>&1; then
    RUNTIME=apptainer
else
    echo "FAIL: no Singularity/Apptainer runtime" >&2
    exit 1
fi
CONTAINER_SHA256=$(sha256sum "$CONTAINER" | awk '{print $1}')
cd "$PROJECT_DIR"
echo CEDIA_ENVIRONMENT_BEGIN
hostname
uname -a
python --version
which python
which sbatch
which squeue
git --version
if [[ "$REQUESTED_GPUS" == 1 ]]; then nvidia-smi; fi
echo CEDIA_ENVIRONMENT_END
TIME_FILE="$RUN_DIR/resources/${PIPELINE_STAGE}-${SLURM_JOB_ID}.time"
GPU_LOG="$RUN_DIR/resources/${PIPELINE_STAGE}-${SLURM_JOB_ID}.gpu.csv"
if [[ "$REQUESTED_GPUS" == 1 ]]; then
    (while true; do nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits; sleep 15; done) > "$GPU_LOG" &
    GPU_SAMPLER_PID=$!
    CONTAINER_GPU_ARGS=(--nv)
    REQUIRE_GPU=true
else
    : > "$GPU_LOG"
    CONTAINER_GPU_ARGS=()
    REQUIRE_GPU=false
fi

export OMP_NUM_THREADS="$REQUESTED_CPUS"
export MKL_NUM_THREADS="$REQUESTED_CPUS"
export OPENBLAS_NUM_THREADS="$REQUESTED_CPUS"
export NUMEXPR_NUM_THREADS="$REQUESTED_CPUS"
set +e
/usr/bin/time -f 'elapsed_seconds=%e\nuser_seconds=%U\nsystem_seconds=%S\nmax_rss_kb=%M\nexit_status=%x' -o "$TIME_FILE" \
    "$RUNTIME" exec "${CONTAINER_GPU_ARGS[@]}" "$CONTAINER" env \
        PYTHONHASHSEED=20260906 PYTHONWARNINGS=error PYTHONPATH="$PROJECT_DIR" \
        PROJECT_DIR="$PROJECT_DIR" RUN_ID="$RUN_ID" RUN_DIR="$RUN_DIR" \
        PIPELINE_STAGE="$PIPELINE_STAGE" PROFILE_NAME="${PROFILE_NAME:-}" \
        REQUIRE_GPU="$REQUIRE_GPU" SLURM_CPUS_PER_TASK="$REQUESTED_CPUS" \
        OMP_NUM_THREADS="$REQUESTED_CPUS" MKL_NUM_THREADS="$REQUESTED_CPUS" \
        OPENBLAS_NUM_THREADS="$REQUESTED_CPUS" NUMEXPR_NUM_THREADS="$REQUESTED_CPUS" \
        SOURCE_GIT_COMMIT="$SOURCE_GIT_COMMIT" SOURCE_GIT_DIRTY="$SOURCE_GIT_DIRTY" \
        SOURCE_INVENTORY_SHA256="$SOURCE_INVENTORY_SHA256" \
        SOURCE_CONTAINER_PATH="$CONTAINER" SOURCE_CONTAINER_SHA256="$CONTAINER_SHA256" \
        SOURCE_CONTAINER_RUNTIME="$RUNTIME" bash run.sh --inside-slurm "$PIPELINE_STAGE"
STAGE_STATUS=$?
set -e
cleanup_sampler
GPU_SAMPLER_PID=

if [[ "$PIPELINE_STAGE" == profile ]]; then
    RESOURCE_OUTPUT="$RUN_DIR/profiles/${PROFILE_NAME}.json"
    BENCHMARK_ARGS=(--benchmark-file "$RUN_DIR/profiles/${PROFILE_NAME}-benchmark.json")
else
    RESOURCE_OUTPUT="$RUN_DIR/resources/${PIPELINE_STAGE}.json"
    BENCHMARK_ARGS=()
fi
"$RUNTIME" exec "${CONTAINER_GPU_ARGS[@]}" "$CONTAINER" python scripts/resource_provenance.py stage \
    --stage "${PROFILE_NAME:-$PIPELINE_STAGE}" --time-file "$TIME_FILE" --gpu-log "$GPU_LOG" \
    "${BENCHMARK_ARGS[@]}" --output "$RESOURCE_OUTPUT" --cpus "$REQUESTED_CPUS" \
    --memory-gb "$REQUESTED_MEMORY_GB" --gpus "$REQUESTED_GPUS" \
    --hostname "$NODE_NAME" --job-id "$SLURM_JOB_ID" --partition "$SLURM_JOB_PARTITION"
[[ "$STAGE_STATUS" -eq 0 ]] || exit "$STAGE_STATUS"

if [[ "$PIPELINE_STAGE" == finalize || "$PIPELINE_STAGE" == promote ]]; then
    "$RUNTIME" exec "$CONTAINER" python scripts/resource_provenance.py aggregate \
        --resources-dir "$RUN_DIR/resources" --profile-selection "$RUN_DIR/resource_profile_selection.json" \
        --output "$RUN_DIR/resource_manifest.json"
    "$RUNTIME" exec "$CONTAINER" env PYTHONHASHSEED=20260906 PYTHONWARNINGS=error PYTHONPATH="$PROJECT_DIR" \
        python scripts/run_experiment.py --stage promote --run-id "$RUN_ID" --run-dir "$RUN_DIR"
fi
