#!/usr/bin/env bash
# Sole top-level entrypoint for tests, profiling, and the full experiment.
set -euo pipefail

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$ROOT"
export PYTHONHASHSEED=20260906
export PYTHONWARNINGS=error
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
SOURCE_SIDECAR="$ROOT/.source_provenance.json"

if [[ "${1:-}" == --inside-slurm ]]; then
    STAGE=${2:-}
    [[ -n "${SLURM_JOB_ID:-}" && -n "${RUN_ID:-}" && -n "${RUN_DIR:-}" ]] || { echo "FAIL: allocated execution lacks Slurm/run context" >&2; exit 1; }
    python scripts/source_provenance.py --validate "$SOURCE_SIDECAR"
    case "$STAGE" in
        tests)
            python -m unittest discover -s tests -v
            printf 'TEST_SUITE_PASS run_id=%s\n' "$RUN_ID"
            ;;
        prepare|model|finalize)
            python scripts/run_experiment.py --stage "$STAGE" --archive data/raw/archive.zip --run-id "$RUN_ID" --run-dir "$RUN_DIR"
            ;;
        profile)
            [[ -n "${PROFILE_NAME:-}" ]] || { echo "FAIL: PROFILE_NAME is required" >&2; exit 1; }
            python scripts/profile_resources.py --run-dir "$RUN_DIR" --output "$RUN_DIR/profiles/${PROFILE_NAME}-benchmark.json"
            ;;
        promote)
            : # Aggregation and promotion run after timing in the Slurm wrapper.
            ;;
        validate)
            python scripts/run_experiment.py --stage validate --run-id "$RUN_ID" --run-dir "$RUN_DIR"
            ;;
        *) echo "FAIL: unknown allocated stage: $STAGE" >&2; exit 2;;
    esac
    exit 0
fi

MODE=full
if [[ $# -eq 1 && "$1" == --tests ]]; then
    MODE=tests
elif [[ $# -eq 1 && "$1" == --profile ]]; then
    MODE=profile
elif [[ $# -ne 0 ]]; then
    echo "Usage: bash run.sh [--tests|--profile]" >&2
    exit 2
fi

if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    git diff --quiet && git diff --cached --quiet && [[ -z "$(git status --porcelain)" ]] || {
        echo "FAIL: scientific runs require a clean committed checkout" >&2
        exit 1
    }
    SOURCE_GIT_COMMIT=$(git rev-parse HEAD)
    SOURCE_GIT_DIRTY=false
    python scripts/source_provenance.py --write "$SOURCE_SIDECAR"
else
    [[ -f "$SOURCE_SIDECAR" ]] || { echo "FAIL: missing laptop source provenance sidecar" >&2; exit 1; }
    python scripts/source_provenance.py --validate "$SOURCE_SIDECAR"
    SOURCE_GIT_COMMIT=$(sed -n 's/.*"git_commit"[[:space:]]*:[[:space:]]*"\([0-9a-f]\{40\}\)".*/\1/p' "$SOURCE_SIDECAR")
    SOURCE_GIT_DIRTY=$(sed -n 's/.*"git_dirty"[[:space:]]*:[[:space:]]*\(true\|false\).*/\1/p' "$SOURCE_SIDECAR")
fi
SOURCE_INVENTORY_SHA256=$(sed -n 's/.*"source_inventory_sha256"[[:space:]]*:[[:space:]]*"\([0-9a-f]\{64\}\)".*/\1/p' "$SOURCE_SIDECAR")
[[ ${#SOURCE_INVENTORY_SHA256} -eq 64 ]] || { echo "FAIL: invalid source inventory hash" >&2; exit 1; }
export SOURCE_GIT_COMMIT SOURCE_GIT_DIRTY SOURCE_INVENTORY_SHA256
command -v sbatch >/dev/null || { echo "FAIL: sbatch is required; invoke on CEDIA" >&2; exit 1; }
[[ -f data/raw/archive.zip ]] || { echo "FAIL: data/raw/archive.zip is missing" >&2; exit 1; }

reconcile_ledger() {
    local ledger="$ROOT/logs/run_ledger.tsv" run job commit node state
    [[ -f "$ledger" ]] || return 0
    while IFS=$'\t' read -r run job commit node; do
        grep -q $'\t'"$job"$'\t'.*$'\t'SCHEDULER_ "$ledger" && continue
        state=$(sacct -n -X -j "$job" --format=State --parsable2 2>/dev/null | sed -n '1p' | tr -d '[:space:]')
        case "$state" in
            COMPLETED|FAILED|CANCELLED*|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL|PREEMPTED)
                printf '%s\t%s\t%s\t%s\t%s\tSCHEDULER_%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$run" "$job" "$commit" "$node" "$state" >> "$ledger"
                ;;
        esac
    done < <(awk -F '\t' '$6 ~ /_STARTED$/ {print $2 "\t" $3 "\t" $4 "\t" $5}' "$ledger")
}
reconcile_ledger

mkdir -p runs logs
if [[ -n "${RESUME_RUN_ID:-}" ]]; then
    RUN_ID="$RESUME_RUN_ID"
    RUN_DIR="$ROOT/runs/$RUN_ID"
    [[ -d "$RUN_DIR" ]] || { echo "FAIL: resume run directory not found: $RUN_DIR" >&2; exit 1; }
    RESUMING=true
else
    RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)-${SOURCE_GIT_COMMIT:0:7}}"
    RUN_DIR="$ROOT/runs/$RUN_ID"
    [[ ! -e "$RUN_DIR" ]] || { echo "FAIL: refusing to overwrite $RUN_DIR" >&2; exit 1; }
    RESUMING=false
fi

wait_job() {
    local job=$1 state
    while squeue -h -j "$job" | grep -q .; do sleep 15; done
    state=$(sacct -n -X -j "$job" --format=State --parsable2 | sed -n '1p' | tr -d '[:space:]')
    [[ "$state" == COMPLETED ]] || { echo "FAIL: Slurm job $job ended as ${state:-unknown}" >&2; exit 1; }
}

submit_stage() {
    local stage=$1 cpus=$2 memory=$3 gpus=$4 dependency=${5:-} profile=${6:-} stage_run_dir=${7:-$RUN_DIR}
    local partition=cpu
    if [[ "$gpus" == 1 ]]; then
        [[ "$cpus" -ge 8 ]] || { echo "FAIL: CEDIA gpu QOS requires at least 8 CPUs" >&2; return 1; }
        partition=gpu
    fi
    local args=(--parsable --partition="$partition" --job-name="sepsis_${stage}${profile:+_$profile}" --cpus-per-task="$cpus" --mem="${memory}G")
    [[ "$gpus" == 0 ]] || args+=(--gres=gpu:a100-sxm4-40gb:1)
    [[ -z "$dependency" ]] || args+=(--dependency="afterok:$dependency")
    sbatch "${args[@]}" --export=ALL,PROJECT_DIR="$ROOT",RUN_ID="$RUN_ID",RUN_DIR="$stage_run_dir",PIPELINE_STAGE="$stage",PROFILE_NAME="$profile",REQUESTED_CPUS="$cpus",REQUESTED_MEMORY_GB="$memory",REQUESTED_GPUS="$gpus" jobs/run_experiment.slurm
}

TEST_SUFFIX=$([[ "$RESUMING" == true ]] && date -u +%Y%m%dT%H%M%SZ || printf initial)
TEST_RUN_DIR="$ROOT/runs/${RUN_ID}-tests-${TEST_SUFFIX}"
[[ ! -e "$TEST_RUN_DIR" ]] || { echo "FAIL: refusing to overwrite $TEST_RUN_DIR" >&2; exit 1; }
if [[ "$MODE" == tests ]]; then
    JOB_ID=$(submit_stage tests 2 8 0 "" "" "$TEST_RUN_DIR")
    wait_job "$JOB_ID"
    printf 'TEST_SUITE_PASS run_id=%s job_id=%s\n' "$RUN_ID" "$JOB_ID"
    exit 0
fi

TEST_JOB=$(submit_stage tests 2 8 0 "" "" "$TEST_RUN_DIR")
wait_job "$TEST_JOB"
PREVIOUS_JOB=$TEST_JOB
if [[ "$RESUMING" == true ]] && python scripts/resource_provenance.py verify-stage --run-dir "$RUN_DIR" --stage prepare --git-commit "$SOURCE_GIT_COMMIT" --source-inventory "$SOURCE_INVENTORY_SHA256" >/dev/null; then
    PREPARE_JOB=reused
else
    [[ "$RESUMING" == false ]] || { echo "FAIL: incomplete/invalid prepare stage is preserved; use a fresh run ID" >&2; exit 1; }
    PREPARE_JOB=$(submit_stage prepare 2 24 0 "$PREVIOUS_JOB")
    wait_job "$PREPARE_JOB"
    PREVIOUS_JOB=$PREPARE_JOB
fi
for spec in cpu8:8:0 cpu16:16:0 cpu32:32:0 gpu8:8:1 gpu16:16:1 gpu32:32:1; do
    IFS=: read -r name cpus gpus <<< "$spec"
    if [[ "$RESUMING" == true ]] && python scripts/resource_provenance.py verify-profile --profile-dir "$RUN_DIR/profiles" --name "$name" --cpus "$cpus" --memory-gb 32 --gpus "$gpus" >/dev/null; then
        PROFILE_JOB=reused
    else
        [[ "$RESUMING" == false || ( ! -e "$RUN_DIR/profiles/$name.json" && ! -e "$RUN_DIR/profiles/$name-benchmark.json" ) ]] || { echo "FAIL: invalid partial profile $name is preserved; use a fresh run ID" >&2; exit 1; }
        PROFILE_JOB=$(submit_stage profile "$cpus" 32 "$gpus" "$PREVIOUS_JOB" "$name")
        wait_job "$PROFILE_JOB"
        PREVIOUS_JOB=$PROFILE_JOB
    fi
done
python scripts/resource_provenance.py select-profile --profile-dir "$RUN_DIR/profiles" --output "$RUN_DIR/resource_profile_selection.json"
if [[ "$MODE" == profile ]]; then
    printf 'RESOURCE_PROFILE_PASS run_id=%s selection=%s\n' "$RUN_ID" "$RUN_DIR/resource_profile_selection.json"
    exit 0
fi

read -r MODEL_CPUS MODEL_MEMORY MODEL_GPUS < <(python - "$RUN_DIR/resource_profile_selection.json" <<'PY'
import json, sys
selected = json.load(open(sys.argv[1], encoding="utf-8"))["selected"]
print(selected["cpus"], selected["memory_gb"], selected["gpus"])
PY
)
if [[ "$RESUMING" == true ]] && python scripts/resource_provenance.py verify-stage --run-dir "$RUN_DIR" --stage model --git-commit "$SOURCE_GIT_COMMIT" --source-inventory "$SOURCE_INVENTORY_SHA256" >/dev/null; then
    MODEL_JOB=reused
else
    if [[ "$RESUMING" == true ]] && compgen -G "$RUN_DIR/*_oof_predictions.csv" >/dev/null; then
        echo "FAIL: partial model artifacts are preserved; use a fresh run ID" >&2
        exit 1
    fi
    MODEL_JOB=$(submit_stage model "$MODEL_CPUS" "$MODEL_MEMORY" "$MODEL_GPUS" "$PREVIOUS_JOB")
    wait_job "$MODEL_JOB"
    PREVIOUS_JOB=$MODEL_JOB
fi
if [[ "$RESUMING" == true ]] && python scripts/resource_provenance.py verify-stage --run-dir "$RUN_DIR" --stage finalize --git-commit "$SOURCE_GIT_COMMIT" --source-inventory "$SOURCE_INVENTORY_SHA256" >/dev/null; then
    RESULT_STATUS=$(python -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8")).get("computational_status", "missing"))' "$RUN_DIR/result_manifest.json")
    if [[ "$RESULT_STATUS" == COMPUTATIONAL_RUN_VALIDATED ]]; then
        printf 'SCIENTIFIC_RUN_ALREADY_VALIDATED run_id=%s run_dir=%s\n' "$RUN_ID" "$RUN_DIR"
        exit 0
    fi
    [[ "$RESULT_STATUS" == PENDING_RESOURCE_AND_FINAL_VALIDATION ]] || { echo "FAIL: finalized run has invalid result status $RESULT_STATUS" >&2; exit 1; }
    FINALIZE_JOB=$(submit_stage promote 2 "$MODEL_MEMORY" 0 "$PREVIOUS_JOB")
else
    if [[ "$RESUMING" == true ]] && [[ -e "$RUN_DIR/metrics.json" || -e "$RUN_DIR/result_manifest.json" ]]; then
        echo "FAIL: partial finalize artifacts are preserved; use a fresh run ID" >&2
        exit 1
    fi
    FINALIZE_JOB=$(submit_stage finalize 2 "$MODEL_MEMORY" 0 "$PREVIOUS_JOB")
fi
wait_job "$FINALIZE_JOB"
printf 'SCIENTIFIC_RUN_PASS run_id=%s test_job=%s prepare_job=%s model_job=%s finalize_job=%s run_dir=%s\n' \
    "$RUN_ID" "$TEST_JOB" "$PREPARE_JOB" "$MODEL_JOB" "$FINALIZE_JOB" "$RUN_DIR"
