#!/usr/bin/env bash
# Canonical top-level entrypoint.  The login path submits one Slurm job; the
# allocated path runs every scientific stage and refuses partial promotion.
set -euo pipefail

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$ROOT"
export PYTHONHASHSEED=20260906
export PYTHONWARNINGS=error
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
SOURCE_SIDECAR="$ROOT/.source_provenance.json"

if [[ "${1:-}" == "--inside-slurm" ]]; then
    [[ -n "${SLURM_JOB_ID:-}" ]] || { echo "FAIL: --inside-slurm requires Slurm" >&2; exit 1; }
    [[ -n "${RUN_ID:-}" ]] || { echo "FAIL: RUN_ID is required" >&2; exit 1; }
    [[ -n "${RUN_DIR:-}" ]] || { echo "FAIL: RUN_DIR is required" >&2; exit 1; }
    python scripts/source_provenance.py --validate "$SOURCE_SIDECAR"
    python -m unittest discover -s tests -v
    if [[ "${RUN_TESTS_ONLY:-false}" == true ]]; then
        printf 'TEST_SUITE_PASS run_id=%s\n' "$RUN_ID"
        exit 0
    fi
    python scripts/run_experiment.py --archive data/raw/archive.zip --run-id "$RUN_ID" --run-dir "$RUN_DIR"
    python scripts/run_experiment.py --validate-only --run-id "$RUN_ID" --run-dir "$RUN_DIR"
    printf 'SCIENTIFIC_RUN_PASS run_id=%s run_dir=%s\n' "$RUN_ID" "$RUN_DIR"
    exit 0
fi

TESTS_ONLY=false
if [[ $# -eq 1 && "${1:-}" == "--tests" ]]; then
    TESTS_ONLY=true
elif [[ $# -ne 0 ]]; then
    echo "Usage: bash run.sh [--tests]" >&2
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
export SOURCE_GIT_COMMIT SOURCE_GIT_DIRTY
command -v sbatch >/dev/null || { echo "FAIL: sbatch is required; invoke on CEDIA" >&2; exit 1; }
[[ -f data/raw/archive.zip ]] || { echo "FAIL: data/raw/archive.zip is missing" >&2; exit 1; }

if [[ -n "${RESUME_RUN_ID:-}" ]]; then
    RUN_ID="$RESUME_RUN_ID"
    RUN_DIR="$ROOT/runs/$RUN_ID"
    [[ -d "$RUN_DIR" ]] || { echo "FAIL: resume run directory not found: $RUN_DIR" >&2; exit 1; }
    export RESUME_EXISTING=true
else
    RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)-${SOURCE_GIT_COMMIT:0:7}}"
    RUN_DIR="$ROOT/runs/$RUN_ID"
    [[ ! -e "$RUN_DIR" ]] || { echo "FAIL: refusing to overwrite $RUN_DIR" >&2; exit 1; }
    export RESUME_EXISTING=false
fi
mkdir -p runs logs
JOB_ID=$(sbatch --parsable --export=ALL,PROJECT_DIR="$ROOT",RUN_ID="$RUN_ID",RUN_DIR="$RUN_DIR",RUN_TESTS_ONLY="$TESTS_ONLY" jobs/run_experiment.slurm)
printf 'Submitted scientific run %s (Slurm job %s)\n' "$RUN_ID" "$JOB_ID"

while squeue -h -j "$JOB_ID" | grep -q .; do
    sleep 15
done
STATE=$(sacct -n -X -j "$JOB_ID" --format=State --parsable2 | sed -n '1p' | tr -d '[:space:]')
[[ "$STATE" == COMPLETED ]] || { echo "FAIL: Slurm job $JOB_ID ended as ${STATE:-unknown}; inspect sepsis_scientific_v2-${JOB_ID}.out and .err" >&2; exit 1; }
if [[ "$TESTS_ONLY" == true ]]; then
    printf 'TEST_SUITE_PASS run_id=%s job_id=%s\n' "$RUN_ID" "$JOB_ID"
else
    printf 'SCIENTIFIC_RUN_PASS run_id=%s job_id=%s run_dir=%s\n' "$RUN_ID" "$JOB_ID" "$RUN_DIR"
fi
