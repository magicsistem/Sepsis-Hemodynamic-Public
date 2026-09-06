#!/usr/bin/env bash
# Canonical top-level entrypoint.  The login path submits one Slurm job; the
# allocated path runs every scientific stage and refuses partial promotion.
set -euo pipefail

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$ROOT"
export PYTHONHASHSEED=20260906
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

if [[ "${1:-}" == "--inside-slurm" ]]; then
    [[ -n "${SLURM_JOB_ID:-}" ]] || { echo "FAIL: --inside-slurm requires Slurm" >&2; exit 1; }
    [[ -n "${RUN_ID:-}" ]] || { echo "FAIL: RUN_ID is required" >&2; exit 1; }
    [[ -n "${RUN_DIR:-}" ]] || { echo "FAIL: RUN_DIR is required" >&2; exit 1; }
    python -m unittest discover -s tests -v
    python scripts/run_experiment.py --archive data/raw/archive.zip --run-id "$RUN_ID" --run-dir "$RUN_DIR"
    python scripts/run_experiment.py --validate-only --run-id "$RUN_ID" --run-dir "$RUN_DIR"
    printf 'SCIENTIFIC_RUN_PASS run_id=%s run_dir=%s\n' "$RUN_ID" "$RUN_DIR"
    exit 0
fi

[[ $# -eq 0 ]] || { echo "Usage: bash run.sh" >&2; exit 2; }
git diff --quiet && git diff --cached --quiet && [[ -z "$(git status --porcelain)" ]] || {
    echo "FAIL: scientific runs require a clean committed checkout" >&2
    exit 1
}
command -v sbatch >/dev/null || { echo "FAIL: sbatch is required; invoke on CEDIA" >&2; exit 1; }
[[ -f data/raw/archive.zip ]] || { echo "FAIL: data/raw/archive.zip is missing" >&2; exit 1; }

RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)-$(git rev-parse --short HEAD)}"
RUN_DIR="$ROOT/runs/$RUN_ID"
[[ ! -e "$RUN_DIR" ]] || { echo "FAIL: refusing to overwrite $RUN_DIR" >&2; exit 1; }
mkdir -p runs logs
JOB_ID=$(sbatch --parsable --export=ALL,PROJECT_DIR="$ROOT",RUN_ID="$RUN_ID",RUN_DIR="$RUN_DIR" jobs/run_experiment.slurm)
printf 'Submitted scientific run %s (Slurm job %s)\n' "$RUN_ID" "$JOB_ID"

while squeue -h -j "$JOB_ID" | grep -q .; do
    sleep 15
done
STATE=$(sacct -n -X -j "$JOB_ID" --format=State --parsable2 | sed -n '1p' | tr -d '[:space:]')
[[ "$STATE" == COMPLETED ]] || { echo "FAIL: Slurm job $JOB_ID ended as ${STATE:-unknown}; inspect logs/run-${JOB_ID}.out and .err" >&2; exit 1; }
printf 'SCIENTIFIC_RUN_PASS run_id=%s job_id=%s run_dir=%s\n' "$RUN_ID" "$JOB_ID" "$RUN_DIR"
