#!/usr/bin/env python3
"""The only Python command called by the canonical shell entrypoint."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.scientific_pipeline import (
    PipelineError,
    finalize_direct_onset_stage,
    model_direct_onset_stage,
    prepare_direct_onset_stage,
    promote_direct_onset_manifest,
    validate_direct_onset_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one fail-closed stage of the direct-onset experiment.")
    parser.add_argument("--archive", type=Path, default=Path("data/raw/archive.zip"))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--stage", choices=("prepare", "model", "finalize", "promote", "validate"), required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    try:
        if args.stage == "prepare":
            result = prepare_direct_onset_stage(root, args.archive, args.run_dir, args.run_id)
        elif args.stage == "model":
            result = model_direct_onset_stage(root, args.run_dir, args.run_id)
        elif args.stage == "finalize":
            result = finalize_direct_onset_stage(root, args.run_dir, args.run_id)
        elif args.stage == "promote":
            result = promote_direct_onset_manifest(args.run_dir)
        else:
            result = validate_direct_onset_manifest(args.run_dir)
    except PipelineError as exc:
        print(f"FAIL: {exc}")
        return 1
    print(f"PASS: {args.stage}: {result.get('status', result.get('scientific_status', 'completed'))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
