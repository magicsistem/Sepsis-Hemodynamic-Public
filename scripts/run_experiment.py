#!/usr/bin/env python3
"""The only Python command called by the canonical shell entrypoint."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.scientific_pipeline import PipelineError, run_scientific_pipeline, validate_final_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run or validate the corrected scientific experiment.")
    parser.add_argument("--archive", type=Path, default=Path("data/raw/archive.zip"))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    try:
        if args.validate_only:
            result = validate_final_manifest(args.run_dir)
        else:
            result = run_scientific_pipeline(root, args.archive, args.run_dir, args.run_id)
    except PipelineError as exc:
        print(f"FAIL: {exc}")
        return 1
    print(f"PASS: {result['status'] if 'status' in result else result['scientific_status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
