#!/usr/bin/env python3
"""Run the enhanced hemodynamic training workflow with public default paths."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run enhanced hemodynamic training.")
    parser.add_argument("--data_dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output_dir", type=Path, default=Path("results/enhanced"))
    parser.add_argument("--n_jobs", type=int, default=64)
    parser.add_argument("--n_gpus", type=int, default=2)
    parser.add_argument("--stage", choices=["features", "train", "all"], default="all")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "src" / "training" / "train_zabihi_cudf.py"
    cmd = [
        sys.executable,
        str(script),
        "--data_dir",
        str(args.data_dir),
        "--output_dir",
        str(args.output_dir),
        "--n_jobs",
        str(args.n_jobs),
        "--n_gpus",
        str(args.n_gpus),
        "--use_hemo",
        "--stage",
        args.stage,
    ]
    return subprocess.call(cmd, cwd=repo_root)


if __name__ == "__main__":
    raise SystemExit(main())
