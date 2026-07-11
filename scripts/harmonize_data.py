#!/usr/bin/env python3
"""Run data harmonization with the public default input path."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Harmonize the repository raw-data snapshot derived from PhysioNet/CinC 2019."
    )
    parser.add_argument("--kaggle_path", type=Path, default=Path("data/raw/archive.zip"))
    parser.add_argument("--output_dir", type=Path, default=Path("data/processed"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "src" / "data" / "data_harmonization.py"
    cmd = [
        sys.executable,
        str(script),
        "--kaggle_path",
        str(args.kaggle_path),
        "--output_dir",
        str(args.output_dir),
    ]
    return subprocess.call(cmd, cwd=repo_root)


if __name__ == "__main__":
    raise SystemExit(main())
