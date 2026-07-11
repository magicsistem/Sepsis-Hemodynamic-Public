#!/usr/bin/env python3
"""Run compact statistics generation with public result-directory defaults."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate compact public statistics tables from final result artifacts."
    )
    parser.add_argument(
        "--baseline-dir",
        default="results/baseline",
        type=Path,
        help="Final baseline result directory.",
    )
    parser.add_argument(
        "--enhanced-dir",
        default="results/enhanced",
        type=Path,
        help="Final enhanced result directory.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/statistics",
        type=Path,
        help="Output directory for compact public statistics.",
    )
    parser.add_argument("--n_bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "src" / "reporting" / "build_statistics.py"
    cmd = [
        sys.executable,
        str(script),
        "--baseline-dir",
        str(args.baseline_dir),
        "--enhanced-dir",
        str(args.enhanced_dir),
        "--output-dir",
        str(args.output_dir),
        "--n_bootstrap",
        str(args.n_bootstrap),
        "--seed",
        str(args.seed),
    ]
    return subprocess.call(cmd, cwd=repo_root)


if __name__ == "__main__":
    raise SystemExit(main())
