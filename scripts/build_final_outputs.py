#!/usr/bin/env python3
"""Run final public table/figure generation with final artifact defaults."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate compact public final tables and figures from final artifacts.")
    parser.add_argument("--statistics-dir", dest="statistics_dir", type=Path, default=Path("results/statistics"))
    parser.add_argument("--internal-robustness-dir", type=Path, default=Path("results/internal_robustness"))
    parser.add_argument("--baseline-dir", type=Path, default=Path("results/baseline"))
    parser.add_argument("--enhanced-dir", type=Path, default=Path("results/enhanced"))
    parser.add_argument("--tables-dir", type=Path, default=Path("results/final_tables"))
    parser.add_argument("--figures-dir", type=Path, default=Path("results/final_figures"))
    parser.add_argument("--output-dir", type=Path, default=None, help="Deprecated legacy single output directory.")
    parser.add_argument("--write-latex-snippets", action="store_true")
    parser.add_argument("--skip-figures", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "src" / "reporting" / "build_final_outputs.py"
    cmd = [
        sys.executable,
        str(script),
        "--statistics-dir",
        str(args.statistics_dir),
        "--internal-robustness-dir",
        str(args.internal_robustness_dir),
        "--baseline-dir",
        str(args.baseline_dir),
        "--enhanced-dir",
        str(args.enhanced_dir),
        "--tables-dir",
        str(args.tables_dir),
        "--figures-dir",
        str(args.figures_dir),
    ]
    if args.output_dir is not None:
        cmd += ["--output-dir", str(args.output_dir)]
    if args.write_latex_snippets:
        cmd.append("--write-latex-snippets")
    if args.skip_figures:
        cmd.append("--skip-figures")
    return subprocess.call(cmd, cwd=repo_root)


if __name__ == "__main__":
    raise SystemExit(main())
