#!/usr/bin/env python3
"""Validate public compact-result manifests and checksum records."""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_manifest_csv(path: Path) -> list[str]:
    errors: list[str] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"path", "size_bytes", "sha256"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            if path.name == "local_only_manifest.csv":
                return []
            return [f"{path}: missing columns {sorted(missing)}"]

        for row_number, row in enumerate(reader, start=2):
            target = Path(row["path"])
            if not target.exists():
                errors.append(f"{path}:{row_number}: missing file {target}")
                continue
            observed_size = target.stat().st_size
            try:
                expected_size = int(row["size_bytes"])
            except ValueError:
                errors.append(f"{path}:{row_number}: invalid size {row['size_bytes']!r}")
                continue
            if observed_size != expected_size:
                errors.append(
                    f"{path}:{row_number}: size mismatch for {target}: "
                    f"{observed_size} != {expected_size}"
                )
            observed_hash = sha256(target)
            if observed_hash != row["sha256"]:
                errors.append(
                    f"{path}:{row_number}: sha256 mismatch for {target}: "
                    f"{observed_hash} != {row['sha256']}"
                )
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--roots",
        nargs="+",
        type=Path,
        required=True,
        help="Public roots containing *_manifest.csv files and optional checksums.csv files.",
    )
    args = parser.parse_args()

    manifests: list[Path] = []
    for root in args.roots:
        if root.is_file() and root.suffix == ".csv":
            manifests.append(root)
        elif root.is_dir():
            manifests.extend(sorted(root.rglob("*manifest.csv")))
            checksums = root / "checksums.csv"
            if checksums.exists():
                manifests.append(checksums)

    errors: list[str] = []
    for manifest in sorted(set(manifests)):
        errors.extend(validate_manifest_csv(manifest))

    if errors:
        for error in errors:
            print(f"FAIL: {error}")
        return 1

    print(f"PASS: validated {len(set(manifests))} manifest/checksum file(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
