"""Verify checkpoint paths and sizes from CHECKPOINT_INDEX.csv."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INDEX = ROOT / "CHECKPOINT_INDEX.csv"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    args = parser.parse_args()
    errors = []
    with args.index.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        path = args.project_root / row["relative_path"]
        if not path.is_file():
            errors.append(f"missing: {row['relative_path']}")
            continue
        if path.stat().st_size != int(row["size_bytes"]):
            errors.append(f"size mismatch: {row['relative_path']}")
    if errors:
        for error in errors[:50]:
            print(f"FAIL: {error}")
        if len(errors) > 50:
            print(f"FAIL: ... and {len(errors) - 50} more")
        return 1
    print(f"PASS: verified {len(rows)} checkpoint paths and sizes")
    return 0


if __name__ == "__main__":
    sys.exit(main())