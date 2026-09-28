"""Verify SHA-256 hashes listed in MANIFEST_SHA256.csv."""

from __future__ import annotations

import csv
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "MANIFEST_SHA256.csv"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    if not MANIFEST.is_file():
        print(f"FAIL: missing {MANIFEST.name}")
        return 1
    errors: list[str] = []
    with MANIFEST.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        path = ROOT / row["relative_path"]
        if not path.is_file():
            errors.append(f"missing: {row['relative_path']}")
            continue
        actual = sha256(path)
        if actual != row["sha256"]:
            errors.append(f"hash mismatch: {row['relative_path']}")
    if errors:
        for error in errors[:50]:
            print(f"FAIL: {error}")
        if len(errors) > 50:
            print(f"FAIL: ... and {len(errors) - 50} more")
        return 1
    print(f"PASS: verified {len(rows)} SHA-256 entries")
    return 0


if __name__ == "__main__":
    sys.exit(main())
