"""Rebuild MANIFEST_SHA256.csv for the public package."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "MANIFEST_SHA256.csv"



def stable_bytes(path: Path) -> bytes:
    """Return bytes as checked out under the repository LF policy."""
    data = path.read_bytes()
    if b"\x00" not in data:
        data = data.replace(b"\r\n", b"\n")
    return data

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    for block in [stable_bytes(path)]:
        digest.update(block)
    return digest.hexdigest()


def main() -> None:
    rows = []
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file():
            continue
        if ".git" in path.parts or "__pycache__" in path.parts or path.suffix == ".pyc" or path == OUTPUT:
            continue
        rows.append(
            {
                "relative_path": path.relative_to(ROOT).as_posix(),
                "size_bytes": len(stable_bytes(path)),
                "sha256": sha256(path),
            }
        )
    with OUTPUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["relative_path", "size_bytes", "sha256"], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"WROTE: {OUTPUT} ({len(rows)} files)")


if __name__ == "__main__":
    main()
