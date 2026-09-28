"""Copy a CSV manifest while rebasing path columns to a new project root."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--old-root", default="D:/landslide_unet_project")
    parser.add_argument("--new-root", required=True, type=Path)
    return parser.parse_args()


def rebase(value: str, old_root: str, new_root: str) -> str:
    normalized = value.replace("\\", "/")
    old = old_root.replace("\\", "/").rstrip("/")
    if normalized.lower() == old.lower():
        return new_root
    if normalized.lower().startswith(old.lower() + "/"):
        return new_root.rstrip("/") + normalized[len(old):]
    return value


def main() -> None:
    args = parse_args()
    with args.input.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        rows = list(reader)
    path_columns = [name for name in fieldnames if "path" in name.lower()]
    new_root = args.new_root.resolve().as_posix()
    for row in rows:
        for column in path_columns:
            row[column] = rebase(row[column], args.old_root, new_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"WROTE: {args.output} ({len(rows)} rows; rebased {len(path_columns)} columns)")


if __name__ == "__main__":
    main()
