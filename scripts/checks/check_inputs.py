"""Validate the exact local data layout required for primary reproduction."""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import sys
from collections import Counter
from pathlib import Path

EXPECTED = {
    "hokkaido_iburi_tobu": 1484,
    "jiuzhai_valley": 5925,
    "lombok": 436,
    "longxi_river": 2504,
    "moxitaidi": 980,
    "palu": 817,
    "wenchuan": 178,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(os.environ.get("LANDSLIDE_PROJECT_ROOT", Path(__file__).resolve().parents[2])),
    )
    parser.add_argument("--verify-hashes", action="store_true")
    args = parser.parse_args()
    root = args.project_root.resolve()
    errors: list[str] = []

    source = root / "repos/AS-UNet/inputs/data_sum_moxizhen+bijie"
    expected_source = {("train", "images"): 1795, ("train", "masks"): 1795, ("val", "images"): 513, ("val", "masks"): 513}
    for (split, kind), count in expected_source.items():
        folder = source / split / kind
        actual = len(list(folder.glob("*.png"))) if folder.is_dir() else 0
        if actual != count:
            errors.append(f"{source / split / kind}: {actual} PNGs, expected {count}")

    manifest = root / "data/processed/benchmark_v2_regions_512/manifest_benchmark_v2.csv"
    if not manifest.is_file():
        errors.append(f"missing benchmark manifest: {manifest}")
        rows = []
    else:
        with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        counts = Counter(row["region"] for row in rows)
        if len(rows) != sum(EXPECTED.values()):
            errors.append(f"benchmark rows: {len(rows)}, expected {sum(EXPECTED.values())}")
        for region, count in EXPECTED.items():
            if counts[region] != count:
                errors.append(f"{region}: {counts[region]} rows, expected {count}")
        missing = 0
        hash_mismatches = 0
        for row in rows:
            for path_key, hash_key in (("image_path", "image_sha256"), ("mask_path", "mask_sha256")):
                path = Path(row[path_key])
                if not path.is_file():
                    missing += 1
                    continue
                if args.verify_hashes and sha256(path) != row[hash_key]:
                    hash_mismatches += 1
        if missing:
            errors.append(f"missing benchmark image/mask files: {missing}")
        if hash_mismatches:
            errors.append(f"benchmark SHA-256 mismatches: {hash_mismatches}")

    if not (root / "data/processed/benchmark_v2_regions_512/target_spatial_index.csv").is_file():
        errors.append("missing target_spatial_index.csv")

    if errors:
        for error in errors:
            print(f"FAIL: {error}")
        return 1
    print(f"PASS: source counts, benchmark counts, paths and{' hashes' if args.verify_hashes else ' layout'} validated")
    return 0


if __name__ == "__main__":
    sys.exit(main())