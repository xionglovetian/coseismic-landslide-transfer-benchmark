"""Create GSD-aware buffered query splits for E4."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

import os as _repo_os
PROJECT = Path(_repo_os.environ.get("LANDSLIDE_PROJECT_ROOT", Path(__file__).resolve().parents[1])).resolve()
INDEX = PROJECT / "data" / "processed" / "benchmark_v2_regions_512" / "target_spatial_index.csv"
MANIFEST = PROJECT / "data" / "processed" / "benchmark_v2_regions_512" / "manifest_benchmark_v2.csv"
SPLIT_DIR = PROJECT / "data" / "processed" / "fewshot_target_splits"
OUT_DIR = SPLIT_DIR / "buffered"
REGIONS = ["hokkaido_iburi_tobu", "lombok", "palu"]
SEEDS = [42, 2026, 777]
BUFFERS = [256, 512]


def support_path(seed):
    return SPLIT_DIR / ("fewshot_support_seed42.csv" if seed == 42 else "fewshot_support_seed2026_777.csv")


def gap_m(left, right, gsd_left, gsd_right):
    dx = abs(int(left["x"]) - int(right["x"])) * 256.0
    dy = abs(int(left["y"]) - int(right["y"])) * 256.0
    center_x = dx * gsd_left
    center_y = dy * gsd_left
    gap_x = max(center_x - 256.0 * gsd_left - 256.0 * gsd_right, 0.0)
    gap_y = max(center_y - 256.0 * gsd_left - 256.0 * gsd_right, 0.0)
    return float(np.hypot(gap_x, gap_y))


def main():
    index_rows = list(csv.DictReader(INDEX.open(encoding="utf-8-sig")))
    manifest_rows = list(csv.DictReader(MANIFEST.open(encoding="utf-8-sig")))
    gsd = {row["id"]: float(row["ground_resolution_m"]) for row in manifest_rows}
    for seed in SEEDS:
        support_rows = [row for row in csv.DictReader(support_path(seed).open(encoding="utf-8-sig")) if int(row["seed"]) == seed and int(row["shot"]) == 20]
        for region in REGIONS:
            support = [row for row in support_rows if row["region"] == region]
            candidates = [row for row in index_rows if row["region"] == region]
            for buffer_m in BUFFERS:
                query = []
                for right in candidates:
                    same = [left for left in support if int(left["component_id"]) == int(right["component_id"])]
                    if same and min(gap_m(left, right, gsd[left["id"]], gsd[right["id"]]) for left in same) < buffer_m:
                        continue
                    row = dict(right)
                    row["seed"] = seed
                    row["buffer_m"] = buffer_m
                    query.append(row)
                path = OUT_DIR / f"eval_seed{seed}_{region}_buffer{buffer_m}.csv"
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("w", newline="", encoding="utf-8-sig") as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(query[0].keys()))
                    writer.writeheader()
                    writer.writerows(query)
                print(seed, region, buffer_m, len(query))


if __name__ == "__main__":
    main()
