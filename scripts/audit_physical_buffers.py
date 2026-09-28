"""Audit physical separation between few-shot support and query chips."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

import os as _repo_os
PROJECT = Path(_repo_os.environ.get("LANDSLIDE_PROJECT_ROOT", Path(__file__).resolve().parents[1])).resolve()
INDEX = PROJECT / "data" / "processed" / "benchmark_v2_regions_512" / "target_spatial_index.csv"
MANIFEST = PROJECT / "data" / "processed" / "benchmark_v2_regions_512" / "manifest_benchmark_v2.csv"
SPLIT_DIR = PROJECT / "data" / "processed" / "fewshot_target_splits"
REPORTS = PROJECT / "reports"
REGIONS = ["hokkaido_iburi_tobu", "lombok", "palu"]
SEEDS = [42, 2026, 777]
BUFFERS_M = [0, 128, 256, 512, 1000]


def load_split(seed):
    if seed == 42:
        support_path = SPLIT_DIR / "fewshot_support_seed42.csv"
        eval_path = SPLIT_DIR / "fewshot_eval_seed42.csv"
    else:
        support_path = SPLIT_DIR / "fewshot_support_seed2026_777.csv"
        eval_path = SPLIT_DIR / "fewshot_eval_seed2026_777.csv"
    support = [row for row in csv.DictReader(support_path.open(encoding="utf-8-sig")) if int(row["seed"]) == seed and int(row["shot"]) == 20]
    query = [row for row in csv.DictReader(eval_path.open(encoding="utf-8-sig")) if int(row["seed"]) == seed]
    return support, query


def box_gap_m(left, right, gsd_left, gsd_right):
    dx = abs(int(left["x"]) - int(right["x"])) * 256.0
    dy = abs(int(left["y"]) - int(right["y"])) * 256.0
    center_x = dx * gsd_left
    center_y = dy * gsd_left
    gap_x = max(center_x - 256.0 * gsd_left - 256.0 * gsd_right, 0.0)
    gap_y = max(center_y - 256.0 * gsd_left - 256.0 * gsd_right, 0.0)
    center_distance = float(np.hypot(center_x, center_y))
    return center_distance, float(np.hypot(gap_x, gap_y))


def main() -> None:
    index_rows = list(csv.DictReader(INDEX.open(encoding="utf-8-sig")))
    manifest_rows = list(csv.DictReader(MANIFEST.open(encoding="utf-8-sig")))
    gsd = {row["id"]: float(row["ground_resolution_m"]) for row in manifest_rows}
    index = {(row["region"], row["id"]): row for row in index_rows}
    summary = []
    buffer_rows = []
    for region in REGIONS:
        for seed in SEEDS:
            support, query = load_split(seed)
            support = [index[(region, row["id"])] for row in support if row["region"] == region]
            query = [index[(region, row["id"])] for row in query if row["region"] == region]
            min_center = float("inf")
            min_gap = float("inf")
            for left in support:
                for right in query:
                    if int(left["component_id"]) != int(right["component_id"]):
                        continue
                    center, gap = box_gap_m(left, right, gsd[left["id"]], gsd[right["id"]])
                    min_center = min(min_center, center)
                    min_gap = min(min_gap, gap)
            for buffer_m in BUFFERS_M:
                retained = 0
                for right in query:
                    same_component = [left for left in support if int(left["component_id"]) == int(right["component_id"])]
                    if not same_component:
                        retained += 1
                        continue
                    gaps = [box_gap_m(left, right, gsd[left["id"]], gsd[right["id"]])[1] for left in same_component]
                    retained += int(min(gaps) >= buffer_m)
                buffer_rows.append({
                    "seed": seed,
                    "region": region,
                    "n_support": len(support),
                    "n_query_current": len(query),
                    "buffer_m": buffer_m,
                    "n_query_retained": retained,
                    "retained_fraction": retained / max(len(query), 1),
                })
            summary.append({
                "seed": seed,
                "region": region,
                "n_support": len(support),
                "n_query_current": len(query),
                "min_center_distance_m": min_center,
                "min_edge_gap_m": min_gap,
                "current_min_chebyshev_grid": min(max(abs(int(left["x"]) - int(right["x"])), abs(int(left["y"]) - int(right["y"]))) for left in support for right in query if int(left["component_id"]) == int(right["component_id"])),
                "min_gsd_m": min([gsd[row["id"]] for row in support + query]),
                "max_gsd_m": max([gsd[row["id"]] for row in support + query]),
            })

    with (REPORTS / "e0_physical_buffer_summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)
    with (REPORTS / "e0_physical_buffer_counts.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(buffer_rows[0].keys()))
        writer.writeheader()
        writer.writerows(buffer_rows)

    lines = [
        "# E0.4: Physical Support-Query Separation Audit",
        "",
        "Distances use the reconstructed stride-256 grid plus per-chip ground sample distance from the CAS manifest. `edge gap` is the distance between axis-aligned 512-pixel chip footprints; 0 m means the chips touch but do not overlap.",
        "",
        "| Seed | Region | Min grid Chebyshev | Min center distance (m) | Min edge gap (m) | GSD range (m) |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(f"| {row['seed']} | {row['region']} | {row['current_min_chebyshev_grid']} | {row['min_center_distance_m']:.1f} | {row['min_edge_gap_m']:.1f} | {row['min_gsd_m']:.3f}-{row['max_gsd_m']:.3f} |")
    lines += [
        "",
        "## Query Retention Under Physical Buffers",
        "",
        "| Seed | Region | Current query | Buffer 0 m | Buffer 256 m | Buffer 512 m | Buffer 1000 m |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    grouped = defaultdict(dict)
    for row in buffer_rows:
        grouped[(row["seed"], row["region"])][row["buffer_m"]] = row["n_query_retained"] / max(row["n_query_current"], 1)
    for key, values in grouped.items():
        current = next(row for row in buffer_rows if row["seed"] == key[0] and row["region"] == key[1] and row["buffer_m"] == 0)
        lines.append(f"| {key[0]} | {key[1]} | {current['n_query_current']} | {values[0]:.3f} | {values[256]:.3f} | {values[512]:.3f} | {values[1000]:.3f} |")
    lines += [
        "",
        "## Conclusion",
        "",
        "The current split guarantees non-overlap, but it does not guarantee a physical buffer: the minimum edge gap can be zero because adjacent stride-256 chips touch. The source train/validation split cannot be assigned physical distances from the current PNG manifest because source geolocation is absent. A revised few-shot split should use an explicit GSD-aware buffer, and the source split should be rebuilt from georeferenced source imagery if available.",
    ]
    (REPORTS / "e0_physical_buffer_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"min_edge_gap_m": min(row["min_edge_gap_m"] for row in summary), "min_center_distance_m": min(row["min_center_distance_m"] for row in summary)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
