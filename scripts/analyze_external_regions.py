"""Compute reproducible image and mask statistics for external benchmark regions."""

from __future__ import annotations

import argparse
import csv
import math
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

import os as _repo_os
PROJECT = Path(_repo_os.environ.get("LANDSLIDE_PROJECT_ROOT", Path(__file__).resolve().parents[1])).resolve()
MANIFEST = PROJECT / "data" / "processed" / "external_regions_512" / "manifest_external.csv"
REPORTS = PROJECT / "reports"
REGION_LABELS = {
    "wenchuan": "Wenchuan",
    "jiuzhai_valley": "Jiuzhai Valley",
    "moxitaidi": "Moxitaidi",
    "longxi_river": "Longxi River",
}
REGION_ORDER = ["wenchuan", "jiuzhai_valley", "moxitaidi", "longxi_river"]

cv2.setNumThreads(1)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(MANIFEST))
    parser.add_argument("--sample-per-region", type=int, default=0, help="0 means use every eligible image")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-prefix", default="external_region_statistics")
    return parser.parse_args()


def numeric(row, key, default=0.0):
    value = row.get(key, "")
    if value in (None, "", "nan", "NaN"):
        return default
    return float(value)


def image_statistics(row):
    image = np.asarray(Image.open(row["image_path"]).convert("RGB"), dtype=np.float32)
    mask = np.asarray(Image.open(row["mask_path"]).convert("L")) > 0
    mask_u8 = mask.astype(np.uint8)
    target_pixels = int(mask.sum())
    image_pixels = int(mask.size)

    luminance = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
    hsv = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2HSV)

    boundary = cv2.morphologyEx(mask_u8, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
    boundary_pixels = int(boundary.sum())
    n_components, _, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    component_areas = [int(stats[index, cv2.CC_STAT_AREA]) for index in range(1, n_components)]
    components_ge_8 = sum(area >= 8 for area in component_areas)
    largest_component_share = max(component_areas, default=0) / max(target_pixels, 1)
    perimeter_area_ratio = boundary_pixels / math.sqrt(max(target_pixels, 1))
    empty = target_pixels == 0

    return {
        "id": row["id"],
        "region": row["region"],
        "width": int(float(row["width"])),
        "height": int(float(row["height"])),
        "target_pixels": target_pixels,
        "foreground_fraction": target_pixels / image_pixels,
        "manifest_foreground_fraction": numeric(row, "foreground_fraction"),
        "empty_mask": int(empty),
        "luminance_mean": float(np.mean(luminance)),
        "luminance_std": float(np.std(luminance)),
        "r_mean": float(np.mean(image[:, :, 0])),
        "g_mean": float(np.mean(image[:, :, 1])),
        "b_mean": float(np.mean(image[:, :, 2])),
        "saturation_mean": float(np.mean(hsv[:, :, 1])),
        "boundary_pixels": boundary_pixels,
        "perimeter_area_ratio": perimeter_area_ratio,
        "components_ge_8": components_ge_8,
        "largest_component_share": largest_component_share,
    }


def safe_mean(values):
    present = [float(value) for value in values if value is not None and not math.isnan(float(value))]
    return statistics.fmean(present) if present else None


def safe_median(values):
    present = [float(value) for value in values if value is not None and not math.isnan(float(value))]
    return statistics.median(present) if present else None


def safe_std(values):
    present = [float(value) for value in values if value is not None and not math.isnan(float(value))]
    return statistics.stdev(present) if len(present) > 1 else (0.0 if present else None)


def percentile(values, fraction):
    present = [float(value) for value in values if value is not None and not math.isnan(float(value))]
    return float(np.percentile(present, fraction * 100)) if present else None


def aggregate_region(region, rows):
    result = {
        "region": region,
        "region_label": REGION_LABELS.get(region, region),
        "n_samples": len(rows),
        "resolution_mode": "NA",
        "width_min": min(row["width"] for row in rows),
        "width_max": max(row["width"] for row in rows),
        "height_min": min(row["height"] for row in rows),
        "height_max": max(row["height"] for row in rows),
        "empty_mask_percent": 100.0 * safe_mean(row["empty_mask"] for row in rows),
    }
    resolution_counts = Counter((row["width"], row["height"]) for row in rows)
    result["resolution_mode"] = "x".join(map(str, resolution_counts.most_common(1)[0][0]))
    result["resolution_unique"] = len(resolution_counts)

    metric_fields = [
        "target_pixels", "foreground_fraction", "luminance_mean", "luminance_std",
        "r_mean", "g_mean", "b_mean", "saturation_mean", "boundary_pixels",
        "perimeter_area_ratio", "components_ge_8", "largest_component_share",
    ]
    for field in metric_fields:
        values = [row[field] for row in rows]
        result[f"{field}_mean"] = safe_mean(values)
        result[f"{field}_std"] = safe_std(values)
        result[f"{field}_median"] = safe_median(values)
        result[f"{field}_p95"] = percentile(values, 0.95)
    return result


def fnum(value, digits=4):
    return "NA" if value is None else f"{float(value):.{digits}f}"


def write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def report_text(aggregates, sample_per_region):
    lines = [
        "# External Region Statistics",
        "",
        f"Sampling: {'all eligible images' if sample_per_region <= 0 else str(sample_per_region) + ' images per region'}.",
        "Foreground and boundary metrics are computed from strict external masks.",
        "",
        "| Region | N | Resolution (mode) | Empty masks | Foreground mean | Foreground median | Luminance | Saturation | RGB mean | Perimeter/sqrt(area) | Components >=8 px | Largest component share |",
        "|---|---:|---|---:|---:|---:|---:|---:|---|---:|---:|---:|",
    ]
    for row in aggregates:
        rgb = f"({fnum(row['r_mean_mean'], 1)}, {fnum(row['g_mean_mean'], 1)}, {fnum(row['b_mean_mean'], 1)})"
        lines.append(
            f"| {row['region_label']} | {row['n_samples']} | {row['resolution_mode']} | "
            f"{fnum(row['empty_mask_percent'], 2)}% | {fnum(100 * row['foreground_fraction_mean'], 3)}% | "
            f"{fnum(100 * row['foreground_fraction_median'], 3)}% | {fnum(row['luminance_mean_mean'], 2)} | "
            f"{fnum(row['saturation_mean_mean'], 2)} | {rgb} | {fnum(row['perimeter_area_ratio_mean'], 3)} | "
            f"{fnum(row['components_ge_8_mean'], 2)} | {fnum(row['largest_component_share_mean'], 4)} |"
        )
    lines += [
        "",
        "## Interpretation Guardrail",
        "",
        "These statistics describe domain and target differences; they do not by themselves prove causal effects on model performance.",
        "Use them alongside per-region LORO metrics and qualitative examples.",
    ]
    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    rows = [row for row in csv.DictReader(Path(args.manifest).open(encoding="utf-8-sig")) if int(row.get("eligible_external", 1)) == 1]
    by_region = defaultdict(list)
    for row in rows:
        by_region[row["region"]].append(row)

    statistics_rows = []
    for region in REGION_ORDER:
        region_rows = by_region.get(region, [])
        if args.sample_per_region > 0 and len(region_rows) > args.sample_per_region:
            rng = random.Random(args.seed + REGION_ORDER.index(region))
            region_rows = rng.sample(region_rows, args.sample_per_region)
        for row in tqdm(region_rows, desc=f"stats {REGION_LABELS.get(region, region)}"):
            statistics_rows.append(image_statistics(row))

    aggregates = [
        aggregate_region(region, [row for row in statistics_rows if row["region"] == region])
        for region in REGION_ORDER
        if any(row["region"] == region for row in statistics_rows)
    ]

    REPORTS.mkdir(parents=True, exist_ok=True)
    prefix = args.output_prefix
    per_image_path = REPORTS / f"{prefix}_per_image.csv"
    aggregate_path = REPORTS / f"{prefix}.csv"
    report_path = REPORTS / f"{prefix}_report.md"

    per_image_fields = list(statistics_rows[0].keys())
    aggregate_fields = list(aggregates[0].keys())
    write_csv(per_image_path, statistics_rows, per_image_fields)
    write_csv(aggregate_path, aggregates, aggregate_fields)
    report_path.write_text(report_text(aggregates, args.sample_per_region), encoding="utf-8")

    print(f"wrote {per_image_path}")
    print(f"wrote {aggregate_path}")
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()
