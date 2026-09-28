"""Join region descriptors with benchmark performance for descriptive analysis."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import os as _repo_os
PROJECT = Path(_repo_os.environ.get("LANDSLIDE_PROJECT_ROOT", Path(__file__).resolve().parents[1])).resolve()
REPORTS = PROJECT / "reports"
STATS_PATH = REPORTS / "external_region_statistics.csv"
LORO_PATH = REPORTS / "loro_benchmark_summary.csv"
OUT_CSV = REPORTS / "external_region_difficulty.csv"
OUT_MD = REPORTS / "external_region_difficulty_report.md"
REGION_LABELS = {
    "wenchuan": "Wenchuan",
    "jiuzhai_valley": "Jiuzhai Valley",
    "moxitaidi": "Moxitaidi",
    "longxi_river": "Longxi River",
}
FEATURES = [
    ("foreground_fraction_mean", "Foreground fraction"),
    ("luminance_mean_mean", "Luminance"),
    ("saturation_mean_mean", "Saturation"),
    ("perimeter_area_ratio_mean", "Boundary complexity"),
    ("components_ge_8_mean", "Components >=8 px"),
    ("largest_component_share_mean", "Largest component share"),
]


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def to_float(value):
    if value in (None, "", "NA"):
        return None
    return float(value)


def pearson(xs, ys):
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    if len(pairs) < 2:
        return None
    x_values = [pair[0] for pair in pairs]
    y_values = [pair[1] for pair in pairs]
    x_mean = sum(x_values) / len(x_values)
    y_mean = sum(y_values) / len(y_values)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in pairs)
    denominator = math.sqrt(
        sum((x - x_mean) ** 2 for x in x_values)
        * sum((y - y_mean) ** 2 for y in y_values)
    )
    return numerator / denominator if denominator else None


def fmt(value, digits=4):
    return "NA" if value is None else f"{float(value):.{digits}f}"


def main():
    stats = {row["region"]: row for row in read_csv(STATS_PATH)}
    loro_rows = read_csv(LORO_PATH)
    zero_shot = {
        row["region"]: row
        for row in loro_rows
        if row["method"] == "CAS-only zero-shot"
    }
    uniform = {
        row["region"]: row
        for row in loro_rows
        if row["method"] == "Uniform multi-source"
    }

    output_rows = []
    for region, stats_row in stats.items():
        zero_row = zero_shot.get(region)
        uniform_row = uniform.get(region)
        output = {
            "region": region,
            "region_label": REGION_LABELS.get(region, region),
            "n_samples": stats_row["n_samples"],
            "zero_shot_iou": to_float(zero_row["iou_mean"]) if zero_row else None,
            "zero_shot_iou_std": to_float(zero_row["iou_std"]) if zero_row else None,
            "uniform_iou": to_float(uniform_row["iou_mean"]) if uniform_row else None,
            "uniform_iou_std": to_float(uniform_row["iou_std"]) if uniform_row else None,
        }
        for key, _ in FEATURES:
            output[key] = to_float(stats_row.get(key))
        output["mean_target_area_pixels"] = (
            output["foreground_fraction_mean"] * 512 * 512
            if output["foreground_fraction_mean"] is not None
            else None
        )
        output_rows.append(output)

    correlations = []
    for performance_key in ["zero_shot_iou", "uniform_iou"]:
        performance = [row[performance_key] for row in output_rows]
        for feature_key, feature_label in FEATURES:
            feature = [row[feature_key] for row in output_rows]
            correlations.append({
                "performance": performance_key,
                "feature": feature_label,
                "pearson_r": pearson(feature, performance),
                "n_regions": sum(x is not None and y is not None for x, y in zip(feature, performance)),
            })

    with OUT_CSV.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0].keys()))
        writer.writeheader()
        writer.writerows(output_rows)

    lines = [
        "# External Region Difficulty Analysis",
        "",
        "All images are evaluated after the frozen 512x512 preprocessing. Descriptors are descriptive only: with four regions, correlations are exploratory and cannot establish causality.",
        "",
        "| Region | N | Zero-shot IoU | Uniform IoU | Mean foreground | Mean target pixels | Luminance | Saturation | Boundary complexity | Components >=8 px |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in output_rows:
        lines.append(
            f"| {row['region_label']} | {row['n_samples']} | {fmt(row['zero_shot_iou'])} | {fmt(row['uniform_iou'])} | "
            f"{fmt(100 * row['foreground_fraction_mean'], 2)}% | {fmt(row['mean_target_area_pixels'], 0)} | "
            f"{fmt(row['luminance_mean_mean'], 2)} | {fmt(row['saturation_mean_mean'], 2)} | "
            f"{fmt(row['perimeter_area_ratio_mean'], 2)} | {fmt(row['components_ge_8_mean'], 2)} |"
        )

    lines += [
        "",
        "## Exploratory Correlations",
        "",
        "| Performance | Descriptor | Pearson r | Regions |",
        "|---|---|---:|---:|",
    ]
    for row in correlations:
        lines.append(
            f"| {row['performance']} | {row['feature']} | {fmt(row['pearson_r'], 3)} | {row['n_regions']} |"
        )

    by_zero = sorted(output_rows, key=lambda row: row["zero_shot_iou"])
    easiest = by_zero[-1]
    hardest = by_zero[0]
    lines += [
        "",
        "## Descriptive Reading",
        "",
        f"- {hardest['region_label']} has the lowest zero-shot IoU ({fmt(hardest['zero_shot_iou'])}), with a small foreground fraction ({fmt(100 * hardest['foreground_fraction_mean'], 2)}%) and high boundary complexity ({fmt(hardest['perimeter_area_ratio_mean'], 2)}).",
        f"- {easiest['region_label']} has the highest zero-shot IoU ({fmt(easiest['zero_shot_iou'])}), with a larger foreground fraction ({fmt(100 * easiest['foreground_fraction_mean'], 2)}%) and lower boundary complexity ({fmt(easiest['perimeter_area_ratio_mean'], 2)}).",
        "- The observed ordering is consistent with target scale and fragmentation mattering, but source appearance shift and annotation protocol differences remain uncontrolled.",
        "- Update this report after all uniform 3-seed runs finish; the uniform IoU column will then support a stronger domain-shift discussion.",
        "",
        "Files:",
        f"- {OUT_CSV}",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT_CSV}")
    print(f"wrote {OUT_MD}")


if __name__ == "__main__":
    main()
