"""Summarize leave-one-region-out (LORO) multi-source experiments.

The script intentionally tolerates partially completed experiment grids.  It
reports coverage explicitly and computes macro statistics only over regions
that are actually available for a method.
"""

from __future__ import annotations

import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

import os as _repo_os
PROJECT = Path(_repo_os.environ.get("LANDSLIDE_PROJECT_ROOT", Path(__file__).resolve().parents[1])).resolve()
REPORTS = PROJECT / "reports"
RAW_EXTERNAL = REPORTS / "external_regions_raw"
OUTPUTS = PROJECT / "outputs"
SEEDS = [42, 2026, 777]
TARGET_REGIONS = ["wenchuan", "jiuzhai_valley", "moxitaidi", "longxi_river"]
RUN_REGION_SLUGS = {"wenchuan": "wenchuan", "jiuzhai_valley": "jiuzhai_valley", "moxitaidi": "moxitaidi", "longxi_river": "longxi"}
REGION_LABELS = {
    "wenchuan": "Wenchuan",
    "jiuzhai_valley": "Jiuzhai Valley",
    "moxitaidi": "Moxitaidi",
    "longxi_river": "Longxi River",
}
METHOD_SPECS = [
    ("CAS-only zero-shot", "zero_shot", None),
    ("Uniform multi-source", "uniform_noalign", "dg_{run_region}_uniform_noalign_seed{seed}"),
    ("Balanced multi-source", "balanced_noalign", "dg_{run_region}_balanced_noalign_seed{seed}"),
    ("Balanced + feature alignment", "balanced_mmd", "dg_{run_region}_balanced_mmd_seed{seed}"),
]
METRICS = ["iou", "dice", "bf1_2", "bf1_4", "hd95"]
EVIDENCE_PATH = REPORTS / "loro_benchmark_evidence.csv"
SUMMARY_PATH = REPORTS / "loro_benchmark_summary.csv"
REPORT_PATH = REPORTS / "loro_benchmark_report.md"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def extract_metrics(region_data: dict) -> dict:
    threshold = region_data.get("threshold_metrics", {}).get("0.5", {})
    return {
        "iou": threshold.get("iou"),
        "dice": threshold.get("dice"),
        "bf1_2": region_data.get("f1_2_mean"),
        "bf1_4": region_data.get("f1_4_mean"),
        "hd95": region_data.get("hd95_mean"),
    }


def parse_zero_shot(region: str, seed: int):
    path = RAW_EXTERNAL / f"group_bottleneckliteaskunetpp_seed{seed}.json"
    if not path.exists():
        return None
    data = read_json(path)
    region_data = data.get("groups", {}).get(region)
    if not region_data:
        return None
    return {
        "metrics": extract_metrics(region_data),
        "source_val_iou": None,
        "best_epoch": None,
        "run_path": str(path),
    }


def parse_dg_run(region: str, seed: int, run_name: str):
    run_dir = OUTPUTS / run_name
    target_path = run_dir / "target_metrics.json"
    all_region_path = run_dir / "all_region_metrics.json"
    path = target_path if target_path.exists() else all_region_path
    if not path.exists():
        return None
    data = read_json(path)
    if data.get("invalidated", False):
        return None
    if "target" in data:
        region_data = data.get("target", {}).get(region)
    else:
        region_data = data.get("groups", {}).get(region)
    if not region_data:
        return None
    source_validation = data.get("source_validation") or {}
    return {
        "metrics": extract_metrics(region_data),
        "source_val_iou": source_validation.get("iou"),
        "best_epoch": data.get("best_epoch"),
        "run_path": str(run_dir),
    }


def mean(values):
    present = [float(value) for value in values if value is not None]
    return statistics.fmean(present) if present else None


def std(values):
    present = [float(value) for value in values if value is not None]
    if len(present) < 2:
        return 0.0 if present else None
    return statistics.stdev(present)


def fmt(value, digits=4):
    return "NA" if value is None else f"{float(value):.{digits}f}"


def pm(mean_value, std_value, digits=4):
    if mean_value is None:
        return "NA"
    return f"{float(mean_value):.{digits}f} +/- {float(std_value or 0.0):.{digits}f}"


def collect_evidence():
    rows = []
    expected = set()
    for method_label, method_key, run_pattern in METHOD_SPECS:
        for region in TARGET_REGIONS:
            for seed in SEEDS:
                expected.add((method_label, region, seed))
                if method_key == "zero_shot":
                    parsed = parse_zero_shot(region, seed)
                else:
                    run_name = run_pattern.format(run_region=RUN_REGION_SLUGS[region], seed=seed)
                    parsed = parse_dg_run(region, seed, run_name)
                if parsed is None:
                    continue
                row = {
                    "method": method_label,
                    "method_key": method_key,
                    "region": region,
                    "region_label": REGION_LABELS[region],
                    "seed": seed,
                    "best_epoch": parsed["best_epoch"],
                    "source_val_iou": parsed["source_val_iou"],
                    "run_path": parsed["run_path"],
                }
                row.update(parsed["metrics"])
                rows.append(row)
    return rows, expected


def aggregate(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["method"], row["method_key"], row["region"], row["region_label"])].append(row)

    summary_rows = []
    for (method, method_key, region, region_label), entries in sorted(
        grouped.items(), key=lambda item: (TARGET_REGIONS.index(item[0][2]), item[0][0])
    ):
        summary = {
            "method": method,
            "method_key": method_key,
            "region": region,
            "region_label": region_label,
            "n_seeds": len(entries),
        }
        for metric in METRICS:
            values = [entry.get(metric) for entry in entries]
            summary[f"{metric}_mean"] = mean(values)
            summary[f"{metric}_std"] = std(values)
        source_values = [entry.get("source_val_iou") for entry in entries]
        summary["source_val_iou_mean"] = mean(source_values)
        summary["source_val_iou_std"] = std(source_values)
        summary_rows.append(summary)
    return summary_rows


def macro_for_method(summary_rows, method, regions=None):
    selected = [
        row for row in summary_rows
        if row["method"] == method and (regions is None or row["region"] in regions)
    ]
    if not selected:
        return None
    record = {
        "method": method,
        "n_regions": len(selected),
        "regions": [row["region"] for row in selected],
    }
    for metric in METRICS:
        values = [row[f"{metric}_mean"] for row in selected if row[f"{metric}_mean"] is not None]
        record[f"macro_{metric}"] = mean(values)
        record[f"worst_{metric}"] = min(values) if values else None
        record[f"region_std_{metric}"] = std(values)
        if metric == "iou" and values:
            worst_value = min(values)
            worst_row = next(row for row in selected if row[f"{metric}_mean"] == worst_value)
            record["worst_iou_region"] = worst_row["region_label"]
            record["mean_seed_std_iou"] = mean([row["iou_std"] for row in selected])
    return record


def write_csv(path: Path, rows: list[dict], fields: list[str]):
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def render_report(evidence_rows, summary_rows, expected):
    found_keys = {(row["method"], row["region"], row["seed"]) for row in evidence_rows}
    missing = expected - found_keys
    methods = [label for label, _, _ in METHOD_SPECS]
    method_macros = [macro_for_method(summary_rows, method) for method in methods]
    method_macros = [row for row in method_macros if row]

    complete_regions = None
    for macro in method_macros:
        region_set = set(macro["regions"])
        complete_regions = region_set if complete_regions is None else complete_regions & region_set
    complete_regions = complete_regions or set()
    complete_macros = [
        macro_for_method(summary_rows, method, complete_regions) for method in methods
    ]
    complete_macros = [row for row in complete_macros if row]

    zero_shot_macro = macro_for_method(summary_rows, "CAS-only zero-shot")
    zero_shot_regions = set(zero_shot_macro["regions"]) if zero_shot_macro else set()
    paired_zero_shot = []
    for method in methods:
        if method == "CAS-only zero-shot":
            continue
        method_macro = macro_for_method(summary_rows, method)
        if not method_macro:
            continue
        common_regions = set(method_macro["regions"]) & zero_shot_regions
        method_comparable = macro_for_method(summary_rows, method, common_regions)
        zero_comparable = macro_for_method(summary_rows, "CAS-only zero-shot", common_regions)
        if method_comparable and zero_comparable:
            paired_zero_shot.append((method_comparable, zero_comparable))

    lines = [
        "# LORO Multi-Region Benchmark",
        "",
        "This report is generated from completed `target_metrics.json` and zero-shot evaluation JSON files.",
        "Only available regions are averaged into each method macro; no missing result is imputed.",
        "Multi-source rows inherit a 50-epoch CAS checkpoint and then train 20 additional epochs. Comparisons to zero-shot are descriptive until the compute-matched CAS-retrain20 control is completed.",
        "",
        "## Per-Region Results",
        "",
        "| Region | Method | Seeds | IoU mean +/- SD | Dice mean +/- SD | BF1@2 mean +/- SD | BF1@4 mean +/- SD | HD95 mean +/- SD | Source Val IoU |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for region in TARGET_REGIONS:
        region_rows = [row for row in summary_rows if row["region"] == region]
        for row in region_rows:
            source_value = pm(row["source_val_iou_mean"], row["source_val_iou_std"])
            lines.append(
                f"| {row['region_label']} | {row['method']} | {row['n_seeds']} | "
                f"{pm(row['iou_mean'], row['iou_std'])} | {pm(row['dice_mean'], row['dice_std'])} | "
                f"{pm(row['bf1_2_mean'], row['bf1_2_std'])} | {pm(row['bf1_4_mean'], row['bf1_4_std'])} | "
                f"{pm(row['hd95_mean'], row['hd95_std'], 2)} | {source_value} |"
            )

    lines += [
        "",
        "## Method Macro (Available Regions)",
        "",
        "| Method | Regions | Macro IoU | Worst IoU (region) | Region SD IoU | Macro Dice | Macro BF1@2 | Macro BF1@4 | Macro HD95 |",
        "|---|---:|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in method_macros:
        worst = f"{fmt(row.get('worst_iou'))} ({row.get('worst_iou_region', 'NA')})"
        lines.append(
            f"| {row['method']} | {row['n_regions']} | {fmt(row.get('macro_iou'))} | {worst} | "
            f"{fmt(row.get('region_std_iou'))} | {fmt(row.get('macro_dice'))} | "
            f"{fmt(row.get('macro_bf1_2'))} | {fmt(row.get('macro_bf1_4'))} | "
            f"{fmt(row.get('macro_hd95'), 2)} |"
        )

    lines += [
        "",
        "## Complete-Case Method Comparison",
        "",
        f"Regions available for every method: {', '.join(REGION_LABELS[r] for r in TARGET_REGIONS if r in complete_regions) or 'none'}.",
        "",
        "| Method | Regions | Macro IoU | Worst IoU (region) | Region SD IoU | Macro Dice | Macro BF1@2 | Macro BF1@4 | Macro HD95 |",
        "|---|---:|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in complete_macros:
        worst = f"{fmt(row.get('worst_iou'))} ({row.get('worst_iou_region', 'NA')})"
        lines.append(
            f"| {row['method']} | {row['n_regions']} | {fmt(row.get('macro_iou'))} | {worst} | "
            f"{fmt(row.get('region_std_iou'))} | {fmt(row.get('macro_dice'))} | "
            f"{fmt(row.get('macro_bf1_2'))} | {fmt(row.get('macro_bf1_4'))} | "
            f"{fmt(row.get('macro_hd95'), 2)} |"
        )

    lines += [
        "",
        "## Comparable to Zero-Shot",
        "",
        "| Method | Common regions | Method Macro IoU | Zero-Shot Macro IoU | Delta IoU | Method Worst IoU | Zero-Shot Worst IoU |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method_row, zero_row in paired_zero_shot:
        delta = method_row["macro_iou"] - zero_row["macro_iou"]
        lines.append(
            f"| {method_row['method']} | {method_row['n_regions']} | {fmt(method_row['macro_iou'])} | "
            f"{fmt(zero_row['macro_iou'])} | {delta:+.4f} | {fmt(method_row['worst_iou'])} | "
            f"{fmt(zero_row['worst_iou'])} |"
        )

    lines += [
        "",
        "## Coverage",
        "",
        f"- Completed runs: {len(found_keys)} / {len(expected)}.",
    ]
    for method in methods:
        completed = len({key for key in found_keys if key[0] == method})
        total = len(TARGET_REGIONS) * len(SEEDS)
        lines.append(f"- {method}: {completed} / {total} region-seed runs.")
    lines += [
        "",
        "Macro and worst-region values are provisional until every target has the same seed coverage.",
        "Zero-shot uses the three CAS-trained Bottleneck-LiteASK seeds; multi-source rows aggregate available LORO seeds.",
        "",
        "Files:",
        f"- {EVIDENCE_PATH}",
        f"- {SUMMARY_PATH}",
    ]
    return "\n".join(lines) + "\n"


def main():
    REPORTS.mkdir(parents=True, exist_ok=True)
    evidence_rows, expected = collect_evidence()
    summary_rows = aggregate(evidence_rows)
    evidence_fields = [
        "method", "method_key", "region", "region_label", "seed", "best_epoch",
        "iou", "dice", "bf1_2", "bf1_4", "hd95", "source_val_iou", "run_path",
    ]
    summary_fields = [
        "method", "method_key", "region", "region_label", "n_seeds",
        "iou_mean", "iou_std", "dice_mean", "dice_std",
        "bf1_2_mean", "bf1_2_std", "bf1_4_mean", "bf1_4_std",
        "hd95_mean", "hd95_std", "source_val_iou_mean", "source_val_iou_std",
    ]
    write_csv(EVIDENCE_PATH, evidence_rows, evidence_fields)
    write_csv(SUMMARY_PATH, summary_rows, summary_fields)
    REPORT_PATH.write_text(render_report(evidence_rows, summary_rows, expected), encoding="utf-8")
    print(f"wrote {EVIDENCE_PATH}")
    print(f"wrote {SUMMARY_PATH}")
    print(f"wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
