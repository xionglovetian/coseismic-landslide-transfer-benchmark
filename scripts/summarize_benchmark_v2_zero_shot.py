"""Summarize 7-region CAS benchmark v2 zero-shot results."""

from __future__ import annotations

import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

import os as _repo_os
PROJECT = Path(_repo_os.environ.get("LANDSLIDE_PROJECT_ROOT", Path(__file__).resolve().parents[1])).resolve()
REPORTS = PROJECT / "reports"
FIGURES = PROJECT / "figures"
RAW_DIR = REPORTS / "benchmark_v2_zero_shot_raw"
OUT_PREFIX = "benchmark_v2_zero_shot"
SEEDS = [42, 2026, 777]
REGION_ORDER = ["wenchuan", "jiuzhai_valley", "moxitaidi", "longxi_river", "hokkaido_iburi_tobu", "lombok", "palu"]
REGION_LABELS = {"wenchuan": "Wenchuan", "jiuzhai_valley": "Jiuzhai Valley", "moxitaidi": "Moxitaidi", "longxi_river": "Longxi River", "hokkaido_iburi_tobu": "Hokkaido", "lombok": "Lombok", "palu": "Palu"}
MAIN_MODEL = "Bottleneck-LiteASK"

def metric_rows():
    rows = []
    for path in sorted(RAW_DIR.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        model = data.get("model", path.stem)
        seed = int(data["seed"])
        for region, group in data["groups"].items():
            threshold = group["threshold_metrics"]["0.5"]
            rows.append({
                "model": model,
                "seed": seed,
                "region": region,
                "region_label": REGION_LABELS.get(region, region),
                "iou": threshold["iou"],
                "dice": threshold["dice"],
                "bf1_2": group.get("f1_2_mean"),
                "bf1_4": group.get("f1_4_mean"),
                "hd95": group.get("hd95_mean"),
            })
    return rows


def aggregate(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["model"], row["region"], row["region_label"])].append(row)
    output = []
    for (model, region, region_label), entries in sorted(
        grouped.items(),
        key=lambda item: (item[0][0], REGION_ORDER.index(item[0][1]) if item[0][1] in REGION_ORDER else 99),
    ):
        record = {"model": model, "region": region, "region_label": region_label, "n_seeds": len(entries)}
        for metric in ["iou", "dice", "bf1_2", "bf1_4", "hd95"]:
            values = [float(entry[metric]) for entry in entries if entry[metric] is not None]
            record[f"{metric}_mean"] = statistics.fmean(values) if values else None
            record[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        output.append(record)
    return output


def model_summary(summary_rows):
    output = []
    for model in sorted({row["model"] for row in summary_rows}):
        entries = [row for row in summary_rows if row["model"] == model]
        region_means = {row["region"]: row["iou_mean"] for row in entries}
        worst_region = min(region_means, key=region_means.get)
        record = {
            "model": model,
            "n_seeds": min(row["n_seeds"] for row in entries),
            "macro_iou": statistics.fmean(region_means.values()),
            "worst_region": REGION_LABELS.get(worst_region, worst_region),
            "worst_region_iou": region_means[worst_region],
        }
        for metric in ["dice", "bf1_2", "bf1_4", "hd95"]:
            record[f"macro_{metric}"] = statistics.fmean(row[f"{metric}_mean"] for row in entries)
        output.append(record)
    return output


def write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def fmt(value, digits=4):
    return "NA" if value is None else f"{float(value):.{digits}f}"


def main():
    rows = metric_rows()
    summary = aggregate(rows)
    models = model_summary(summary)
    write_csv(REPORTS / f"{OUT_PREFIX}_evidence.csv", rows, list(rows[0].keys()))
    write_csv(REPORTS / f"{OUT_PREFIX}_region_summary.csv", summary, list(summary[0].keys()))
    write_csv(REPORTS / f"{OUT_PREFIX}_model_summary.csv", models, list(models[0].keys()))
    lines = [
        "# CAS Landslide Benchmark v2: Seven-Region Zero-Shot Results",
        "",
        "All models are trained on the CAS source split and evaluated on the same seven held-out regions.",
        "",
        "| Model | Seeds | Macro IoU | Worst region | Worst IoU | Macro Dice | Macro BF1@2 | Macro BF1@4 | Macro HD95 |",
        "|---|---:|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in models:
        lines.append(
            f"| {row['model']} | {row['n_seeds']} | {fmt(row['macro_iou'])} | {row['worst_region']} | "
            f"{fmt(row['worst_region_iou'])} | {fmt(row['macro_dice'])} | {fmt(row['macro_bf1_2'])} | "
            f"{fmt(row['macro_bf1_4'])} | {fmt(row['macro_hd95'], 2)} |"
        )
    lines += ["", "## Per-Region IoU", "", "| Model | Region | Seeds | IoU mean +/- SD | Dice | BF1@2 | BF1@4 | HD95 |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in summary:
        lines.append(
            f"| {row['model']} | {row['region_label']} | {row['n_seeds']} | "
            f"{fmt(row['iou_mean'])} +/- {fmt(row['iou_std'])} | {fmt(row['dice_mean'])} | "
            f"{fmt(row['bf1_2_mean'])} | {fmt(row['bf1_4_mean'])} | {fmt(row['hd95_mean'], 2)} |"
        )
    (REPORTS / f"{OUT_PREFIX}_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("completed checkpoints", len(rows))


if __name__ == "__main__":
    main()
