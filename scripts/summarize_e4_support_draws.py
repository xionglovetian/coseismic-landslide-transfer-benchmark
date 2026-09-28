"""Summarize support-set draw robustness for ResUNet 20-shot adaptation."""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

import os as _repo_os
PROJECT = Path(_repo_os.environ.get("LANDSLIDE_PROJECT_ROOT", Path(__file__).resolve().parents[1])).resolve()
OUTPUTS = PROJECT / "outputs"
REPORTS = PROJECT / "reports"
REGIONS = ["hokkaido_iburi_tobu", "lombok", "palu"]
LABELS = {"hokkaido_iburi_tobu": "Hokkaido", "lombok": "Lombok", "palu": "Palu"}
MODES = ["full", "decoder-only"]
DRAWS = [42, 2026, 777]


def mean(values):
    values = list(values)
    return float(statistics.fmean(values)) if values else float("nan")


def std(values):
    values = list(values)
    return float(statistics.stdev(values)) if len(values) > 1 else 0.0


def metrics_from_json(path, region):
    data = json.loads(path.read_text(encoding="utf-8"))
    group = data["groups"][region]
    threshold = group["threshold_metrics"]["0.5"]
    return {"iou": float(threshold["iou"]), "dice": float(threshold["dice"]), "bf1_2": float(group["f1_2_mean"]), "bf1_4": float(group["f1_4_mean"]), "hd95": float(group["hd95_mean"])}


def metrics_from_split(path, region):
    data = json.loads(path.read_text(encoding="utf-8"))
    threshold = data["threshold_metrics"]["0.5"]
    return {"iou": float(threshold["iou"]), "dice": float(threshold["dice"]), "bf1_2": float(data["boundary"]["f1_2_mean"]), "bf1_4": float(data["boundary"]["f1_4_mean"]), "hd95": float(data["boundary"]["hd95_mean"])}


def main():
    rows = []
    for draw in DRAWS:
        for mode in MODES:
            for region in REGIONS:
                if draw == 42:
                    adapted_path = OUTPUTS / f"bench_v2_fewshot128_resunet_{region}_20shot_{mode}_seed42" / "metrics.json"
                    source_path = OUTPUTS / f"bench_v2_fewshot128_resunet_{region}_zeroshot_seed42" / "metrics.json"
                    source = metrics_from_json(source_path, region)
                else:
                    adapted_path = OUTPUTS / f"bench_v2_e4_supportdraw{draw}_resunet_{region}_20shot_{mode}_init42" / "metrics.json"
                    source_path = REPORTS / "e4_support_draw_raw" / f"seed{draw}" / f"source_seed42_{region}.json"
                    source = metrics_from_split(source_path, region)
                adapted = metrics_from_json(adapted_path, region)
                row = {"support_draw": draw, "mode": mode, "region": region, "region_label": LABELS[region]}
                for metric in ["iou", "dice", "bf1_2", "bf1_4", "hd95"]:
                    row[f"source_{metric}"] = source[metric]
                    row[f"adapted_{metric}"] = adapted[metric]
                    row[f"delta_{metric}"] = adapted[metric] - source[metric]
                rows.append(row)
    with (REPORTS / "e4_support_draws_per_region.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = []
    for mode in MODES:
        for region in REGIONS:
            selected = [row for row in rows if row["mode"] == mode and row["region"] == region]
            summary.append({
                "mode": mode,
                "region": region,
                "region_label": LABELS[region],
                "n_draws": len(selected),
                "delta_iou_mean": mean(row["delta_iou"] for row in selected),
                "delta_iou_std": std(row["delta_iou"] for row in selected),
                "delta_iou_min": min(row["delta_iou"] for row in selected),
                "delta_iou_max": max(row["delta_iou"] for row in selected),
                "positive_draws": sum(row["delta_iou"] > 0 for row in selected),
                "material_draws": sum(row["delta_iou"] >= 0.01 for row in selected),
                "delta_bf1_2_mean": mean(row["delta_bf1_2"] for row in selected),
                "delta_hd95_mean": mean(row["delta_hd95"] for row in selected),
            })
    with (REPORTS / "e4_support_draws_summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)

    lines = [
        "# E4-C: Support-Set Draw Robustness",
        "",
        "The source initialization is fixed to ResUNet seed42. Three deterministic 20-shot support draws are compared: model/support seed42, support seed2026, and support seed777.",
        "",
        "| Mode | Region | Delta IoU mean +/- SD | Range | Positive draws | Material draws | Delta BF1@2 | Delta HD95 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(f"| {row['mode']} | {row['region_label']} | {row['delta_iou_mean']:+.4f} +/- {row['delta_iou_std']:.4f} | [{row['delta_iou_min']:+.4f}, {row['delta_iou_max']:+.4f}] | {row['positive_draws']}/3 | {row['material_draws']}/3 | {row['delta_bf1_2_mean']:+.4f} | {row['delta_hd95_mean']:+.2f} |")
    robust = all(row["positive_draws"] >= 2 for row in summary)
    lines += [
        "",
        "## Decision",
        "",
        f"Support-draw robustness criterion: every mode-region cell has a positive gain in at least 2/3 draws. Result: {'supported' if robust else 'not supported'}.",
        "",
        "This experiment isolates support-set sampling from model initialization because all adapted runs start from the same ResUNet seed42 source checkpoint.",
    ]
    (REPORTS / "e4_support_draws_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("summary", summary, "robust", robust)


if __name__ == "__main__":
    main()
