"""Summarize stitched full-map E4 evaluation."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import os as _repo_os
PROJECT = Path(_repo_os.environ.get("LANDSLIDE_PROJECT_ROOT", Path(__file__).resolve().parents[1])).resolve()
RAW = PROJECT / "reports" / "e4_stitched_raw" / "seed42"
REPORTS = PROJECT / "reports"
REGIONS = ["hokkaido_iburi_tobu", "lombok", "palu"]
MODES = ["full", "decoder-only"]
LABELS = {"hokkaido_iburi_tobu": "Hokkaido", "lombok": "Lombok", "palu": "Palu"}


def load(label, region):
    data = json.loads((RAW / f"{label}_{region}.json").read_text(encoding="utf-8"))
    return data["metrics"]


def main():
    rows = []
    for mode in MODES:
        for region in REGIONS:
            source = load("source_seed42", region)
            adapted = load(f"adapted_{mode}_seed42", region)
            row = {"mode": mode, "region": region, "region_label": LABELS[region]}
            for metric in ["iou", "dice", "precision", "recall", "accuracy"]:
                row[f"source_{metric}"] = source[metric]
                row[f"adapted_{metric}"] = adapted[metric]
                row[f"delta_{metric}"] = adapted[metric] - source[metric]
            rows.append(row)
    with (REPORTS / "e4_stitched_summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# E4-D: Stitched Full-Map Check",
        "",
        "Seed 42, ResUNet. Overlapping 512-pixel query chips were stitched with averaged probabilities according to the reconstructed stride-256 grid. Metrics are computed on the union of query-chip pixels, excluding support and guard chips.",
        "",
        "| Mode | Region | Source IoU | Stitched adapted IoU | Delta IoU | Delta precision | Delta recall |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(f"| {row['mode']} | {row['region_label']} | {row['source_iou']:.4f} | {row['adapted_iou']:.4f} | {row['delta_iou']:+.4f} | {row['delta_precision']:+.4f} | {row['delta_recall']:+.4f} |")
    lines += [
        "",
        "## Decision",
        "",
        "The stitched-map check supports the same direction as the tile-level evaluation if the adapted map improves IoU without a precision collapse. It remains a query-covered stitched evaluation, not a claim about unsurveyed terrain.",
    ]
    (REPORTS / "e4_stitched_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(rows)


if __name__ == "__main__":
    main()
