"""Summarize E4 query-buffer, threshold, and calibration robustness."""

from __future__ import annotations

import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

import os as _repo_os
PROJECT = Path(_repo_os.environ.get("LANDSLIDE_PROJECT_ROOT", Path(__file__).resolve().parents[1])).resolve()
REPORTS = PROJECT / "reports"
sys.path.insert(0, str(PROJECT / "scripts"))
from audit_threshold_uncertainty import derived_metrics

RAW_ROOT = REPORTS / "e4_query_raw"
REGIONS = ["hokkaido_iburi_tobu", "lombok", "palu"]
LABELS = {"hokkaido_iburi_tobu": "Hokkaido", "lombok": "Lombok", "palu": "Palu"}
SEEDS = [42, 2026, 777]
BUFFERS = [0, 256, 512]
MODES = ["full", "decoder-only"]
THRESHOLDS = ["0.3", "0.4", "0.5", "0.6", "0.7"]


def mean(values):
    values = list(values)
    return float(statistics.fmean(values)) if values else float("nan")


def std(values):
    values = list(values)
    return float(statistics.stdev(values)) if len(values) > 1 else 0.0


def load_rows():
    rows = []
    for buffer in BUFFERS:
        for path in sorted((RAW_ROOT / f"buffer{buffer}").glob("*.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            label = data["checkpoint_label"]
            if label.startswith("source_"):
                role = "source"
                seed = int(label.split("_")[1].replace("seed", ""))
                mode = "source"
                run_id = f"source_seed{seed}"
            else:
                parts = label.split("_")
                role = "adapted"
                mode = parts[1]
                seed = int(parts[-1].replace("seed", ""))
                run_id = label
            region = data["region"]
            for threshold in THRESHOLDS:
                raw = data["threshold_metrics"][threshold]
                total_pixels = int(data["n_eval"]) * 128 * 128
                metrics = derived_metrics(raw, total_pixels)
                rows.append({
                    "buffer_m": buffer,
                    "role": role,
                    "mode": mode,
                    "seed": seed,
                    "run_id": run_id,
                    "region": region,
                    "region_label": LABELS[region],
                    "threshold": float(threshold),
                    "n_eval": int(data["n_eval"]),
                    "iou": metrics["iou"],
                    "mcc": metrics["mcc"],
                    "balanced_iou": metrics["balanced_iou"],
                    "high_confidence_error_rate": float(data["calibration"]["high_confidence_error_rate"]),
                })
    return rows

def main():
    rows = load_rows()
    with (REPORTS / "e4_query_metrics.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    fixed = {(row["buffer_m"], row["role"], row["mode"], row["seed"], row["region"]): row for row in rows if row["threshold"] == 0.5}
    deltas = []
    for mode in MODES:
        for buffer in BUFFERS:
            for seed in SEEDS:
                for region in REGIONS:
                    source = fixed[(buffer, "source", "source", seed, region)]
                    adapted = fixed[(buffer, "adapted", mode, seed, region)]
                    deltas.append({
                        "mode": mode,
                        "buffer_m": buffer,
                        "seed": seed,
                        "region": region,
                        "region_label": LABELS[region],
                        "source_iou": source["iou"],
                        "adapted_iou": adapted["iou"],
                        "delta_iou": adapted["iou"] - source["iou"],
                        "source_mcc": source["mcc"],
                        "adapted_mcc": adapted["mcc"],
                        "delta_mcc": adapted["mcc"] - source["mcc"],
                        "source_high_conf_error": source["high_confidence_error_rate"],
                        "adapted_high_conf_error": adapted["high_confidence_error_rate"],
                        "delta_high_conf_error": adapted["high_confidence_error_rate"] - source["high_confidence_error_rate"],
                    })
    with (REPORTS / "e4_query_deltas.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(deltas[0].keys()))
        writer.writeheader()
        writer.writerows(deltas)

    summary = []
    rng = np.random.default_rng(20260924)
    for mode in MODES:
        for buffer in BUFFERS:
            selected = [row for row in deltas if row["mode"] == mode and row["buffer_m"] == buffer]
            by_seed = defaultdict(list)
            for row in selected:
                by_seed[row["seed"]].append(row["delta_iou"])
            boot = []
            for _ in range(10000):
                sampled_seeds = rng.choice(SEEDS, size=len(SEEDS), replace=True)
                sample = []
                for seed in sampled_seeds:
                    sample.extend(rng.choice(by_seed[int(seed)], size=len(by_seed[int(seed)]), replace=True))
                boot.append(float(np.mean(sample)))
            summary.append({
                "mode": mode,
                "buffer_m": buffer,
                "mean_delta_iou": mean(row["delta_iou"] for row in selected),
                "std_delta_iou": std(row["delta_iou"] for row in selected),
                "ci95_low": float(np.percentile(boot, 2.5)),
                "ci95_high": float(np.percentile(boot, 97.5)),
                "probability_positive": float(np.mean(np.asarray(boot) > 0)),
                "positive_regions": sum(mean(row["delta_iou"] for row in selected if row["region"] == region) > 0 for region in REGIONS),
                "material_regions": sum(mean(row["delta_iou"] for row in selected if row["region"] == region) >= 0.01 for region in REGIONS),
            })
    with (REPORTS / "e4_query_summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)

    threshold_rows = []
    for mode in MODES:
        for buffer in BUFFERS:
            for threshold in [float(value) for value in THRESHOLDS]:
                selected = []
                for row in rows:
                    if row["threshold"] != threshold:
                        continue
                    lookup = {(item["role"], item["mode"], item["seed"], item["region"]): item for item in rows if item["buffer_m"] == buffer and item["threshold"] == threshold}
                    source = lookup[("source", "source", row["seed"], row["region"])]
                    if row["role"] == "adapted" and row["mode"] == mode and row["buffer_m"] == buffer:
                        selected.append(row["iou"] - source["iou"])
                threshold_rows.append({
                    "mode": mode,
                    "buffer_m": buffer,
                    "threshold": threshold,
                    "mean_delta_iou": mean(selected),
                    "positive_cases": sum(value > 0 for value in selected),
                    "material_cases": sum(value >= 0.01 for value in selected),
                    "n_cases": len(selected),
                })
    with (REPORTS / "e4_query_threshold_summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(threshold_rows[0].keys()))
        writer.writeheader()
        writer.writerows(threshold_rows)

    lines = [
        "# E4-A/B: Query-Buffer, Threshold, and Calibration Robustness",
        "",
        "The same ResUNet source and 20-shot adapted checkpoints were evaluated under the existing guard split and GSD-aware 256 m/512 m physical buffers.",
        "",
        "## Query-Buffer Adaptation Gain",
        "",
        "| Mode | Buffer (m) | Mean Delta IoU | 95% interval | P(delta>0) | Positive regions | Material regions |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(f"| {row['mode']} | {row['buffer_m']} | {row['mean_delta_iou']:+.4f} | [{row['ci95_low']:+.4f}, {row['ci95_high']:+.4f}] | {row['probability_positive']:.3f} | {row['positive_regions']}/3 | {row['material_regions']}/3 |")
    lines += [
        "",
        "## Threshold Robustness",
        "",
        "| Mode | Buffer (m) | Threshold | Mean Delta IoU | Positive cases | Material cases |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in threshold_rows:
        lines.append(f"| {row['mode']} | {row['buffer_m']} | {row['threshold']:.1f} | {row['mean_delta_iou']:+.4f} | {row['positive_cases']}/{row['n_cases']} | {row['material_cases']}/{row['n_cases']} |")
    lines += [
        "",
        "## Calibration",
        "",
        "| Mode | Buffer (m) | Source high-confidence error | Adapted high-confidence error | Delta |",
        "|---|---:|---:|---:|---:|",
    ]
    for mode in MODES:
        for buffer in BUFFERS:
            selected = [row for row in deltas if row["mode"] == mode and row["buffer_m"] == buffer]
            lines.append(f"| {mode} | {buffer} | {mean(row['source_high_conf_error'] for row in selected):.4f} | {mean(row['adapted_high_conf_error'] for row in selected):.4f} | {mean(row['delta_high_conf_error'] for row in selected):+.4f} |")
    lines += [
        "",
        "## Decision Rule",
        "",
        "Adaptation robustness is supported only if the positive IoU gain persists under 256/512 m buffers, remains sign-consistent across most region-seed cases, and does not come with a systematic calibration deterioration.",
    ]
    (REPORTS / "e4_query_robustness_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("rows", len(rows), "summary", summary)


if __name__ == "__main__":
    main()
