"""Summarize E1 exposure-matched source-expansion runs."""

from __future__ import annotations

import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np

import os as _repo_os
PROJECT = Path(_repo_os.environ.get("LANDSLIDE_PROJECT_ROOT", Path(__file__).resolve().parents[1])).resolve()
RAW_ROOT = PROJECT / "reports" / "e1_raw"
REPORTS = PROJECT / "reports"
REGIMES = ["cas-single", "cas-multistream", "pooled"]
SEEDS = [42, 2026, 777]
REGIONS = ["hokkaido_iburi_tobu", "lombok", "palu"]
LABELS = {"hokkaido_iburi_tobu": "Hokkaido", "lombok": "Lombok", "palu": "Palu"}


def mean(values):
    values = list(values)
    return float(statistics.fmean(values)) if values else float("nan")


def std(values):
    values = list(values)
    return float(statistics.stdev(values)) if len(values) > 1 else 0.0


def main() -> None:
    rows = []
    for regime in REGIMES:
        for seed in SEEDS:
            run_dir = RAW_ROOT / f"bench_v2_e1_{regime}_seed{seed}"
            for path in sorted(run_dir.glob("step_*.json")):
                data = json.loads(path.read_text(encoding="utf-8"))
                record = {
                    "regime": regime,
                    "seed": seed,
                    "step": int(data["step"]),
                    "source_val_iou": float(data["source_validation"]["iou"]),
                    "source_val_dice": float(data["source_validation"]["dice"]),
                }
                target_values = []
                for region in REGIONS:
                    value = float(data["target"][region]["threshold_metrics"]["0.5"]["iou"])
                    record[f"{region}_iou"] = value
                    target_values.append(value)
                record["target_macro_iou"] = mean(target_values)
                rows.append(record)
    if not rows:
        raise RuntimeError(f"No E1 results in {RAW_ROOT}")
    rows.sort(key=lambda row: (row["regime"], row["seed"], row["step"]))
    with (REPORTS / "e1_exposure_matched_per_run.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = []
    for regime in REGIMES:
        for step in sorted({row["step"] for row in rows}):
            entries = [row for row in rows if row["regime"] == regime and row["step"] == step]
            record = {"regime": regime, "step": step, "n_seeds": len(entries)}
            for metric in ["source_val_iou", "target_macro_iou"] + [f"{region}_iou" for region in REGIONS]:
                record[f"{metric}_mean"] = mean(row[metric] for row in entries)
                record[f"{metric}_std"] = std(row[metric] for row in entries)
            summary.append(record)
    with (REPORTS / "e1_exposure_matched_summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)

    index = {(row["regime"], row["seed"], row["step"]): row for row in rows}
    contrasts = {}
    for contrast in [("pooled", "cas-multistream"), ("pooled", "cas-single")]:
        after, before = contrast
        records = []
        for seed in SEEDS:
            for step in sorted({row["step"] for row in rows}):
                a = index[(after, seed, step)]
                b = index[(before, seed, step)]
                for metric in [f"{region}_iou" for region in REGIONS] + ["target_macro_iou"]:
                    key = metric[:-4] if metric != "target_macro_iou" else "target_macro"
                    records.append({"contrast": f"{after}-{before}", "seed": seed, "step": step, "region": key, "delta_iou": a[metric] - b[metric]})
        contrasts[f"{after}-{before}"] = records
    all_contrasts = [record for records in contrasts.values() for record in records]
    with (REPORTS / "e1_exposure_matched_contrasts.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_contrasts[0].keys()))
        writer.writeheader()
        writer.writerows(all_contrasts)

    final_step = max(row["step"] for row in rows)
    bootstrap = {}
    region_bootstrap = []
    rng = np.random.default_rng(20260924)
    for name, records in contrasts.items():
        for region_name in REGIONS + ["target_macro"]:
            fixed = [row for row in records if row["step"] == final_step and row["region"] == region_name]
            by_seed = defaultdict(list)
            for row in fixed:
                by_seed[row["seed"]].append(row["delta_iou"])
            values = []
            for _ in range(10000):
                sampled_seeds = rng.choice(SEEDS, size=len(SEEDS), replace=True)
                sample = []
                for seed in sampled_seeds:
                    sample.extend(rng.choice(by_seed[int(seed)], size=len(by_seed[int(seed)]), replace=True))
                values.append(float(np.mean(sample)))
            result = {
                "mean_delta": mean(row["delta_iou"] for row in fixed),
                "ci95_low": float(np.percentile(values, 2.5)),
                "ci95_high": float(np.percentile(values, 97.5)),
                "probability_positive": float(np.mean(np.asarray(values) > 0)),
                "probability_material": float(np.mean(np.asarray(values) >= 0.01)),
            }
            if region_name == "target_macro":
                bootstrap[name] = result
            region_bootstrap.append({"contrast": name, "region": region_name, **result})
    with (REPORTS / "e1_exposure_matched_bootstrap.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(region_bootstrap[0].keys()))
        writer.writeheader()
        writer.writerows(region_bootstrap)

    lines = [
        "# E1: Exposure-Matched Source Expansion",
        "",
        "All regimes use 1,120 optimizer steps, batch size 32, 35,840 image exposures per run, and a step-based cosine schedule. Pooled and CAS multi-stream replay both use five domain slots.",
        "",
        "## Final-Step Summary",
        "",
        "| Regime | Seed count | CAS val IoU | Target Macro IoU | Hokkaido IoU | Lombok IoU | Palu IoU |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for regime in REGIMES:
        row = next(item for item in summary if item["regime"] == regime and item["step"] == final_step)
        lines.append(f"| {regime} | {row['n_seeds']} | {row['source_val_iou_mean']:.4f} | {row['target_macro_iou_mean']:.4f} | {row['hokkaido_iburi_tobu_iou_mean']:.4f} | {row['lombok_iou_mean']:.4f} | {row['palu_iou_mean']:.4f} |")
    lines += ["", "## Final-Step Contrasts", "", "| Contrast | Mean Delta Macro IoU | 95% Interval | P(delta>0) | P(delta>=0.01) |", "|---|---:|---:|---:|---:|"]
    for name, values in bootstrap.items():
        lines.append(f"| {name} | {values['mean_delta']:+.4f} | [{values['ci95_low']:+.4f}, {values['ci95_high']:+.4f}] | {values['probability_positive']:.3f} | {values['probability_material']:.3f} |")
    lines += [
        "",
        "## Final-Step Per-Region Contrasts",
        "",
        "| Contrast | Region | Mean Delta IoU | 95% Interval | P(delta>0) | P(delta>=0.01) |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in region_bootstrap:
        lines.append(f"| {row['contrast']} | {row['region']} | {row['mean_delta']:+.4f} | [{row['ci95_low']:+.4f}, {row['ci95_high']:+.4f}] | {row['probability_positive']:.3f} | {row['probability_material']:.3f} |")
    lines += [
        "",
        "## Per-Seed Added-Region Deltas",
        "",
        "| Seed | Hokkaido | Lombok | Palu | Macro |",
        "|---:|---:|---:|---:|---:|",
    ]
    main_records = contrasts["pooled-cas-multistream"]
    for seed in SEEDS:
        values = {row["region"]: row["delta_iou"] for row in main_records if row["seed"] == seed and row["step"] == final_step}
        lines.append(f"| {seed} | {values['hokkaido_iburi_tobu']:+.4f} | {values['lombok']:+.4f} | {values['palu']:+.4f} | {values['target_macro']:+.4f} |")
    lines += [
        "",
        "## Decision",
        "",
        "The exposure-matched pooled-source regime has a positive Macro IoU effect relative to the CAS multi-stream replay placebo. However, the effect is strongly region-dependent: Hokkaido has a large material gain, Palu has a small positive gain below the pre-specified 0.01 material threshold, and Lombok is effectively unchanged. The causal conclusion is therefore `added regions can improve aggregate transfer in this benchmark, but the effect is not uniformly positive across target regions`.",
        "",
        "## Interpretation Rule",
        "",
        "The causal added-region contrast is `pooled - cas-multistream`, because both have five domain slots and identical exposure. `pooled - cas-single` measures the combined effect of added regions and multi-stream exposure.",
        "",
        "Do not claim a stable added-region effect unless the pooled-minus-placebo contrast has a positive interval, consistent event signs, and an effect of practical size.",
    ]
    (REPORTS / "e1_exposure_matched_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("rows", len(rows), "final_step", final_step, "bootstrap", bootstrap)


if __name__ == "__main__":
    main()
