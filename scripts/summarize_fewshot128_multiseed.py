"""Summarize the three-seed selected few-shot adaptation configurations."""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

import os as _repo_os
PROJECT = Path(_repo_os.environ.get("LANDSLIDE_PROJECT_ROOT", Path(__file__).resolve().parents[1])).resolve()
OUTPUTS = PROJECT / "outputs"
REPORTS = PROJECT / "reports"
OUT_PREFIX = "benchmark_v2_fewshot128_multiseed"
SEEDS = [42, 2026, 777]
REGIONS = ["hokkaido_iburi_tobu", "lombok", "palu"]
REGION_LABELS = {"hokkaido_iburi_tobu": "Hokkaido", "lombok": "Lombok", "palu": "Palu"}
SELECTED = [
    ("ResUNet", "full", 5),
    ("ResUNet", "full", 10),
    ("ResUNet", "full", 20),
    ("ResUNet", "decoder-only", 20),
    ("SegFormerB0", "full", 10),
    ("SegFormerB0", "full", 20),
]
METRICS = ["iou", "dice", "bf1_2", "bf1_4", "hd95"]


def read_rows() -> list[dict]:
    rows = []
    for path in sorted(OUTPUTS.glob("bench_v2_fewshot128_*/metrics.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        region = data["region"]
        seed = int(data["seed"])
        if seed not in SEEDS:
            continue
        group = data["groups"][region]
        threshold = group["threshold_metrics"]["0.5"]
        rows.append({
            "model": data["model"],
            "mode": data["mode"],
            "region": region,
            "region_label": REGION_LABELS[region],
            "seed": seed,
            "shot": 0 if data["mode"] == "zero-shot" else int(data["shot"]),
            "n_eval": int(data["n_eval"]),
            "iou": float(threshold["iou"]),
            "dice": float(threshold["dice"]),
            "bf1_2": float(group["f1_2_mean"]),
            "bf1_4": float(group["f1_4_mean"]),
            "hd95": float(group["hd95_mean"]),
            "run_dir": str(path.parent),
        })
    return rows


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def mean(values) -> float:
    return statistics.fmean(float(value) for value in values)


def std(values) -> float:
    return statistics.stdev(float(value) for value in values) if len(values) > 1 else 0.0


def fmt(value, digits=4) -> str:
    return f"{float(value):.{digits}f}"
def main() -> None:
    rows = read_rows()
    baseline = {(row["model"], row["seed"], row["region"]): row for row in rows if row["mode"] == "zero-shot"}
    required = [(model, "zero-shot", 0) for model in sorted({item[0] for item in SELECTED})] + SELECTED
    missing = []
    for model, mode, shot in required:
        for seed in SEEDS:
            for region in REGIONS:
                if not any(row["model"] == model and row["mode"] == mode and row["shot"] == shot and row["seed"] == seed and row["region"] == region for row in rows):
                    missing.append((model, mode, shot, seed, region))
    if missing:
        raise RuntimeError(f"Missing runs: {missing[:10]}")
    for row in rows:
        base = baseline[(row["model"], row["seed"], row["region"])]
        for metric in METRICS:
            row[f"delta_{metric}"] = row[metric] - base[metric]
    write_csv(REPORTS / f"{OUT_PREFIX}_per_run.csv", rows, list(rows[0].keys()))

    macro_rows = []
    region_rows = []
    for model, mode, shot in required:
        entries = [row for row in rows if row["model"] == model and row["mode"] == mode and row["shot"] == shot]
        macro_record = {"model": model, "mode": mode, "shot": shot, "n_seeds": len(SEEDS)}
        seed_macro_deltas = []
        for seed in SEEDS:
            seed_entries = [row for row in entries if row["seed"] == seed]
            delta = mean(row["delta_iou"] for row in seed_entries)
            seed_macro_deltas.append(delta)
            for metric in METRICS:
                macro_record[f"seed{seed}_{metric}"] = mean(row[metric] for row in seed_entries)
        for metric in METRICS:
            values = [macro_record[f"seed{seed}_{metric}"] for seed in SEEDS]
            macro_record[f"macro_{metric}_mean"] = mean(values)
            macro_record[f"macro_{metric}_std"] = std(values)
        macro_record["macro_delta_iou_mean"] = mean(seed_macro_deltas)
        macro_record["macro_delta_iou_std"] = std(seed_macro_deltas)
        macro_record["positive_seeds"] = sum(delta > 0 for delta in seed_macro_deltas)
        positive_regions = 0
        for region in REGIONS:
            region_record = {"model": model, "mode": mode, "shot": shot, "region": region, "region_label": REGION_LABELS[region], "n_seeds": len(SEEDS)}
            region_entries = [row for row in entries if row["region"] == region]
            for metric in METRICS:
                values = [row[metric] for row in region_entries]
                deltas = [row[f"delta_{metric}"] for row in region_entries]
                region_record[f"{metric}_mean"] = mean(values)
                region_record[f"{metric}_std"] = std(values)
                region_record[f"delta_{metric}_mean"] = mean(deltas)
                region_record[f"delta_{metric}_std"] = std(deltas)
            region_record["positive_seed_count_iou"] = sum(row["delta_iou"] > 0 for row in region_entries)
            positive_regions += int(region_record["delta_iou_mean"] > 0)
            region_rows.append(region_record)
        macro_record["positive_regions"] = positive_regions
        macro_record["effective"] = macro_record["macro_delta_iou_mean"] > 0 and macro_record["positive_seeds"] >= 2 and positive_regions >= 2
        macro_record["strict_support"] = macro_record["effective"] and macro_record["positive_seeds"] == len(SEEDS) and positive_regions == len(REGIONS)
        macro_rows.append(macro_record)
    write_csv(REPORTS / f"{OUT_PREFIX}_macro.csv", macro_rows, list(macro_rows[0].keys()))
    write_csv(REPORTS / f"{OUT_PREFIX}_by_region.csv", region_rows, list(region_rows[0].keys()))
    adapted = [row for row in macro_rows if row["shot"] > 0]
    adapted.sort(key=lambda row: row["macro_delta_iou_mean"], reverse=True)
    lines = [
        "# Benchmark v2: Three-Seed Few-Shot Adaptation Summary",
        "",
        "Seeds: 42, 2026, 777. Each seed uses its own deterministic nested support set and leakage-guarded query set; training and evaluation protocols are otherwise identical.",
        "",
        "## Selected Configuration Summary",
        "",
        "| Model | Mode | Shot | Macro IoU (mean +/- SD) | Delta IoU | Positive seeds | Positive regions | Decision |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in adapted:
        lines.append(
            f"| {row['model']} | {row['mode']} | {row['shot']} | {fmt(row['macro_iou_mean'])} +/- {fmt(row['macro_iou_std'])} | "
            f"{row['macro_delta_iou_mean']:+.4f} +/- {row['macro_delta_iou_std']:.4f} | {row['positive_seeds']}/3 | "
            f"{row['positive_regions']}/3 | {'cross-region support' if row['strict_support'] else ('partial support' if row['effective'] else 'not supported')} |"
        )
    lines += [
        "",
        "## Per-Region IoU Delta",
        "",
        "| Model | Mode | Shot | Hokkaido | Lombok | Palu |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for macro in adapted:
        records = {row["region"]: row for row in region_rows if row["model"] == macro["model"] and row["mode"] == macro["mode"] and row["shot"] == macro["shot"]}
        lines.append(
            f"| {macro['model']} | {macro['mode']} | {macro['shot']} | "
            f"{records['hokkaido_iburi_tobu']['delta_iou_mean']:+.4f} +/- {records['hokkaido_iburi_tobu']['delta_iou_std']:.4f} | "
            f"{records['lombok']['delta_iou_mean']:+.4f} +/- {records['lombok']['delta_iou_std']:.4f} | "
            f"{records['palu']['delta_iou_mean']:+.4f} +/- {records['palu']['delta_iou_std']:.4f} |"
        )
    strict = [row for row in adapted if row["strict_support"]]
    lines += [
        "",
        "## Cross-Region Supported Configuration Detail",
        "",
        "| Model | Mode | Shot | Region | Zero-shot IoU | Adapted IoU | Delta IoU | Zero-shot BF1@2 | Adapted BF1@2 | Zero-shot HD95 | Adapted HD95 |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for macro in strict:
        records = {row["region"]: row for row in region_rows if row["model"] == macro["model"] and row["mode"] == macro["mode"] and row["shot"] == macro["shot"]}
        baseline_records = {row["region"]: row for row in region_rows if row["model"] == macro["model"] and row["mode"] == "zero-shot" and row["shot"] == 0}
        for region in REGIONS:
            current = records[region]
            base = baseline_records[region]
            lines.append(
                f"| {macro['model']} | {macro['mode']} | {macro['shot']} | {REGION_LABELS[region]} | "
                f"{fmt(base['iou_mean'])} | {fmt(current['iou_mean'])} | {current['delta_iou_mean']:+.4f} | "
                f"{fmt(base['bf1_2_mean'])} | {fmt(current['bf1_2_mean'])} | {fmt(base['hd95_mean'], 2)} | {fmt(current['hd95_mean'], 2)} |"
            )
    lines += [
        "",
        "## Decision Rule",
        "",
        "`Partial support` requires a positive mean macro-IoU gain, at least two of three seeds improving, and at least two of three regions improving on average. `Cross-region support` additionally requires all three seeds and all three regions to improve. Other configurations remain exploratory or negative evidence.",
    ]
    (REPORTS / f"{OUT_PREFIX}_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("selected configs", len(adapted))
    print("supported configs", sum(row["effective"] for row in adapted))


if __name__ == "__main__":
    main()
