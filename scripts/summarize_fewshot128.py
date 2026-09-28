"""Summarize target-domain few-shot adaptation runs."""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

import os as _repo_os
PROJECT = Path(_repo_os.environ.get("LANDSLIDE_PROJECT_ROOT", Path(__file__).resolve().parents[1])).resolve()
OUTPUTS = PROJECT / "outputs"
REPORTS = PROJECT / "reports"
OUT_PREFIX = "benchmark_v2_fewshot128"
REGIONS = ["hokkaido_iburi_tobu", "lombok", "palu"]
REGION_LABELS = {
    "hokkaido_iburi_tobu": "Hokkaido",
    "lombok": "Lombok",
    "palu": "Palu",
}
METRICS = ["iou", "dice", "bf1_2", "bf1_4", "hd95"]


def metric_values(path: Path) -> dict | None:
    data = json.loads(path.read_text(encoding="utf-8"))
    if int(data["seed"]) != 42:
        return None
    region = data["region"]
    group = data["groups"][region]
    threshold = group["threshold_metrics"]["0.5"]
    record = {
        "model": data["model"],
        "mode": data["mode"],
        "region": region,
        "region_label": REGION_LABELS[region],
        "seed": int(data["seed"]),
        "shot": 0 if data["mode"] == "zero-shot" else int(data["shot"]),
        "n_support": int(data["n_support"]),
        "n_eval": int(data["n_eval"]),
        "max_steps": int(data["max_steps"]),
        "run_dir": str(path.parent),
        "iou": float(threshold["iou"]),
        "dice": float(threshold["dice"]),
        "bf1_2": float(group["f1_2_mean"]),
        "bf1_4": float(group["f1_4_mean"]),
        "hd95": float(group["hd95_mean"]),
    }
    return record


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def mean(values) -> float:
    return statistics.fmean(float(value) for value in values)


def fmt(value, digits=4) -> str:
    return "NA" if value is None else f"{float(value):.{digits}f}"


def main() -> None:
    rows = []
    for path in sorted(OUTPUTS.glob("bench_v2_fewshot128_*/metrics.json")):
        record = metric_values(path)
        if record is not None:
            rows.append(record)
    if not rows:
        raise RuntimeError("No few-shot metrics found")

    baseline = {
        (row["model"], row["region"]): row
        for row in rows
        if row["mode"] == "zero-shot"
    }
    for row in rows:
        base = baseline[(row["model"], row["region"])]
        for metric in METRICS:
            row[f"delta_{metric}"] = row[metric] - base[metric]

    fields = list(rows[0].keys())
    write_csv(REPORTS / f"{OUT_PREFIX}_per_run.csv", rows, fields)

    region_rows = []
    for model in sorted({row["model"] for row in rows}):
        for mode in ["zero-shot", "full", "decoder-only"]:
            for shot in [0, 5, 10, 20]:
                entries = [
                    row for row in rows
                    if row["model"] == model and row["mode"] == mode and row["shot"] == shot
                ]
                if not entries:
                    continue
                record = {
                    "model": model,
                    "mode": mode,
                    "shot": shot,
                    "n_regions": len(entries),
                }
                for metric in METRICS + [f"delta_{metric}" for metric in METRICS]:
                    record[f"macro_{metric}"] = mean(row[metric] for row in entries)
                region_rows.append(record)
    write_csv(REPORTS / f"{OUT_PREFIX}_macro_by_config.csv", region_rows, list(region_rows[0].keys()))

    per_region_rows = []
    for row in sorted(rows, key=lambda item: (item["region"], item["model"], item["mode"], item["shot"])):
        per_region_rows.append(row)
    write_csv(REPORTS / f"{OUT_PREFIX}_by_region.csv", per_region_rows, fields)

    adapted = [row for row in region_rows if row["shot"] > 0]
    ranked = sorted(adapted, key=lambda row: (row["macro_delta_iou"], row["macro_iou"]), reverse=True)
    top = ranked[:8]

    lines = [
        "# Benchmark v2: 128x128 Target-Domain Few-Shot Adaptation",
        "",
        "Protocol: seed 42; nested 5/10/20-shot support sets; one common leakage-guarded query set per region; 400 fixed optimizer steps; batch size 4; identical augmentation and learning rate for full and decoder-only fine-tuning.",
        "",
        "## Macro Results Across Hokkaido, Lombok, and Palu",
        "",
        "| Model | Mode | Shot | Macro IoU | Delta IoU | Macro BF1@2 | Delta BF1@2 | Macro HD95 | Delta HD95 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in region_rows:
        lines.append(
            f"| {row['model']} | {row['mode']} | {row['shot']} | {fmt(row['macro_iou'])} | "
            f"{row['macro_delta_iou']:+.4f} | {fmt(row['macro_bf1_2'])} | {row['macro_delta_bf1_2']:+.4f} | "
            f"{fmt(row['macro_hd95'], 2)} | {row['macro_delta_hd95']:+.2f} |"
        )

    lines += [
        "",
        "## Seed-42 Candidate Configs for Stage 2",
        "",
        "| Rank | Model | Mode | Shot | Macro IoU | Delta IoU | Delta BF1@2 | Delta HD95 |",
        "|---:|---|---|---:|---:|---:|---:|---:|",
    ]
    for rank, row in enumerate(top, start=1):
        lines.append(
            f"| {rank} | {row['model']} | {row['mode']} | {row['shot']} | {fmt(row['macro_iou'])} | "
            f"{row['macro_delta_iou']:+.4f} | {row['macro_delta_bf1_2']:+.4f} | {row['macro_delta_hd95']:+.2f} |"
        )

    lines += [
        "",
        "## Interpretation Rules",
        "",
        "- Candidate ranking uses macro IoU gain over the same-query zero-shot baseline, with absolute macro IoU as the tie-breaker.",
        "- Do not promote a configuration based only on one event; inspect the per-region CSV and require cross-event consistency before adding seeds 2026 and 777.",
        "- All metrics are computed at 128x128 on a common query set, so Hokkaido, Lombok, and Palu are directly comparable across shots and modes.",
    ]
    (REPORTS / f"{OUT_PREFIX}_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("rows", len(rows), "configs", len(region_rows))
    print("top:", [(row["model"], row["mode"], row["shot"], row["macro_delta_iou"]) for row in top])


if __name__ == "__main__":
    main()
