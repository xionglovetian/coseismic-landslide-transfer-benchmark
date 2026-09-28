"""Summarize E2 early-vs-late source checkpoint few-shot adaptation."""

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
METRICS = ["iou", "dice", "bf1_2", "bf1_4", "hd95"]


def mean(values):
    values = list(values)
    return float(statistics.fmean(values)) if values else float("nan")


def read_run(path: Path, init_epoch: int, mode: str, region: str):
    data = json.loads(path.read_text(encoding="utf-8"))
    group = data["groups"][region]
    threshold = group["threshold_metrics"]["0.5"]
    return {
        "init_epoch": init_epoch,
        "mode": mode,
        "region": region,
        "region_label": LABELS[region],
        "iou": float(threshold["iou"]),
        "dice": float(threshold["dice"]),
        "bf1_2": float(group["f1_2_mean"]),
        "bf1_4": float(group["f1_4_mean"]),
        "hd95": float(group["hd95_mean"]),
    }


def main() -> None:
    rows = []
    for mode in MODES:
        for region in REGIONS:
            baseline_path = OUTPUTS / f"bench_v2_fewshot128_resunet_{region}_zeroshot_seed42" / "metrics.json"
            baseline = read_run(baseline_path, 0, mode, region)
            for init_epoch in [20, 30, 50]:
                if init_epoch == 50:
                    path = OUTPUTS / f"bench_v2_fewshot128_resunet_{region}_20shot_{mode}_seed42" / "metrics.json"
                else:
                    path = OUTPUTS / f"bench_v2_e2_epoch{init_epoch}_resunet_{region}_20shot_{mode}_seed42" / "metrics.json"
                row = read_run(path, init_epoch, mode, region)
                for metric in METRICS:
                    row[f"delta_{metric}"] = row[metric] - baseline[metric]
                rows.append(row)
    with (REPORTS / "e2_early_late_per_region.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    macro_rows = []
    for mode in MODES:
        for init_epoch in [20, 30, 50]:
            entries = [row for row in rows if row["mode"] == mode and row["init_epoch"] == init_epoch]
            record = {"mode": mode, "init_epoch": init_epoch}
            for metric in METRICS + [f"delta_{metric}" for metric in METRICS]:
                record[f"macro_{metric}"] = mean(row[metric] for row in entries)
            macro_rows.append(record)
    with (REPORTS / "e2_early_late_macro.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(macro_rows[0].keys()))
        writer.writeheader()
        writer.writerows(macro_rows)

    index = {(row["mode"], row["init_epoch"], row["region"]): row for row in rows}
    contrasts = []
    for mode in MODES:
        for early in [20, 30]:
            for region in REGIONS + ["macro"]:
                if region == "macro":
                    delta = mean(index[(mode, early, item)]["iou"] for item in REGIONS) - mean(index[(mode, 50, item)]["iou"] for item in REGIONS)
                else:
                    delta = index[(mode, early, region)]["iou"] - index[(mode, 50, region)]["iou"]
                contrasts.append({"mode": mode, "early_epoch": early, "comparison": f"epoch{early}-epoch50", "region": region, "delta_iou": delta})
    with (REPORTS / "e2_early_late_contrasts.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(contrasts[0].keys()))
        writer.writeheader()
        writer.writerows(contrasts)

    lines = [
        "# E2: Early-vs-Late Source Checkpoint Adaptation",
        "",
        "Model: ResUNet, seed 42, 20-shot adaptation, 400 optimizer steps. The epoch-50 results are the same P3 runs; epoch-20 and epoch-30 results are the new E2 runs.",
        "",
        "## Macro Results",
        "",
        "| Mode | Init epoch | Macro IoU | Delta vs zero-shot | Macro BF1@2 | Macro HD95 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in macro_rows:
        lines.append(f"| {row['mode']} | {row['init_epoch']} | {row['macro_iou']:.4f} | {row['macro_delta_iou']:+.4f} | {row['macro_bf1_2']:.4f} | {row['macro_hd95']:.2f} |")
    lines += ["", "## Early-minus-Late Contrasts", "", "| Mode | Contrast | Region | Delta IoU |", "|---|---|---|---:|"]
    for row in contrasts:
        lines.append(f"| {row['mode']} | {row['comparison']} | {row['region']} | {row['delta_iou']:+.4f} |")
    lines += [
        "",
        "## Decision",
        "",
        "No early checkpoint meets the pre-specified criterion of improving at least two of three target regions and reaching a +0.01 Macro IoU gain relative to epoch 50. Epoch-30 full fine-tuning is slightly higher on Macro IoU (+0.0062), but the gain is driven by Hokkaido while Lombok is unchanged and Palu is +0.0016. Decoder-only early initialization does not beat epoch 50. The current evidence therefore does not show that early stopping improves few-shot adaptation; it weakens the causal claim that source overfitting materially harms target adaptation.",
        "",
        "## Decision Rule",
        "",
        "Source overfitting is causally harmful for adaptation only if an early checkpoint improves at least two of three target regions and reaches a Macro IoU gain of at least 0.01 relative to epoch 50 under the same adaptation protocol.",
        "",
        "This is a seed-42 screening experiment; a positive result requires seed2026/777 confirmation before making a general claim.",
    ]
    (REPORTS / "e2_early_late_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(macro_rows, ensure_ascii=False, indent=2))
    print(json.dumps(contrasts, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
