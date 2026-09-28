import argparse
import csv
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import os as _repo_os
PROJECT = Path(_repo_os.environ.get("LANDSLIDE_PROJECT_ROOT", Path(__file__).resolve().parents[1])).resolve()
RAW_DIR = PROJECT / "reports" / "threshold_sensitivity_raw"
REPORTS = PROJECT / "reports"
FIGURES = PROJECT / "figures"
SEEDS = [42, 2026, 777]
REGIONS = ["a1", "a2", "a3", "b", "c1", "c2", "c3"]
THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7]
MODELS = ["UNet", "NestedUNet", "AS_UNet", "U-Net++", "ASK-UNet++", "Bottleneck-LiteASK"]
RUN_PREFIX = {
    "UNet": "group_unet",
    "NestedUNet": "group_nestedunet",
    "AS_UNet": "group_asunet32",
    "U-Net++": "group_unetpp",
    "ASK-UNet++": "group_askunetpp",
    "Bottleneck-LiteASK": "group_bottleneckliteaskunetpp",
}


def load_records(allow_incomplete):
    records, missing = [], []
    for model in MODELS:
        for seed in SEEDS:
            path = RAW_DIR / f"{RUN_PREFIX[model]}_seed{seed}.json"
            if path.exists():
                records.append(json.loads(path.read_text(encoding="utf-8")))
            else:
                missing.append(str(path))
    if missing and not allow_incomplete:
        raise SystemExit("Missing threshold results:\n" + "\n".join(missing))
    return records


def write_seed_table(records):
    rows = []
    for record in records:
        for region in REGIONS:
            for threshold in THRESHOLDS:
                values = record["groups"][region][str(threshold)]
                rows.append({
                    "model": record["model"],
                    "seed": record["seed"],
                    "group": region,
                    "threshold": threshold,
                    "iou": values["iou"],
                    "dice": values["dice"],
                    "precision": values["precision"],
                    "recall": values["recall"],
                })
    fields = ["model", "seed", "group", "threshold", "iou", "dice", "precision", "recall"]
    with (REPORTS / "threshold_sensitivity_seed_group_results.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def model_threshold_summary(rows):
    output = []
    for model in MODELS:
        for threshold in THRESHOLDS:
            subset = [row for row in rows if row["model"] == model and row["threshold"] == threshold]
            seed_group_means = []
            for seed in SEEDS:
                values = [row["iou"] for row in subset if row["seed"] == seed]
                if values:
                    seed_group_means.append(statistics.mean(values))
            output.append({
                "model": model,
                "threshold": threshold,
                "macro_iou_mean": statistics.mean(seed_group_means),
                "macro_iou_std_seed": statistics.stdev(seed_group_means) if len(seed_group_means) > 1 else 0.0,
                "macro_dice_mean": statistics.mean(row["dice"] for row in subset),
                "macro_precision": statistics.mean(row["precision"] for row in subset),
                "macro_recall": statistics.mean(row["recall"] for row in subset),
            })
    fields = ["model", "threshold", "macro_iou_mean", "macro_iou_std_seed", "macro_dice_mean", "macro_precision", "macro_recall"]
    with (REPORTS / "threshold_sensitivity_model_summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output)
    return output


def group_threshold_summary(rows):
    output = []
    for model in MODELS:
        for region in REGIONS:
            for threshold in THRESHOLDS:
                values = [row["iou"] for row in rows if row["model"] == model and row["group"] == region and row["threshold"] == threshold]
                output.append({
                    "model": model,
                    "group": region,
                    "threshold": threshold,
                    "iou_mean": statistics.mean(values),
                    "iou_std": statistics.stdev(values) if len(values) > 1 else 0.0,
                })
    return output


def write_figure(model_summary):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    panels = [
        ("macro_iou_mean", "Macro IoU across 7 groups"),
        ("macro_precision", "Macro Precision across 7 groups"),
        ("macro_recall", "Macro Recall across 7 groups"),
    ]
    for ax, (key, title) in zip(axes, panels):
        for model in MODELS:
            values = [next(row[key] for row in model_summary if row["model"] == model and row["threshold"] == threshold) for threshold in THRESHOLDS]
            ax.plot(THRESHOLDS, values, marker="o", label=model)
        ax.set_title(title)
        ax.set_xlabel("Threshold")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle("Threshold Sensitivity on Original ResUNet-BFA Images", fontsize=14)
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "threshold_sensitivity_summary.png", dpi=200)
    plt.close(fig)


def write_report(records, model_summary, group_summary):
    lines = [
        "# Threshold Sensitivity",
        "",
        "Date: 2026-09-21",
        "",
        f"Completed checkpoints: {len(records)}/{len(MODELS) * len(SEEDS)}.",
        "",
        "## Protocol",
        "",
        "- Original, non-augmented images only.",
        "- Thresholds: 0.3, 0.4, 0.5, 0.6, 0.7.",
        "- Macro metrics weight a1/a2/a3/b/c1/c2/c3 equally.",
        "",
        "## Best Macro Threshold",
        "",
        "| Model | Best Threshold | Best Macro IoU | IoU at 0.5 | Delta |",
        "|---|---:|---:|---:|---:|",
    ]
    for model in MODELS:
        rows = [row for row in model_summary if row["model"] == model]
        best = max(rows, key=lambda row: row["macro_iou_mean"])
        default = next(row for row in rows if row["threshold"] == 0.5)
        lines.append(f'| {model} | {best["threshold"]:.1f} | {best["macro_iou_mean"]:.4f} | {default["macro_iou_mean"]:.4f} | {best["macro_iou_mean"] - default["macro_iou_mean"]:+.4f} |')

    lines += ["", "## Macro IoU By Threshold", "", "| Model | " + " | ".join(f"{value:.1f}" for value in THRESHOLDS) + " |", "|---|" + "|".join(["---:"] * len(THRESHOLDS)) + "|"]
    for model in MODELS:
        values = [next(row["macro_iou_mean"] for row in model_summary if row["model"] == model and row["threshold"] == threshold) for threshold in THRESHOLDS]
        lines.append(f'| {model} | ' + " | ".join(f"{value:.4f}" for value in values) + " |")

    lines += ["", "## Best Threshold By Group (IoU)", "", "| Model | " + " | ".join(REGIONS) + " |", "|---|" + "|".join(["---:"] * len(REGIONS)) + "|"]
    for model in MODELS:
        values = []
        for region in REGIONS:
            rows = [row for row in group_summary if row["model"] == model and row["group"] == region]
            best = max(rows, key=lambda row: row["iou_mean"])
            values.append(f'{best["threshold"]:.1f} ({best["iou_mean"]:.3f})')
        lines.append(f'| {model} | ' + " | ".join(values) + " |")

    lines += [
        "",
        "## Files",
        "",
        "- Raw JSON: D:\\landslide_unet_project\\reports\\threshold_sensitivity_raw",
        "- Per-seed results: D:\\landslide_unet_project\\reports\\threshold_sensitivity_seed_group_results.csv",
        "- Model summary: D:\\landslide_unet_project\\reports\\threshold_sensitivity_model_summary.csv",
        "- Figure: D:\\landslide_unet_project\\figures\\threshold_sensitivity_summary.png",
    ]
    (REPORTS / "threshold_sensitivity_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    records = load_records(args.allow_incomplete)
    if not records:
        raise SystemExit("No threshold records found")
    seed_rows = write_seed_table(records)
    model_summary = model_threshold_summary(seed_rows)
    group_summary = group_threshold_summary(seed_rows)
    write_figure(model_summary)
    write_report(records, model_summary, group_summary)
    print("completed checkpoints", len(records))
    print("wrote", REPORTS / "threshold_sensitivity_report.md")


if __name__ == "__main__":
    main()
