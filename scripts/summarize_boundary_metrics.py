import argparse
import csv
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import os as _repo_os
PROJECT = Path(_repo_os.environ.get("LANDSLIDE_PROJECT_ROOT", Path(__file__).resolve().parents[1])).resolve()
RAW_DIR = PROJECT / "reports" / "boundary_metrics_raw"
REPORTS = PROJECT / "reports"
FIGURES = PROJECT / "figures"
SEEDS = [42, 2026, 777]
REGIONS = ["a1", "a2", "a3", "b", "c1", "c2", "c3"]
MODELS = ["UNet", "NestedUNet", "AS_UNet", "U-Net++", "ASK-UNet++", "Bottleneck-LiteASK"]
RUN_PREFIX = {
    "UNet": "group_unet",
    "NestedUNet": "group_nestedunet",
    "AS_UNet": "group_asunet32",
    "U-Net++": "group_unetpp",
    "ASK-UNet++": "group_askunetpp",
    "Bottleneck-LiteASK": "group_bottleneckliteaskunetpp",
}
METRIC_KEYS = ["f1_2_mean", "f1_4_mean", "hd95_mean", "precision_2_mean", "recall_2_mean", "precision_4_mean", "recall_4_mean"]


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
        raise SystemExit("Missing boundary results:\n" + "\n".join(missing))
    return records


def write_tables(records):
    rows = []
    for record in records:
        for region in REGIONS:
            values = record["groups"][region]
            row = {"model": record["model"], "seed": record["seed"], "group": region}
            row.update({key: values[key] for key in METRIC_KEYS})
            row.update({
                "n_scored": values["n_scored"],
                "n_skipped_empty_target": values["n_skipped_empty_target"],
            })
            rows.append(row)
    fields = ["model", "seed", "group", *METRIC_KEYS, "n_scored", "n_skipped_empty_target"]
    with (REPORTS / "boundary_metrics_seed_group_results.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    summary = []
    for model in MODELS:
        for region in REGIONS:
            group_rows = [row for row in rows if row["model"] == model and row["group"] == region]
            if not group_rows:
                continue
            item = {"model": model, "group": region, "seeds": len(group_rows)}
            for key in METRIC_KEYS:
                vals = [row[key] for row in group_rows]
                item[f"{key}_seed_mean"] = statistics.mean(vals)
                item[f"{key}_seed_std"] = statistics.stdev(vals) if len(vals) > 1 else 0.0
            summary.append(item)
    fields = ["model", "group", "seeds", *[f"{key}_{suffix}" for key in METRIC_KEYS for suffix in ["seed_mean", "seed_std"]]]
    with (REPORTS / "boundary_metrics_seed_group_summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary)
    return rows, summary


def model_summary(rows):
    output = []
    for model in MODELS:
        model_rows = [row for row in rows if row["model"] == model]
        item = {"model": model}
        for key in METRIC_KEYS:
            group_means = [
                statistics.mean(row[key] for row in model_rows if row["group"] == region)
                for region in REGIONS
            ]
            item[key] = statistics.mean(group_means)
            item[f"{key}_group_std"] = statistics.stdev(group_means)
        hd_group_means = {
            region: statistics.mean(row["hd95_mean"] for row in model_rows if row["group"] == region)
            for region in REGIONS
        }
        f1_group_means = {
            region: statistics.mean(row["f1_2_mean"] for row in model_rows if row["group"] == region)
            for region in REGIONS
        }
        item["worst_group_hd95"] = max(REGIONS, key=lambda region: hd_group_means[region])
        item["worst_group_hd95_value"] = hd_group_means[item["worst_group_hd95"]]
        item["worst_group_f1_2"] = min(REGIONS, key=lambda region: f1_group_means[region])
        item["worst_group_f1_2_value"] = f1_group_means[item["worst_group_f1_2"]]
        output.append(item)
    fields = ["model", *[f"{key}{suffix}" for key in METRIC_KEYS for suffix in ["", "_group_std"]], "worst_group_hd95", "worst_group_hd95_value", "worst_group_f1_2", "worst_group_f1_2_value"]
    with (REPORTS / "boundary_metrics_model_summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output)
    return output


def write_figure(group_summary, summary):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    for ax, (key, title, cmap) in zip(axes[0], [("f1_2_mean", "Boundary F1@2", "viridis"), ("hd95_mean", "HD95 (lower is better)", "magma")]):
        matrix = np.array([
            [next(row[f"{key}_seed_mean"] for row in group_summary if row["model"] == model and row["group"] == region) for region in REGIONS]
            for model in MODELS
        ])
        image = ax.imshow(matrix, cmap=cmap, aspect="auto")
        ax.set_xticks(range(len(REGIONS)), REGIONS)
        ax.set_yticks(range(len(MODELS)), MODELS)
        ax.set_title(title)
        for i in range(len(MODELS)):
            for j in range(len(REGIONS)):
                ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", color="white", fontsize=8)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    for ax, (key, title) in zip(axes[1], [("f1_2_mean", "Macro Boundary F1@2"), ("hd95_mean", "Macro HD95")]):
        values = [next(row[key] for row in summary if row["model"] == model) for model in MODELS]
        ax.bar(range(len(MODELS)), values, color=plt.cm.tab10.colors[:len(MODELS)])
        ax.set_xticks(range(len(MODELS)), MODELS, rotation=25, ha="right")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Original-Only Boundary Metrics Across ResUNet-BFA Groups", fontsize=15)
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "boundary_metrics_summary.png", dpi=200)
    plt.close(fig)


def write_report(records, group_summary, summary):
    lines = [
        "# Boundary Metrics Across ResUNet-BFA Groups",
        "",
        "Date: 2026-09-21",
        "",
        f"Completed checkpoints: {len(records)}/{len(MODELS) * len(SEEDS)}.",
        "",
        "## Protocol",
        "",
        "- Original, non-augmented images only.",
        "- Binary prediction threshold: 0.5.",
        "- Boundary F1 tolerances: 2 and 4 pixels at 128x128.",
        "- HD95 uses symmetric 95th-percentile boundary distance.",
        "- Samples with empty target boundaries are excluded from BF1/HD95.",
        "",
        "## Macro Results",
        "",
        "| Model | BF1@2 | BF1@4 | HD95 | Worst BF1@2 Group | Worst HD95 Group |",
        "|---|---:|---:|---:|---|---|",
    ]
    for row in summary:
        lines.append(
            f'| {row["model"]} | {row["f1_2_mean"]:.4f} | {row["f1_4_mean"]:.4f} | {row["hd95_mean"]:.3f} | '
            f'{row["worst_group_f1_2"]} ({row["worst_group_f1_2_value"]:.4f}) | '
            f'{row["worst_group_hd95"]} ({row["worst_group_hd95_value"]:.3f}) |'
        )
    lines += ["", "## Per-Group BF1@2", "", "| Model | " + " | ".join(REGIONS) + " |", "|---|" + "|".join(["---:"] * len(REGIONS)) + "|"]
    for model in MODELS:
        vals = []
        for region in REGIONS:
            row = next(row for row in group_summary if row["model"] == model and row["group"] == region)
            vals.append(f'{row["f1_2_mean_seed_mean"]:.4f}')
        lines.append(f'| {model} | ' + " | ".join(vals) + " |")
    lines += ["", "## Per-Group HD95", "", "| Model | " + " | ".join(REGIONS) + " |", "|---|" + "|".join(["---:"] * len(REGIONS)) + "|"]
    for model in MODELS:
        vals = []
        for region in REGIONS:
            row = next(row for row in group_summary if row["model"] == model and row["group"] == region)
            vals.append(f'{row["hd95_mean_seed_mean"]:.2f}')
        lines.append(f'| {model} | ' + " | ".join(vals) + " |")
    lines += [
        "",
        "## Guardrails",
        "",
        "- HD95 is sensitive to empty or fragmented predictions; empty predicted boundaries are assigned the image diagonal.",
        "- Group macro averages weight a1/a2/a3/b/c1/c2/c3 equally.",
        "- These boundary metrics use checkpoints trained only on CAS and evaluated externally without retraining.",
        "",
        "## Files",
        "",
        "- Raw JSON: D:\\landslide_unet_project\\reports\\boundary_metrics_raw",
        "- Per-seed results: D:\\landslide_unet_project\\reports\\boundary_metrics_seed_group_results.csv",
        "- Group summary: D:\\landslide_unet_project\\reports\\boundary_metrics_seed_group_summary.csv",
        "- Model summary: D:\\landslide_unet_project\\reports\\boundary_metrics_model_summary.csv",
        "- Figure: D:\\landslide_unet_project\\figures\\boundary_metrics_summary.png",
    ]
    (REPORTS / "boundary_metrics_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    records = load_records(args.allow_incomplete)
    if not records:
        raise SystemExit("No boundary records found")
    rows, group_summary = write_tables(records)
    summary = model_summary(rows)
    write_figure(group_summary, summary)
    write_report(records, group_summary, summary)
    print("completed checkpoints", len(records))
    print("wrote", REPORTS / "boundary_metrics_report.md")


if __name__ == "__main__":
    main()
