import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import os as _repo_os
PROJECT = Path(_repo_os.environ.get("LANDSLIDE_PROJECT_ROOT", Path(__file__).resolve().parents[1])).resolve()
RAW_DIR = PROJECT / "reports" / "cross_group_raw"
REPORTS = PROJECT / "reports"
FIGURES = PROJECT / "figures"
MANIFEST = PROJECT / "data" / "splits" / "resunet_bfa_group_manifest.csv"
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
SUBSETS = ["all", "original"]
METRICS = ["iou", "dice", "f1", "precision", "recall", "accuracy"]


def load_raw(allow_incomplete):
    records = []
    missing = []
    for model in MODELS:
        for seed in SEEDS:
            path = RAW_DIR / f"{RUN_PREFIX[model]}_seed{seed}.json"
            if not path.exists():
                missing.append(str(path))
                continue
            records.append(json.loads(path.read_text(encoding="utf-8")))
    if missing and not allow_incomplete:
        raise SystemExit("Missing cross-group results:\n" + "\n".join(missing))
    return records


def manifest_counts():
    counts = {subset: defaultdict(int) for subset in SUBSETS}
    with MANIFEST.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            region = row["region"]
            if region not in REGIONS:
                continue
            counts["all"][region] += 1
            if int(row["augmented"]) == 0:
                counts["original"][region] += 1
    return counts


def write_seed_tables(records, counts):
    rows = []
    for record in records:
        for subset in SUBSETS:
            for region in REGIONS:
                values = record["groups"][subset][region]
                row = {
                    "model": record["model"],
                    "seed": record["seed"],
                    "best_epoch": record["best_epoch"],
                    "subset": subset,
                    "group": region,
                    "n": counts[subset][region],
                }
                row.update({metric: values[metric] for metric in METRICS})
                row.update({key: values[key] for key in ["tp", "fp", "fn", "tn"]})
                rows.append(row)
    fields = ["model", "seed", "best_epoch", "subset", "group", "n", *METRICS, "tp", "fp", "fn", "tn"]
    with (REPORTS / "cross_group_seed_group_results.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    summary = []
    for model in MODELS:
        for subset in SUBSETS:
            for region in REGIONS:
                group_rows = [row for row in rows if row["model"] == model and row["subset"] == subset and row["group"] == region]
                if not group_rows:
                    continue
                item = {
                    "model": model,
                    "subset": subset,
                    "group": region,
                    "n": counts[subset][region],
                    "seeds": len(group_rows),
                }
                for metric in METRICS:
                    values = [row[metric] for row in group_rows]
                    item[f"{metric}_mean"] = statistics.mean(values)
                    item[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
                summary.append(item)
    fields = ["model", "subset", "group", "n", "seeds", *[f"{metric}_{suffix}" for metric in METRICS for suffix in ["mean", "std"]]]
    with (REPORTS / "cross_group_seed_group_summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary)
    return rows, summary


def model_summary(rows):
    summary = []
    for model in MODELS:
        for subset in SUBSETS:
            model_rows = [row for row in rows if row["model"] == model and row["subset"] == subset]
            if not model_rows:
                continue
            seed_macros = {}
            for seed in SEEDS:
                seed_rows = [row for row in model_rows if row["seed"] == seed]
                if not seed_rows:
                    continue
                seed_macros[seed] = {
                    metric: statistics.mean(row[metric] for row in seed_rows)
                    for metric in METRICS
                }
            group_means = {
                region: {
                    metric: statistics.mean(
                        row[metric]
                        for row in model_rows
                        if row["group"] == region
                    )
                    for metric in METRICS
                }
                for region in REGIONS
            }
            worst_group = min(REGIONS, key=lambda region: group_means[region]["iou"])
            best_group = max(REGIONS, key=lambda region: group_means[region]["iou"])
            item = {
                "model": model,
                "subset": subset,
                "macro_iou_mean": statistics.mean(seed_macros[seed]["iou"] for seed in seed_macros),
                "macro_iou_std_seed": statistics.stdev(seed_macros[seed]["iou"] for seed in seed_macros) if len(seed_macros) > 1 else 0.0,
                "macro_dice_mean": statistics.mean(seed_macros[seed]["dice"] for seed in seed_macros),
                "macro_dice_std_seed": statistics.stdev(seed_macros[seed]["dice"] for seed in seed_macros) if len(seed_macros) > 1 else 0.0,
                "worst_group": worst_group,
                "worst_group_iou": group_means[worst_group]["iou"],
                "best_group": best_group,
                "best_group_iou": group_means[best_group]["iou"],
                "group_iou_std": statistics.stdev(group_means[region]["iou"] for region in REGIONS),
            }
            for metric in METRICS:
                item[f"macro_{metric}"] = statistics.mean(
                    group_means[region][metric] for region in REGIONS
                )
            summary.append(item)
    fields = [
        "model", "subset", "macro_iou_mean", "macro_iou_std_seed", "macro_dice_mean", "macro_dice_std_seed",
        "worst_group", "worst_group_iou", "best_group", "best_group_iou", "group_iou_std",
        *[f"macro_{metric}" for metric in METRICS],
    ]
    with (REPORTS / "cross_group_model_summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary)
    return summary


def write_figure(summary, group_summary):
    model_names = MODELS
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    for ax, subset in zip(axes[0], SUBSETS):
        matrix = np.array([
            [next(row for row in summary if row["model"] == model and row["subset"] == subset)["macro_iou_mean"]]
            for model in model_names
        ])
        group_matrix = []
        for model in model_names:
            model_rows = [row for row in group_summary if row["model"] == model and row["subset"] == subset]
            group_matrix.append([next(row["iou_mean"] for row in model_rows if row["group"] == region) for region in REGIONS])
        image = ax.imshow(np.array(group_matrix), cmap="viridis", aspect="auto")
        ax.set_xticks(range(len(REGIONS)), REGIONS)
        ax.set_yticks(range(len(model_names)), model_names)
        ax.set_title(f"IoU by source group ({subset})")
        for i in range(len(model_names)):
            for j in range(len(REGIONS)):
                ax.text(j, i, f"{group_matrix[i][j]:.3f}", ha="center", va="center", color="white", fontsize=8)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    for ax, (value_key, title) in zip(axes[1], [("macro_iou_mean", "Macro-average IoU"), ("worst_group_iou", "Worst-group IoU")]):
        width = 0.36
        x = np.arange(len(model_names))
        for offset, subset in zip([-width / 2, width / 2], SUBSETS):
            values = [next(row for row in summary if row["model"] == model and row["subset"] == subset)[value_key] for model in model_names]
            ax.bar(x + offset, values, width=width, label=subset)
        ax.set_xticks(x, model_names, rotation=25, ha="right")
        ax.set_title(title)
        ax.set_ylabel("IoU")
        ax.grid(axis="y", alpha=0.25)
        ax.legend()
    fig.suptitle("ResUNet-BFA Cross-Group Evaluation", fontsize=15)
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "cross_group_summary.png", dpi=200)
    plt.close(fig)


def write_report(records, seed_rows, group_summary, summary):
    complete = len(records) == len(MODELS) * len(SEEDS)
    lines = [
        "# ResUNet-BFA Cross-Group Evaluation",
        "",
        "Date: 2026-09-21",
        "",
        f"Completed checkpoints: {len(records)}/{len(MODELS) * len(SEEDS)}.",
        "",
        "## Scope",
        "",
        "- Evaluates the already trained CAS models without retraining.",
        "- Groups: a1, a2, a3, b, c1, c2, c3.",
        "- `all` matches the existing c3 protocol, including offline augmented variants.",
        "- `original` uses only original (non-augmented) images and is a leakage-aware sensitivity view.",
        "- Macro metrics weight source groups equally rather than weighting by image count.",
        "",
        "## Macro Results",
        "",
        "| Model | Subset | Macro IoU | Seed Std | Macro Dice | Worst Group | Worst IoU | Best Group |",
        "|---|---|---:|---:|---:|---|---:|---|",
    ]
    for row in summary:
        lines.append(
            f'| {row["model"]} | {row["subset"]} | {row["macro_iou_mean"]:.4f} | '
            f'{row["macro_iou_std_seed"]:.4f} | {row["macro_dice_mean"]:.4f} | '
            f'{row["worst_group"]} | {row["worst_group_iou"]:.4f} | {row["best_group"]} |'
        )

    lines += ["", "## Per-Group Mean IoU (All Samples)", "", "| Model | " + " | ".join(REGIONS) + " |", "|---|" + "|".join(["---:"] * len(REGIONS)) + "|"]
    for model in MODELS:
        values = []
        for region in REGIONS:
            row = next(row for row in group_summary if row["model"] == model and row["subset"] == "all" and row["group"] == region)
            values.append(f'{row["iou_mean"]:.4f} ± {row["iou_std"]:.4f}')
        lines.append(f'| {model} | ' + " | ".join(values) + " |")

    lines += ["", "## Per-Group Mean IoU (Original Only)", "", "| Model | " + " | ".join(REGIONS) + " |", "|---|" + "|".join(["---:"] * len(REGIONS)) + "|"]
    for model in MODELS:
        values = []
        for region in REGIONS:
            row = next(row for row in group_summary if row["model"] == model and row["subset"] == "original" and row["group"] == region)
            values.append(f'{row["iou_mean"]:.4f} ± {row["iou_std"]:.4f}')
        lines.append(f'| {model} | ' + " | ".join(values) + " |")

    lines += ["", "## ASK-UNet++ Group-Wise Deltas", "", "| Subset | Comparator | Macro Delta IoU | Groups Better | Groups Worse |", "|---|---|---:|---:|---:|"]
    for subset in SUBSETS:
        ask = next(row for row in summary if row["model"] == "ASK-UNet++" and row["subset"] == subset)
        for comparator in ["UNet", "U-Net++", "AS_UNet"]:
            comp = next(row for row in summary if row["model"] == comparator and row["subset"] == subset)
            deltas = []
            for region in REGIONS:
                ask_group = next(row for row in group_summary if row["model"] == "ASK-UNet++" and row["subset"] == subset and row["group"] == region)
                comp_group = next(row for row in group_summary if row["model"] == comparator and row["subset"] == subset and row["group"] == region)
                deltas.append(ask_group["iou_mean"] - comp_group["iou_mean"])
            lines.append(
                f'| {subset} | {comparator} | {ask["macro_iou_mean"] - comp["macro_iou_mean"]:.4f} | '
                f'{sum(value > 0 for value in deltas)} | {sum(value < 0 for value in deltas)} |'
            )

    lines += [
        "",
        "## Interpretation Guardrails",
        "",
        "- `all` reproduces the prior c3 protocol; `original` is preferred for leakage-aware sensitivity analysis.",
        "- Group macro averages prevent the large b group from dominating the conclusion.",
        "- These are external evaluations of CAS-trained checkpoints, not models retrained for each source group.",
        "- Only one external dataset family is used; broader cross-dataset claims still require additional datasets.",
        "",
        "## Files",
        "",
        "- Raw per-checkpoint JSON: D:\\landslide_unet_project\\reports\\cross_group_raw",
        "- Per-seed/group metrics: D:\\landslide_unet_project\\reports\\cross_group_seed_group_results.csv",
        "- Seed/group summary: D:\\landslide_unet_project\\reports\\cross_group_seed_group_summary.csv",
        "- Model macro summary: D:\\landslide_unet_project\\reports\\cross_group_model_summary.csv",
        "- Figure: D:\\landslide_unet_project\\figures\\cross_group_summary.png",
    ]
    if not complete:
        lines.insert(5, "WARNING: evaluation is incomplete.")
    (REPORTS / "cross_group_experiment_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    records = load_raw(args.allow_incomplete)
    if not records:
        raise SystemExit("No cross-group records found")
    counts = manifest_counts()
    seed_rows, group_summary = write_seed_tables(records, counts)
    summary = model_summary(seed_rows)
    write_figure(summary, group_summary)
    write_report(records, seed_rows, group_summary, summary)
    print("completed checkpoints", len(records))
    print("wrote", REPORTS / "cross_group_experiment_report.md")


if __name__ == "__main__":
    main()
