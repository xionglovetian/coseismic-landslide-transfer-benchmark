import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

import os as _repo_os
PROJECT = Path(_repo_os.environ.get("LANDSLIDE_PROJECT_ROOT", Path(__file__).resolve().parents[1])).resolve()
REPORTS = PROJECT / "reports"
RAW_DIR = REPORTS / "external_regions_raw"
FIGURES = PROJECT / "figures"
SEEDS = [42, 2026, 777]
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
REGION_NAMES = {
    "longxi_river": "Longxi River",
    "wenchuan": "Wenchuan",
    "jiuzhai_valley": "Jiuzhai Valley",
    "moxitaidi": "Moxitaidi",
}
REGIONS = list(REGION_NAMES)

def load_records():
    records = []
    for model in MODELS:
        for seed in SEEDS:
            path = RAW_DIR / f"{RUN_PREFIX[model]}_seed{seed}.json"
            if not path.exists():
                raise FileNotFoundError(path)
            records.append(json.loads(path.read_text(encoding="utf-8")))
    return records

def write_seed_tables(records):
    rows = []
    for record in records:
        for region in REGIONS:
            values = record["groups"][region]
            row = {
                "model": record["model"],
                "seed": record["seed"],
                "region": region,
                "region_name": REGION_NAMES[region],
                "n_scored_boundary": values["boundary_n_scored"],
                "n_skipped_empty_target": values["boundary_n_skipped_empty_target"],
            }
            for threshold in THRESHOLDS:
                metrics = values["threshold_metrics"][str(threshold)]
                row[f"iou_{threshold}"] = metrics["iou"]
                row[f"dice_{threshold}"] = metrics["dice"]
            row.update({
                "bf1_2": values["f1_2_mean"],
                "bf1_4": values["f1_4_mean"],
                "hd95": values["hd95_mean"],
            })
            rows.append(row)
    fields = list(rows[0].keys())
    with (REPORTS / "external_benchmark_seed_region_results.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    summary = []
    for model in MODELS:
        for region in REGIONS:
            group_rows = [row for row in rows if row["model"] == model and row["region"] == region]
            item = {"model": model, "region": region, "region_name": REGION_NAMES[region], "seeds": len(group_rows)}
            for metric in ["iou_0.5", "dice_0.5", "bf1_2", "bf1_4", "hd95"]:
                values = [float(row[metric]) for row in group_rows]
                item[f"{metric}_mean"] = statistics.mean(values)
                item[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
            summary.append(item)
    with (REPORTS / "external_benchmark_region_summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)
    return rows, summary

def model_summary(rows):
    output = []
    for model in MODELS:
        model_rows = [row for row in rows if row["model"] == model]
        seed_macro = {}
        for seed in SEEDS:
            values = [float(row["iou_0.5"]) for row in model_rows if row["seed"] == seed]
            seed_macro[seed] = statistics.mean(values)
        group_means = {
            region: statistics.mean(float(row["iou_0.5"]) for row in model_rows if row["region"] == region)
            for region in REGIONS
        }
        worst = min(REGIONS, key=lambda region: group_means[region])
        best = max(REGIONS, key=lambda region: group_means[region])
        output.append({
            "model": model,
            "macro_iou": statistics.mean(seed_macro.values()),
            "macro_iou_seed_std": statistics.stdev(seed_macro.values()),
            "worst_region": worst,
            "worst_iou": group_means[worst],
            "best_region": best,
            "best_iou": group_means[best],
            "macro_bf1_2": statistics.mean(float(row["bf1_2"]) for row in model_rows),
            "macro_hd95": statistics.mean(float(row["hd95"]) for row in model_rows),
        })
    with (REPORTS / "external_benchmark_model_summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output[0].keys()))
        writer.writeheader()
        writer.writerows(output)
    return output

def paired_comparisons(rows):
    output = []
    for comparator in ["UNet", "NestedUNet", "AS_UNet", "U-Net++", "ASK-UNet++"]:
        candidate = []
        baseline = []
        deltas_by_region = []
        for seed in SEEDS:
            candidate_macro = statistics.mean(float(row["iou_0.5"]) for row in rows if row["model"] == "Bottleneck-LiteASK" and row["seed"] == seed)
            baseline_macro = statistics.mean(float(row["iou_0.5"]) for row in rows if row["model"] == comparator and row["seed"] == seed)
            candidate.append(candidate_macro)
            baseline.append(baseline_macro)
        for region in REGIONS:
            ask_values = [float(row["iou_0.5"]) for row in rows if row["model"] == "Bottleneck-LiteASK" and row["region"] == region]
            other_values = [float(row["iou_0.5"]) for row in rows if row["model"] == comparator and row["region"] == region]
            deltas_by_region.append(statistics.mean(ask_values) - statistics.mean(other_values))
        deltas = [a - b for a, b in zip(candidate, baseline)]
        output.append({
            "comparator": comparator,
            "mean_delta": statistics.mean(deltas),
            "std_delta": statistics.stdev(deltas),
            "p_value": float(stats.ttest_rel(candidate, baseline).pvalue),
            "delta_seed_42": deltas[0],
            "delta_seed_2026": deltas[1],
            "delta_seed_777": deltas[2],
            "groups_better": sum(value > 0 for value in deltas_by_region),
            "groups_worse": sum(value < 0 for value in deltas_by_region),
        })
    with (REPORTS / "external_benchmark_paired_comparisons.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output[0].keys()))
        writer.writeheader()
        writer.writerows(output)
    return output

def write_figure(group_summary, model_summary_data):
    matrix = np.array([
        [next(row["iou_0.5_mean"] for row in group_summary if row["model"] == model and row["region"] == region) for region in REGIONS]
        for model in MODELS
    ])
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)
    image = axes[0].imshow(matrix, cmap="viridis", aspect="auto")
    axes[0].set_xticks(range(len(REGIONS)), [REGION_NAMES[r] for r in REGIONS], rotation=20, ha="right")
    axes[0].set_yticks(range(len(MODELS)), MODELS)
    axes[0].set_title("External IoU by region")
    for i in range(len(MODELS)):
        for j in range(len(REGIONS)):
            axes[0].text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center", color="white", fontsize=8)
    fig.colorbar(image, ax=axes[0], fraction=0.046, pad=0.04)
    values = [row["macro_iou"] for row in model_summary_data]
    worst = [row["worst_iou"] for row in model_summary_data]
    x = np.arange(len(MODELS)); width = 0.38
    axes[1].bar(x - width/2, values, width, label="Macro IoU")
    axes[1].bar(x + width/2, worst, width, label="Worst-region IoU")
    axes[1].set_xticks(x, MODELS, rotation=25, ha="right")
    axes[1].set_title("Macro and worst-region performance")
    axes[1].grid(axis="y", alpha=0.25); axes[1].legend()
    fig.suptitle("New Independent External-Region Benchmark", fontsize=14)
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "external_benchmark_summary.png", dpi=200)
    plt.close(fig)

def write_report(records, group_summary, model_summary_data, paired):
    lines = [
        "# New Independent External-Region Benchmark",
        "",
        "Date: 2026-09-21",
        "",
        f"Checkpoints: {len(records)}/{len(MODELS) * len(SEEDS)}.",
        "",
        "## Dataset",
        "",
        "- Longxi River: 2,504 pairs",
        "- Wenchuan: 178 pairs",
        "- Jiuzhai Valley: 5,925 pairs",
        "- Moxitaidi: 980 eligible pairs after excluding four CAS-overlap suspects",
        "- All images are 512x512 RGB; masks are binary 0/255 PNG.",
        "",
        "## Macro Results",
        "",
        "| Model | Macro IoU | Seed Std | Worst Region | Worst IoU | Macro BF1@2 | Macro HD95 |",
        "|---|---:|---:|---|---:|---:|---:|",
    ]
    for row in model_summary_data:
        lines.append(
            f'| {row["model"]} | {row["macro_iou"]:.4f} | {row["macro_iou_seed_std"]:.4f} | '
            f'{REGION_NAMES[row["worst_region"]]} | {row["worst_iou"]:.4f} | '
            f'{row["macro_bf1_2"]:.4f} | {row["macro_hd95"]:.2f} |'
        )
    lines += ["", "## Region IoU", "", "| Model | " + " | ".join(REGION_NAMES[r] for r in REGIONS) + " |", "|---|" + "|".join(["---:"] * len(REGIONS)) + "|"]
    for model in MODELS:
        vals = []
        for region in REGIONS:
            row = next(row for row in group_summary if row["model"] == model and row["region"] == region)
            vals.append(f'{row["iou_0.5_mean"]:.4f} ± {row["iou_0.5_std"]:.4f}')
        lines.append(f'| {model} | ' + " | ".join(vals) + " |")
    lines += ["", "## Paired Candidate vs Baselines", "", "| Comparator | Mean delta | Std delta | Paired p | Groups better | Groups worse |", "|---|---:|---:|---:|---:|---:|"]
    for row in paired:
        lines.append(
            f'| {row["comparator"]} | {row["mean_delta"]:+.4f} | {row["std_delta"]:.4f} | '
            f'{row["p_value"]:.4f} | {row["groups_better"]} | {row["groups_worse"]} |'
        )
    lines += [
        "",
        "## Guardrails",
        "",
        "- These regions were not used for training or checkpoint selection.",
        "- Moxi town was excluded because of severe overlap with CAS training data.",
        "- Moxitaidi is an unseen region but may share source lineage with CAS; interpret it separately from Longxi/Wenchuan/Jiuzhai.",
        "- Four Moxitaidi samples with pHASH distance 8 to CAS were excluded from the strict manifest.",
        "",
        "## Files",
        "",
        "- Prepared dataset: D:\\landslide_unet_project\\data\\processed\\external_regions_512",
        "- Per-seed results: D:\\landslide_unet_project\\reports\\external_benchmark_seed_region_results.csv",
        "- Region summary: D:\\landslide_unet_project\\reports\\external_benchmark_region_summary.csv",
        "- Model summary: D:\\landslide_unet_project\\reports\\external_benchmark_model_summary.csv",
        "- Paired comparisons: D:\\landslide_unet_project\\reports\\external_benchmark_paired_comparisons.csv",
        "- Figure: D:\\landslide_unet_project\\figures\\external_benchmark_summary.png",
    ]
    (REPORTS / "external_benchmark_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

def main():
    records = load_records()
    rows, group_summary = write_seed_tables(records)
    summary = model_summary(rows)
    paired = paired_comparisons(rows)
    write_figure(group_summary, summary)
    write_report(records, group_summary, summary, paired)
    print("completed checkpoints", len(records))
    print("wrote", REPORTS / "external_benchmark_report.md")

if __name__ == "__main__":
    main()
