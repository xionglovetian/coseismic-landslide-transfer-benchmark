import csv
import statistics
from pathlib import Path

from scipy import stats

import os as _repo_os
PROJECT = Path(_repo_os.environ.get("LANDSLIDE_PROJECT_ROOT", Path(__file__).resolve().parents[1])).resolve()
REPORTS = PROJECT / "reports"
MODELS = ["UNet", "NestedUNet", "AS_UNet", "U-Net++", "ASK-UNet++", "Bottleneck-LiteASK"]
SEEDS = [42, 2026, 777]
REGIONS = ["a1", "a2", "a3", "b", "c1", "c2", "c3"]


def read_csv(name):
    with (REPORTS / name).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def row_for(rows, **criteria):
    return next(row for row in rows if all(str(row[key]) == str(value) for key, value in criteria.items()))


main = read_csv("group_model_seed_summary.csv")
cross_model = read_csv("cross_group_model_summary.csv")
cross_group = read_csv("cross_group_seed_group_summary.csv")
cross_seed = read_csv("cross_group_seed_group_results.csv")
boundary = read_csv("boundary_metrics_model_summary.csv")
complexity = read_csv("model_complexity.csv")
threshold = read_csv("threshold_sensitivity_model_summary.csv")

rows = []
for model in MODELS:
    cas = row_for(main, model=model, split="cas_val", metric="iou")
    c3 = row_for(cross_group, model=model, subset="all", group="c3")
    macro_all = row_for(cross_model, model=model, subset="all")
    macro_orig = row_for(cross_model, model=model, subset="original")
    boundary_row = row_for(boundary, model=model)
    complexity_row = row_for(complexity, model=model)
    threshold_rows = [row for row in threshold if row["model"] == model]
    best_threshold = max(threshold_rows, key=lambda row: float(row["macro_iou_mean"]))
    default_threshold = row_for(threshold, model=model, threshold="0.5")
    rows.append({
        "model": model,
        "cas_iou_mean": float(cas["mean"]),
        "cas_iou_std": float(cas["std"]),
        "c3_all_iou_mean": float(c3["iou_mean"]),
        "c3_all_iou_std": float(c3["iou_std"]),
        "macro_all_iou": float(macro_all["macro_iou_mean"]),
        "macro_all_seed_std": float(macro_all["macro_iou_std_seed"]),
        "macro_original_iou": float(macro_orig["macro_iou_mean"]),
        "macro_original_seed_std": float(macro_orig["macro_iou_std_seed"]),
        "worst_group_original": macro_orig["worst_group"],
        "worst_group_original_iou": float(macro_orig["worst_group_iou"]),
        "boundary_f1_2": float(boundary_row["f1_2_mean"]),
        "boundary_f1_4": float(boundary_row["f1_4_mean"]),
        "hd95": float(boundary_row["hd95_mean"]),
        "best_threshold": float(best_threshold["threshold"]),
        "best_threshold_macro_iou": float(best_threshold["macro_iou_mean"]),
        "threshold_gain_vs_05": float(best_threshold["macro_iou_mean"]) - float(default_threshold["macro_iou_mean"]),
        "params_million": float(complexity_row["params_million"]),
        "gflops": float(complexity_row["gflops_b1_128"]),
        "fps_batch32": float(complexity_row["fps_b32_bf16"]),
    })

fields = list(rows[0].keys())
with (REPORTS / "combined_model_assessment.csv").open("w", newline="", encoding="utf-8-sig") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)

lines = [
    "# Combined Model Assessment",
    "",
    "Date: 2026-09-21",
    "",
    "## Integrated Metrics",
    "",
    "| Model | CAS IoU | c3 IoU | Macro IoU (all) | Macro IoU (original) | Worst Original Group | BF1@2 | HD95 | Params (M) | GFLOPs | FPS b32 |",
    "|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|",
]
for row in rows:
    lines.append(
        f'| {row["model"]} | {row["cas_iou_mean"]:.4f} ± {row["cas_iou_std"]:.4f} | '
        f'{row["c3_all_iou_mean"]:.4f} ± {row["c3_all_iou_std"]:.4f} | '
        f'{row["macro_all_iou"]:.4f} | {row["macro_original_iou"]:.4f} | '
        f'{row["worst_group_original"]} ({row["worst_group_original_iou"]:.4f}) | '
        f'{row["boundary_f1_2"]:.4f} | {row["hd95"]:.2f} | '
        f'{row["params_million"]:.2f} | {row["gflops"]:.2f} | {row["fps_batch32"]:.1f} |'
    )


def macro_iou(model, subset, seed):
    values = [
        float(row["iou"])
        for row in cross_seed
        if row["model"] == model
        and row["subset"] == subset
        and int(row["seed"]) == seed
        and row["group"] in REGIONS
    ]
    return statistics.mean(values)


lines += [
    "",
    "## Paired Macro Comparisons",
    "",
    "| Subset | Comparator | Mean delta | Paired p | Delta by seed |",
    "|---|---|---:|---:|---|",
]
for subset in ["all", "original"]:
    candidate = [macro_iou("Bottleneck-LiteASK", subset, seed) for seed in SEEDS]
    for comparator in ["UNet", "U-Net++", "ASK-UNet++"]:
        baseline = [macro_iou(comparator, subset, seed) for seed in SEEDS]
        deltas = [a - b for a, b in zip(candidate, baseline)]
        p_value = stats.ttest_rel(candidate, baseline).pvalue
        lines.append(
            f'| {subset} | {comparator} | {statistics.mean(deltas):+.4f} | {p_value:.4f} | '
            + ", ".join(f"{value:+.4f}" for value in deltas)
            + " |"
        )

lines += [
    "",
    "## Joint Interpretation",
    "",
    "- Bottleneck-LiteASK is the strongest candidate in the current six-model comparison: CAS IoU 0.7186, cross-group all-sample Macro IoU 0.2567, and original-only Macro IoU 0.2743.",
    "- In the three-seed screen, Bottleneck-LiteASK improves all-sample Macro IoU over U-Net++ by 0.0337 (p=0.0251) with all three deltas positive.",
    "- The five-seed robustness run reduces the Macro IoU delta to +0.0215 (p=0.0570, all samples) and +0.0181 (p=0.1610, original only); the effect is therefore a consistent positive trend, not yet confirmatory.",
    "- Five-seed paired results remain significant for CAS IoU (+0.0111, p=0.0289) and HD95 (-2.91 pixels, p=0.0113), while c3 IoU is not significant (p=0.1625).",
    "- The parameter cost is small relative to U-Net++: 9.758M versus 9.163M parameters, with essentially the same GFLOPs and batch-32 throughput.",
    "- Bottleneck-LiteASK improves the worst original group to IoU 0.1616, compared with 0.1270 for U-Net++ and 0.1226 for UNet.",
    "- Compared with UNet, the macro improvements are positive on average but not statistically stable across three seeds (all p>0.5); additional seeds are required.",
    "- Boundary F1@2 is 0.2104, essentially tied with U-Net++ (0.2128) and below UNet (0.2249); HD95 is 60.62, better than U-Net++ (63.74) but worse than UNet (59.45).",
    "- Heavy ASK-UNet++ remains slightly better on c3 (0.2550 vs 0.2356) but is worse on CAS, worse on full-group macro behavior, and uses more than twice the parameters.",
    "- The emerging paper story should focus on a parameter-efficient bottleneck SK design that improves CAS accuracy, HD95 and cross-group robustness over U-Net++, while explicitly reporting that macro IoU improvement remains marginal at five seeds.",
    "- Five-seed robustness report: D:\\landslide_unet_project\\reports\\five_seed_key_models_report.md",
    "",
    "## Files",
    "",
    "- Combined table: D:\\landslide_unet_project\\reports\\combined_model_assessment.csv",
    "- Extended c3 report: D:\\landslide_unet_project\\reports\\group_extended_experiment_report.md",
    "- Cross-group report: D:\\landslide_unet_project\\reports\\cross_group_experiment_report.md",
    "- Boundary report: D:\\landslide_unet_project\\reports\\boundary_metrics_report.md",
    "- Complexity report: D:\\landslide_unet_project\\reports\\model_complexity_report.md",
    "- Threshold report: D:\\landslide_unet_project\\reports\\threshold_sensitivity_report.md",
]
(REPORTS / "combined_model_assessment.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("wrote", REPORTS / "combined_model_assessment.md")
