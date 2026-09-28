import csv
import json
import statistics
from pathlib import Path

from scipy import stats

import os as _repo_os
PROJECT = Path(_repo_os.environ.get("LANDSLIDE_PROJECT_ROOT", Path(__file__).resolve().parents[1])).resolve()
REPORTS = PROJECT / "reports"
OUTPUTS = PROJECT / "outputs"
SEEDS = [42, 2026, 777, 123, 2025]
REGIONS = ["a1", "a2", "a3", "b", "c1", "c2", "c3"]
MODELS = {
    "U-Net++": "group_unetpp",
    "Bottleneck-LiteASK": "group_bottleneckliteaskunetpp",
}


def load_seed_metrics(prefix, seed):
    train = json.loads((OUTPUTS / f"{prefix}_seed{seed}" / "metrics.json").read_text(encoding="utf-8"))
    cross = json.loads((REPORTS / "cross_group_raw" / f"{prefix}_seed{seed}.json").read_text(encoding="utf-8"))
    boundary = json.loads((REPORTS / "boundary_metrics_raw" / f"{prefix}_seed{seed}.json").read_text(encoding="utf-8"))
    threshold = json.loads((REPORTS / "threshold_sensitivity_raw" / f"{prefix}_seed{seed}.json").read_text(encoding="utf-8"))
    macro = {
        subset: statistics.mean(cross["groups"][subset][region]["iou"] for region in REGIONS)
        for subset in ["all", "original"]
    }
    boundary_macro = {
        "f1_2": statistics.mean(boundary["groups"][region]["f1_2_mean"] for region in REGIONS),
        "hd95": statistics.mean(boundary["groups"][region]["hd95_mean"] for region in REGIONS),
    }
    threshold_macro = {
        threshold_value: statistics.mean(threshold["groups"][region][threshold_value]["iou"] for region in REGIONS)
        for threshold_value in ["0.3", "0.4", "0.5", "0.6", "0.7"]
    }
    return {
        "cas_iou": train["cas_val"]["iou"],
        "c3_iou": train["resunet_bfa_external_test"]["iou"],
        "macro_all": macro["all"],
        "macro_original": macro["original"],
        "boundary_f1_2": boundary_macro["f1_2"],
        "hd95": boundary_macro["hd95"],
        "best_threshold": max(threshold_macro, key=threshold_macro.get),
        "best_threshold_iou": max(threshold_macro.values()),
        "threshold_iou": threshold_macro,
        "groups": cross["groups"],
    }


def summarize(values):
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0


data = {
    model: {seed: load_seed_metrics(prefix, seed) for seed in SEEDS}
    for model, prefix in MODELS.items()
}
rows = []
for model in MODELS:
    for seed in SEEDS:
        values = data[model][seed]
        rows.append({
            "model": model,
            "seed": seed,
            "cas_iou": values["cas_iou"],
            "c3_iou": values["c3_iou"],
            "macro_all_iou": values["macro_all"],
            "macro_original_iou": values["macro_original"],
            "boundary_f1_2": values["boundary_f1_2"],
            "hd95": values["hd95"],
            "best_threshold": values["best_threshold"],
            "best_threshold_iou": values["best_threshold_iou"],
        })
with (REPORTS / "five_seed_key_models_results.csv").open("w", newline="", encoding="utf-8-sig") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

metrics = [
    ("cas_iou", "CAS IoU"),
    ("c3_iou", "c3 IoU"),
    ("macro_all", "Macro IoU (all)"),
    ("macro_original", "Macro IoU (original)"),
    ("boundary_f1_2", "Boundary F1@2"),
    ("hd95", "HD95"),
]
summary_rows = []
for model in MODELS:
    for key, label in metrics:
        mean, std = summarize([data[model][seed][key] for seed in SEEDS])
        summary_rows.append({"model": model, "metric": label, "mean": mean, "std": std})
with (REPORTS / "five_seed_key_models_summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
    writer = csv.DictWriter(handle, fieldnames=["model", "metric", "mean", "std"])
    writer.writeheader()
    writer.writerows(summary_rows)

lines = [
    "# Five-Seed Key-Model Robustness",
    "",
    "Date: 2026-09-21",
    "",
    "Seeds: 42, 2026, 777, 123, 2025.",
    "",
    "## Mean And Standard Deviation",
    "",
    "| Model | CAS IoU | c3 IoU | Macro IoU (all) | Macro IoU (original) | BF1@2 | HD95 |",
    "|---|---:|---:|---:|---:|---:|---:|",
]
for model in MODELS:
    def stat(key):
        values = [data[model][seed][key] for seed in SEEDS]
        return summarize(values)
    cas = stat("cas_iou"); c3 = stat("c3_iou"); all_iou = stat("macro_all"); orig_iou = stat("macro_original"); bf1 = stat("boundary_f1_2"); hd = stat("hd95")
    lines.append(
        f'| {model} | {cas[0]:.4f} ± {cas[1]:.4f} | {c3[0]:.4f} ± {c3[1]:.4f} | '
        f'{all_iou[0]:.4f} ± {all_iou[1]:.4f} | {orig_iou[0]:.4f} ± {orig_iou[1]:.4f} | '
        f'{bf1[0]:.4f} ± {bf1[1]:.4f} | {hd[0]:.2f} ± {hd[1]:.2f} |'
    )

lines += ["", "## Paired Candidate Minus U-Net++", "", "| Metric | Mean delta | Std delta | Paired p | Delta by seed |", "|---|---:|---:|---:|---|"]
for key, label in metrics:
    candidate = [data["Bottleneck-LiteASK"][seed][key] for seed in SEEDS]
    baseline = [data["U-Net++"][seed][key] for seed in SEEDS]
    deltas = [a - b for a, b in zip(candidate, baseline)]
    p_value = stats.ttest_rel(candidate, baseline).pvalue
    lines.append(f'| {label} | {statistics.mean(deltas):+.4f} | {statistics.stdev(deltas):.4f} | {p_value:.4f} | ' + ", ".join(f"{value:+.4f}" for value in deltas) + " |")

lines += ["", "## Per-Group Original-Sample IoU", "", "| Model | " + " | ".join(REGIONS) + " |", "|---|" + "|".join(["---:"] * len(REGIONS)) + "|"]
for model in MODELS:
    values = [statistics.mean(data[model][seed]["groups"]["original"][region]["iou"] for seed in SEEDS) for region in REGIONS]
    lines.append(f'| {model} | ' + " | ".join(f"{value:.4f}" for value in values) + " |")

lines += ["", "## Best Macro Threshold By Seed", "", "| Model | Seed | Best Threshold | Macro IoU |", "|---|---:|---:|---:|"]
for model in MODELS:
    for seed in SEEDS:
        value = data[model][seed]
        lines.append(f'| {model} | {seed} | {value["best_threshold"]} | {value["best_threshold_iou"]:.4f} |')

lines += [
    "",
    "## Files",
    "",
    "- Per-seed table: D:\\landslide_unet_project\\reports\\five_seed_key_models_results.csv",
    "- Summary table: D:\\landslide_unet_project\\reports\\five_seed_key_models_summary.csv",
]
(REPORTS / "five_seed_key_models_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("wrote", REPORTS / "five_seed_key_models_report.md")
