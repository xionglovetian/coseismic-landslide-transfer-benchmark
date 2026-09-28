"""E0 audit of threshold sensitivity, prevalence, uncertainty, and rank stability."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

import os as _repo_os
PROJECT = Path(_repo_os.environ.get("LANDSLIDE_PROJECT_ROOT", Path(__file__).resolve().parents[1])).resolve()
REPORTS = PROJECT / "reports"
OUTPUTS = PROJECT / "outputs"
THRESHOLDS = ["0.3", "0.4", "0.5", "0.6", "0.7"]
REGIONS = ["wenchuan", "jiuzhai_valley", "moxitaidi", "longxi_river", "hokkaido_iburi_tobu", "lombok", "palu"]
TARGET_REGIONS = ["hokkaido_iburi_tobu", "lombok", "palu"]
REGION_LABELS = {
    "wenchuan": "Wenchuan",
    "jiuzhai_valley": "Jiuzhai Valley",
    "moxitaidi": "Moxitaidi",
    "longxi_river": "Longxi River",
    "hokkaido_iburi_tobu": "Hokkaido",
    "lombok": "Lombok",
    "palu": "Palu",
}
SEEDS = [42, 2026, 777]
MODEL_SPECS = {
    "Bottleneck-LiteASK": ("group_bottleneckliteaskunetpp", "BottleneckLiteASKUNetPlusPlus"),
    "ResUNet": ("bench_v2_resunet", "ResUNet"),
    "DeepLabV3+": ("bench_v2_deeplabv3plus", "DeepLabV3Plus"),
    "SegFormer-B0": ("bench_v2_segformerb0", "SegFormerB0"),
}


def mean(values) -> float:
    values = list(values)
    return float(sum(values) / len(values)) if values else float("nan")


def std(values) -> float:
    values = list(values)
    return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def read_manifest_counts():
    counts = defaultdict(int)
    prevalence = defaultdict(list)
    gsd = defaultdict(list)
    path = PROJECT / "data" / "processed" / "benchmark_v2_regions_512" / "manifest_benchmark_v2.csv"
    for row in csv.DictReader(path.open(encoding="utf-8-sig")):
        counts[row["region"]] += 1
        prevalence[row["region"]].append(float(row["foreground_fraction"]))
        gsd[row["region"]].append(float(row["ground_resolution_m"]))
    return counts, {key: mean(values) for key, values in prevalence.items()}, {key: sorted(set(values)) for key, values in gsd.items()}


def reconstruct_counts(metrics: dict, total_pixels: int) -> dict:
    precision = float(metrics["precision"])
    recall = float(metrics["recall"])
    accuracy = float(metrics["accuracy"])
    denominator = 1.0 / precision + 1.0 / recall - 2.0
    tp = (1.0 - accuracy) * total_pixels / denominator
    fp = tp * (1.0 / precision - 1.0)
    fn = tp * (1.0 / recall - 1.0)
    tn = total_pixels - tp - fp - fn
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def derived_metrics(metrics: dict, total_pixels: int) -> dict:
    counts = reconstruct_counts(metrics, total_pixels)
    tp, fp, fn, tn = counts["tp"], counts["fp"], counts["fn"], counts["tn"]
    mcc_den = math.sqrt(max((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn), 0.0))
    mcc = (tp * tn - fp * fn) / mcc_den if mcc_den else 0.0
    sensitivity = tp / max(tp + fn, 1e-12)
    specificity = tn / max(tn + fp, 1e-12)
    foreground_iou = tp / max(tp + fp + fn, 1e-12)
    background_iou = tn / max(tn + fp + fn, 1e-12)
    return {
        "iou": float(metrics["iou"]),
        "dice": float(metrics["dice"]),
        "precision": float(metrics["precision"]),
        "recall": float(metrics["recall"]),
        "f1": float(metrics["f1"]),
        "accuracy": float(metrics["accuracy"]),
        "mcc": mcc,
        "balanced_accuracy": 0.5 * (sensitivity + specificity),
        "youden_j": sensitivity + specificity - 1.0,
        "balanced_iou": 0.5 * (foreground_iou + background_iou),
        "background_iou": background_iou,
    }

def zero_shot_records(counts):
    rows = []
    for model, (prefix, _) in MODEL_SPECS.items():
        for seed in SEEDS:
            path = REPORTS / "benchmark_v2_zero_shot_raw" / f"{prefix}_seed{seed}.json"
            data = json.loads(path.read_text(encoding="utf-8"))
            for region in REGIONS:
                total_pixels = counts[region] * 128 * 128
                for threshold in THRESHOLDS:
                    raw = data["groups"][region]["threshold_metrics"][threshold]
                    metrics = derived_metrics(raw, total_pixels)
                    rows.append({"source": "zero-shot", "model": model, "seed": seed, "region": region, "region_label": REGION_LABELS[region], "threshold": float(threshold), **metrics})
    return rows


def control_records(counts):
    rows = []
    for seed in SEEDS:
        path = REPORTS / "benchmark_v2_control_raw" / f"cas_retrain20_seed{seed}.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        for region in REGIONS:
            total_pixels = counts[region] * 128 * 128
            for threshold in THRESHOLDS:
                raw = data["groups"][region]["threshold_metrics"][threshold]
                metrics = derived_metrics(raw, total_pixels)
                rows.append({"source": "cas-retrain20", "model": "Bottleneck-LiteASK", "seed": seed, "region": region, "region_label": REGION_LABELS[region], "threshold": float(threshold), **metrics})
    return rows


def extended_records(counts):
    rows = []
    for seed in SEEDS:
        path = OUTPUTS / f"bench_v2_extended_uniform_seed{seed}" / "target_metrics.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        for region in TARGET_REGIONS:
            total_pixels = counts[region] * 128 * 128
            for threshold in THRESHOLDS:
                raw = data["target"][region]["threshold_metrics"][threshold]
                metrics = derived_metrics(raw, total_pixels)
                rows.append({"source": "fixed-source-uniform", "model": "Bottleneck-LiteASK", "seed": seed, "region": region, "region_label": REGION_LABELS[region], "threshold": float(threshold), **metrics})
    return rows


def source_validation_iou():
    values = {}
    for model, (prefix, _) in MODEL_SPECS.items():
        for seed in SEEDS:
            path = OUTPUTS / f"{prefix}_seed{seed}" / "metrics.json"
            data = json.loads(path.read_text(encoding="utf-8"))
            values[(model, seed)] = float(data["cas_val"]["iou"])
    return values

def main() -> None:
    counts, prevalence, _ = read_manifest_counts()
    threshold_rows = zero_shot_records(counts) + control_records(counts) + extended_records(counts)
    threshold_fields = list(threshold_rows[0].keys())
    with (REPORTS / "e0_threshold_metrics.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=threshold_fields)
        writer.writeheader()
        writer.writerows(threshold_rows)

    # Region-level prevalence and IoU association at the fixed 0.5 threshold.
    prevalence_rows = []
    zero_05 = [row for row in threshold_rows if row["source"] == "zero-shot" and row["threshold"] == 0.5]
    for region in REGIONS:
        rows = [row for row in zero_05 if row["region"] == region]
        prevalence_rows.append({
            "region": region,
            "region_label": REGION_LABELS[region],
            "mean_foreground_fraction": prevalence[region],
            "mean_iou": mean(row["iou"] for row in rows),
            "std_iou": std(row["iou"] for row in rows),
            "mean_mcc": mean(row["mcc"] for row in rows),
            "mean_balanced_iou": mean(row["balanced_iou"] for row in rows),
            "mean_balanced_accuracy": mean(row["balanced_accuracy"] for row in rows),
        })
    with (REPORTS / "e0_prevalence_metrics.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(prevalence_rows[0].keys()))
        writer.writeheader()
        writer.writerows(prevalence_rows)
    prev_values = [row["mean_foreground_fraction"] for row in prevalence_rows]
    iou_values = [row["mean_iou"] for row in prevalence_rows]
    spearman = stats.spearmanr(prev_values, iou_values)
    leave_one_out = []
    for index in range(len(prevalence_rows)):
        subset_prev = [value for pos, value in enumerate(prev_values) if pos != index]
        subset_iou = [value for pos, value in enumerate(iou_values) if pos != index]
        leave_one_out.append(float(stats.spearmanr(subset_prev, subset_iou).statistic))

    # Architecture-level rank uncertainty across seeds and regions.
    scores = defaultdict(dict)
    for region in REGIONS:
        for model in MODEL_SPECS:
            for seed in SEEDS:
                rows = [row for row in zero_05 if row["region"] == region and row["model"] == model and row["seed"] == seed]
                scores[(model, seed)][region] = mean(row["iou"] for row in rows)
    source_val = source_validation_iou()
    macro_rows = []
    source_target_corrs = []
    for model in MODEL_SPECS:
        for seed in SEEDS:
            macro_iou = mean(scores[(model, seed)][region] for region in REGIONS)
            macro_rows.append({"model": model, "seed": seed, "source_val_iou": source_val[(model, seed)], "target_macro_iou": macro_iou})
    for seed in SEEDS:
        models = list(MODEL_SPECS)
        source_rank_values = [source_val[(model, seed)] for model in models]
        target_rank_values = [mean(scores[(model, seed)][region] for region in REGIONS) for model in models]
        corr = stats.spearmanr(source_rank_values, target_rank_values)
        loo = []
        for region_index in range(len(REGIONS)):
            source_values = source_rank_values
            target_values = [mean(scores[(model, seed)][region] for pos, region in enumerate(REGIONS) if pos != region_index) for model in models]
            loo.append(float(stats.spearmanr(source_values, target_values).statistic))
        source_target_corrs.append({
            "seed": seed,
            "spearman_rho": float(corr.statistic),
            "p_value": float(corr.pvalue),
            "leave_one_region_out_min": min(loo),
            "leave_one_region_out_max": max(loo),
        })

    rng = np.random.default_rng(20260924)
    models = list(MODEL_SPECS)
    rank_sum = {model: 0.0 for model in models}
    best_count = {model: 0 for model in models}
    iterations = 10000
    for _ in range(iterations):
        sampled_regions = rng.choice(REGIONS, size=len(REGIONS), replace=True)
        seed_ranks = []
        for seed in SEEDS:
            model_means = {model: mean(scores[(model, seed)][region] for region in sampled_regions) for model in models}
            order = sorted(models, key=lambda model: model_means[model], reverse=True)
            seed_ranks.append({model: order.index(model) + 1 for model in models})
            best_count[order[0]] += 1
        for model in models:
            rank_sum[model] += mean(rank[model] for rank in seed_ranks)
    rank_rows = []
    for model in models:
        rank_rows.append({
            "model": model,
            "mean_rank": rank_sum[model] / iterations,
            "probability_rank1": best_count[model] / (iterations * len(SEEDS)),
        })
    with (REPORTS / "e0_architecture_rank.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rank_rows[0].keys()))
        writer.writeheader()
        writer.writerows(rank_rows)
    # Paired transfer deltas for the fixed-source uniform regime versus the CAS continuation control.
    control = {(row["seed"], row["region"], row["threshold"]): row for row in threshold_rows if row["source"] == "cas-retrain20"}
    pooled = {(row["seed"], row["region"], row["threshold"]): row for row in threshold_rows if row["source"] == "fixed-source-uniform"}
    transfer_rows = []
    for seed in SEEDS:
        for region in TARGET_REGIONS:
            for threshold in [float(value) for value in THRESHOLDS]:
                before = control[(seed, region, threshold)]
                after = pooled[(seed, region, threshold)]
                transfer_rows.append({
                    "seed": seed,
                    "region": region,
                    "region_label": REGION_LABELS[region],
                    "threshold": threshold,
                    "control_iou": before["iou"],
                    "pooled_iou": after["iou"],
                    "delta_iou": after["iou"] - before["iou"],
                    "control_mcc": before["mcc"],
                    "pooled_mcc": after["mcc"],
                    "delta_mcc": after["mcc"] - before["mcc"],
                })
    with (REPORTS / "e0_transfer_deltas.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(transfer_rows[0].keys()))
        writer.writeheader()
        writer.writerows(transfer_rows)

    transfer_summary = []
    for region in TARGET_REGIONS:
        for threshold in [float(value) for value in THRESHOLDS]:
            entries = [row for row in transfer_rows if row["region"] == region and row["threshold"] == threshold]
            deltas = [row["delta_iou"] for row in entries]
            transfer_summary.append({
                "region": region,
                "region_label": REGION_LABELS[region],
                "threshold": threshold,
                "delta_iou_mean": mean(deltas),
                "delta_iou_std": std(deltas),
                "positive_seeds": sum(value > 0 for value in deltas),
                "material_seeds": sum(value >= 0.01 for value in deltas),
            })

    with (REPORTS / "e0_transfer_threshold_summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(transfer_summary[0].keys()))
        writer.writeheader()
        writer.writerows(transfer_summary)

    fixed_rows = [row for row in transfer_rows if row["threshold"] == 0.5]
    by_seed_region = defaultdict(list)
    for row in fixed_rows:
        by_seed_region[row["seed"]].append(row["delta_iou"])
    boot_means = []
    rng = np.random.default_rng(20260924)
    for _ in range(10000):
        sampled_seeds = rng.choice(SEEDS, size=len(SEEDS), replace=True)
        sampled_values = []
        for seed in sampled_seeds:
            values = by_seed_region[int(seed)]
            sampled_values.extend(rng.choice(values, size=len(values), replace=True))
        boot_means.append(float(np.mean(sampled_values)))
    bootstrap = {
        "mean_delta_iou": mean(row["delta_iou"] for row in fixed_rows),
        "ci95_low": float(np.percentile(boot_means, 2.5)),
        "ci95_high": float(np.percentile(boot_means, 97.5)),
        "probability_delta_positive": float(np.mean(np.asarray(boot_means) > 0)),
        "probability_delta_at_least_0.01": float(np.mean(np.asarray(boot_means) >= 0.01)),
        "iterations": 10000,
    }
    # Region-level threshold sensitivity of the mean zero-shot metrics.
    region_threshold_rows = []
    for region in REGIONS:
        values = []
        selected = [row for row in threshold_rows if row["source"] == "zero-shot" and row["region"] == region]
        for threshold in [float(value) for value in THRESHOLDS]:
            entries = [row for row in selected if row["threshold"] == threshold]
            values.append({
                "threshold": threshold,
                "mean_iou": mean(row["iou"] for row in entries),
                "mean_mcc": mean(row["mcc"] for row in entries),
                "mean_balanced_iou": mean(row["balanced_iou"] for row in entries),
            })
        best_iou = max(values, key=lambda row: row["mean_iou"])
        best_mcc = max(values, key=lambda row: row["mean_mcc"])
        region_threshold_rows.append({
            "region": region,
            "region_label": REGION_LABELS[region],
            "iou_0.3": values[0]["mean_iou"],
            "iou_0.5": values[2]["mean_iou"],
            "iou_0.7": values[4]["mean_iou"],
            "iou_range": max(row["mean_iou"] for row in values) - min(row["mean_iou"] for row in values),
            "best_iou_threshold": best_iou["threshold"],
            "mcc_range": max(row["mean_mcc"] for row in values) - min(row["mean_mcc"] for row in values),
            "best_mcc_threshold": best_mcc["threshold"],
        })
    with (REPORTS / "e0_region_threshold_sensitivity.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(region_threshold_rows[0].keys()))
        writer.writeheader()
        writer.writerows(region_threshold_rows)

    lines = [
        "# E0.2-E0.3: Threshold, Prevalence, Transfer, and Rank Audit",
        "",
        "All metrics use the existing seed-42/2026/777 benchmark-v2 raw JSON. Threshold metrics are recomputed into MCC, balanced accuracy, and balanced IoU from precision, recall, accuracy, and the known 128x128 pixel count; no new model inference was run.",
        "",
        "## Prevalence Association at Threshold 0.5",
        "",
        "| Region | Mean foreground fraction | Mean zero-shot IoU | Mean MCC | Mean balanced IoU |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in prevalence_rows:
        lines.append(f"| {row['region_label']} | {row['mean_foreground_fraction']:.4f} | {row['mean_iou']:.4f} | {row['mean_mcc']:.4f} | {row['mean_balanced_iou']:.4f} |")
    lines += [
        "",
        f"Spearman correlation across seven regions: rho={spearman.statistic:.3f}, p={spearman.pvalue:.4f}. Leave-one-region-out rho range: {min(leave_one_out):.3f} to {max(leave_one_out):.3f}.",
        "",
        "This is an association, not evidence that prevalence causes regional difficulty. It must not be reported as a causal mechanism.",
        "",
        "## Threshold Sensitivity",
        "",
        "| Region | IoU@0.3 | IoU@0.5 | IoU@0.7 | IoU range | Best IoU threshold | MCC range |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in region_threshold_rows:
        lines.append(f"| {row['region_label']} | {row['iou_0.3']:.4f} | {row['iou_0.5']:.4f} | {row['iou_0.7']:.4f} | {row['iou_range']:.4f} | {row['best_iou_threshold']:.1f} | {row['mcc_range']:.4f} |")
    lines += [
        "",
        "## Architecture Rank at Threshold 0.5",
        "",
        "| Model | Mean rank | Probability rank 1 |",
        "|---|---:|---:|",
    ]
    for row in rank_rows:
        lines.append(f"| {row['model']} | {row['mean_rank']:.3f} | {row['probability_rank1']:.3f} |")
    lines += ["", "| Seed | Spearman source-vs-target rho | p | LOO-region rho min | LOO-region rho max |", "|---:|---:|---:|---:|---:|"]
    for row in source_target_corrs:
        lines.append(f"| {row['seed']} | {row['spearman_rho']:.3f} | {row['p_value']:.3f} | {row['leave_one_region_out_min']:.3f} | {row['leave_one_region_out_max']:.3f} |")
    lines += [
        "",
        "## Source-Expansion Transfer Delta at Threshold 0.5",
        "",
        "| Region | Delta IoU mean | SD across seeds | Positive seeds | Seeds >= +0.01 |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in transfer_summary:
        if row["threshold"] != 0.5:
            continue
        lines.append(f"| {row['region_label']} | {row['delta_iou_mean']:+.4f} | {row['delta_iou_std']:.4f} | {row['positive_seeds']}/3 | {row['material_seeds']}/3 |")
    lines += [
        "",
        "## Transfer Sign Across Thresholds",
        "",
        "| Region | Threshold | Mean delta IoU | Positive seeds | Seeds >= +0.01 |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in transfer_summary:
        lines.append(f"| {row['region_label']} | {row['threshold']:.1f} | {row['delta_iou_mean']:+.4f} | {row['positive_seeds']}/3 | {row['material_seeds']}/3 |")
    lines += [
        "",
        f"Hierarchical seed/region bootstrap of the three-region mean delta: {bootstrap['mean_delta_iou']:+.4f} [{bootstrap['ci95_low']:+.4f}, {bootstrap['ci95_high']:+.4f}], P(delta>0)={bootstrap['probability_delta_positive']:.3f}, P(delta>=0.01)={bootstrap['probability_delta_at_least_0.01']:.3f}.",
        "",
        "The transfer delta is reported here only as an exploratory epoch-matched result because E0.1 showed a 6.34x exposure mismatch. It cannot support the causal added-region claim.",
        "",
        "## Required Claim Changes",
        "",
        "- Replace the prevalence-dominance claim with an association claim.",
        "- Report threshold curves and MCC/balanced metrics alongside IoU.",
        "- Report per-seed paired transfer deltas and intervals.",
        "- Replace the retention-ratio headline with absolute paired deltas.",
        "- Do not combine the four-region LORO target set with the three-region fixed-source target set in one aggregate.",
    ]
    (REPORTS / "e0_threshold_uncertainty_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("threshold rows", len(threshold_rows))
    print("transfer rows", len(transfer_rows))
    print("bootstrap", bootstrap)
    print("prevalence spearman", spearman.statistic, spearman.pvalue)


if __name__ == "__main__":
    main()
