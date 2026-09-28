import argparse
import csv
import json
import statistics
from pathlib import Path

from scipy import stats

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import os as _repo_os
PROJECT = Path(_repo_os.environ.get("LANDSLIDE_PROJECT_ROOT", Path(__file__).resolve().parents[1])).resolve()
OUTPUTS = PROJECT / "outputs"
REPORTS = PROJECT / "reports"
FIGURES = PROJECT / "figures"
SEEDS = [42, 2026, 777]
METRICS = ["iou", "dice", "f1", "precision", "recall"]
MODELS = [
    ("UNet", "unet"),
    ("NestedUNet", "nestedunet"),
    ("AS_UNet", "asunet"),
    ("U-Net++", "unetpp"),
    ("ASK-UNet++", "askunetpp"),
    ("Bottleneck-LiteASK", "bottleneckliteaskunetpp"),
]
MODEL_SLUGS = dict(MODELS)


def resolve_slug(model, seed):
    # Prefer the strict batch-32 AS_UNet rerun only when all three seeds exist.
    if model == "AS_UNet" and all(
        (OUTPUTS / f"group_asunet32_seed{value}" / "metrics.json").exists()
        for value in SEEDS
    ):
        return "asunet32"
    return MODEL_SLUGS[model]


def load_rows(allow_incomplete: bool):
    rows = []
    missing = []
    for model, slug in MODELS:
        for seed in SEEDS:
            metrics_path = OUTPUTS / f"group_{resolve_slug(model, seed)}_seed{seed}" / "metrics.json"
            if not metrics_path.exists():
                missing.append(str(metrics_path))
                continue
            data = json.loads(metrics_path.read_text(encoding="utf-8"))
            config_path = metrics_path.with_name("config.json")
            config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
            row = {
                "model": model,
                "seed": seed,
                "best_epoch": data["best_epoch"],
                "batch_size": config.get("batch_size"),
                "input_size": config.get("input_size"),
                "epochs": config.get("epochs"),
                "loss": config.get("loss"),
            }
            for split_key, prefix in [
                ("cas_val", "cas_val"),
                ("resunet_bfa_external_test", "external"),
            ]:
                for metric in METRICS:
                    row[f"{prefix}_{metric}"] = data[split_key][metric]
            rows.append(row)
    if missing and not allow_incomplete:
        raise SystemExit("Missing metrics:\n" + "\n".join(missing))
    return rows


def summarize(rows):
    summary = []
    for model, _ in MODELS:
        model_rows = [row for row in rows if row["model"] == model]
        if not model_rows:
            continue
        for prefix in ["cas_val", "external"]:
            for metric in METRICS:
                values = [row[f"{prefix}_{metric}"] for row in model_rows]
                summary.append(
                    {
                        "model": model,
                        "split": prefix,
                        "metric": metric,
                        "mean": statistics.mean(values),
                        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
                        "n": len(values),
                    }
                )
    return summary


def write_tables(rows, summary):
    fields = [
        "model",
        "seed",
        "best_epoch",
        "batch_size",
        "input_size",
        "epochs",
        "loss",
        *[f"cas_val_{metric}" for metric in METRICS],
        *[f"external_{metric}" for metric in METRICS],
    ]
    with (REPORTS / "group_model_seed_results.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    summary_fields = ["model", "split", "metric", "mean", "std", "n"]
    with (REPORTS / "group_model_seed_summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary)

    lines = [
        "# Group-Split Five-Model Multi-Seed Summary",
        "",
        f"Completed runs: {len(rows)}/{len(MODELS) * len(SEEDS)}",
        "",
        "## Protocol By Model",
        "",
        "| Model | Batch Size | Input | Epochs | Loss | Completed Seeds |",
        "|---|---:|---:|---:|---|---:|",
    ]
    for model, _ in MODELS:
        model_rows = [row for row in rows if row["model"] == model]
        if not model_rows:
            continue
        batches = ",".join(str(value) for value in sorted({row["batch_size"] for row in model_rows}, key=str))
        inputs = ",".join(str(value) for value in sorted({row["input_size"] for row in model_rows}, key=str))
        epochs = ",".join(str(value) for value in sorted({row["epochs"] for row in model_rows}, key=str))
        losses = ",".join(sorted({str(row["loss"]) for row in model_rows}))
        lines.append(f"| {model} | {batches} | {inputs} | {epochs} | {losses} | {len(model_rows)} |")
    lines += [
        "",
        "## Mean And Standard Deviation",
        "",
        "| Model | Split | Metric | Mean | Std | N |",
        "|---|---|---|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f'| {row["model"]} | {row["split"]} | {row["metric"]} | '
            f'{row["mean"]:.4f} | {row["std"]:.4f} | {row["n"]} |'
        )
    (REPORTS / "group_model_seed_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_figure(rows):
    model_names = [model for model, _ in MODELS if any(row["model"] == model for row in rows)]
    panels = [
        ("cas_val", "iou", "CAS Validation IoU"),
        ("external", "iou", "ResUNet-BFA c3 External IoU"),
        ("external", "dice", "ResUNet-BFA c3 External Dice"),
    ]
    colors = plt.cm.tab10.colors
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), constrained_layout=True)
    for ax, (split, metric, title) in zip(axes, panels):
        means = []
        stds = []
        for model in model_names:
            values = [row[f"{split}_{metric}"] for row in rows if row["model"] == model]
            means.append(statistics.mean(values))
            stds.append(statistics.stdev(values) if len(values) > 1 else 0.0)
        x = list(range(len(model_names)))
        ax.bar(x, means, yerr=stds, capsize=4, color=[colors[i % len(colors)] for i in x], alpha=0.82)
        for i, model in enumerate(model_names):
            values = [row[f"{split}_{metric}"] for row in rows if row["model"] == model]
            jitter = [i - 0.12 + 0.24 * j / max(len(values) - 1, 1) for j in range(len(values))]
            ax.scatter(jitter, values, color="black", s=22, zorder=3)
        ax.set_xticks(x, model_names, rotation=22, ha="right")
        ax.set_title(title)
        ax.set_ylabel(metric.upper())
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Group-Split Multi-Seed Model Comparison", fontsize=14)
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "group_model_seed_summary.png", dpi=200)
    plt.close(fig)


def stat(summary, model, split, metric, field):
    row = next(
        row
        for row in summary
        if row["model"] == model and row["split"] == split and row["metric"] == metric
    )
    return row[field]


def write_report(rows, summary):
    complete_models = [
        model
        for model, _ in MODELS
        if len([row for row in rows if row["model"] == model]) == len(SEEDS)
    ]
    lines = [
        "# Extended Group-Split Experiment Report",
        "",
        "Date: 2026-09-20",
        "",
        "## Completion Status",
        "",
        f"Completed runs: {len(rows)}/{len(MODELS) * len(SEEDS)}.",
        "",
        "## Protocol",
        "",
        "- Training data: CAS Moxi+Bijie subset (1795 train, 513 CAS validation).",
        "- External test: ResUNet-BFA held-out source group c3 (430 images).",
        "- Input size: 128 x 128.",
        "- Epochs: 50.",
        "- Seeds: 42, 2026, 777.",
        "- Loss: PolyGHMDiceLoss.",
        "- Optimizer: Adam, learning rate 1e-4, weight decay 1e-4.",
        "- Precision: BF16 mixed precision.",
        "- Validation interval: every 5 epochs; best checkpoint selected by CAS validation IoU.",
        "",
        "| Model | Batch Size | Loss | Completed Seeds |",
        "|---|---:|---|---:|",
    ]
    for model, _ in MODELS:
        model_rows = [row for row in rows if row["model"] == model]
        if not model_rows:
            continue
        batches = ",".join(str(value) for value in sorted({row["batch_size"] for row in model_rows}, key=str))
        losses = ",".join(sorted({str(row["loss"]) for row in model_rows}))
        lines.append(f"| {model} | {batches} | {losses} | {len(model_rows)} |")

    lines += [
        "",
        "## New Model Implementations",
        "",
        "- U-Net++: five-stage VGG-style encoder/decoder with the standard nested dense skip-connection topology.",
        "- ASK-UNet++: U-Net++ with Selective Kernel attention (parallel 3x3 and 5x5 branches, reduction=16, minimum hidden width=32) after every encoder stage.",
        "- Bottleneck-LiteASK: U-Net++ with depthwise-separable SK attention only on the encoder bottleneck; adds only about 0.6M parameters over U-Net++.\n- All models use one output head and the same unified dataset, loss, optimizer, augmentation, and evaluation code as the existing baselines.",
        "",
        "## Mean And Standard Deviation",
        "",
        "| Model | CAS IoU | CAS Dice | External IoU | External Dice |",
        "|---|---:|---:|---:|---:|",
    ]
    for model, _ in MODELS:
        model_rows = [row for row in rows if row["model"] == model]
        if not model_rows:
            continue
        lines.append(
            f"| {model} | "
            f"{stat(summary, model, 'cas_val', 'iou', 'mean'):.4f} ± {stat(summary, model, 'cas_val', 'iou', 'std'):.4f} | "
            f"{stat(summary, model, 'cas_val', 'dice', 'mean'):.4f} ± {stat(summary, model, 'cas_val', 'dice', 'std'):.4f} | "
            f"{stat(summary, model, 'external', 'iou', 'mean'):.4f} ± {stat(summary, model, 'external', 'iou', 'std'):.4f} | "
            f"{stat(summary, model, 'external', 'dice', 'mean'):.4f} ± {stat(summary, model, 'external', 'dice', 'std'):.4f} |"
        )

    if complete_models:
        lines += ["", "## Rankings", ""]
        for split, metric, label in [
            ("cas_val", "iou", "CAS validation IoU"),
            ("external", "iou", "External c3 IoU"),
            ("external", "dice", "External c3 Dice"),
        ]:
            ranked = sorted(
                complete_models,
                key=lambda model: stat(summary, model, split, metric, "mean"),
                reverse=True,
            )
            lines.append(f"- {label}: " + " > ".join(
                f"{model} ({stat(summary, model, split, metric, 'mean'):.4f})"
                for model in ranked
            ))

    lines += ["", "## Exploratory Paired Checks", "", "ASK-UNet++ is compared seed-by-seed with each baseline. With only three seeds these checks are exploratory and must not be presented as confirmatory significance tests.", "", "| Task metric | Comparator | Mean delta | Std delta | Paired p |", "|---|---|---:|---:|---:|"]
    for split, metric, label in [
        ("cas_val", "iou", "CAS IoU"),
        ("external", "iou", "External IoU"),
        ("external", "dice", "External Dice"),
    ]:
        ask_values = [
            row[f"{split}_{metric}"]
            for row in rows
            if row["model"] == "ASK-UNet++"
        ]
        for comparator in ["UNet", "U-Net++", "AS_UNet"]:
            comparator_values = [
                row[f"{split}_{metric}"]
                for row in rows
                if row["model"] == comparator
            ]
            if len(ask_values) < 2 or len(ask_values) != len(comparator_values):
                continue
            deltas = [a - b for a, b in zip(ask_values, comparator_values)]
            p_value = stats.ttest_rel(ask_values, comparator_values).pvalue
            lines.append(
                f"| {label} | {comparator} | {statistics.mean(deltas):.4f} | "
                f"{statistics.stdev(deltas):.4f} | {p_value:.4f} |"
            )

    as_batch_sizes = {
        row["batch_size"] for row in rows if row["model"] == "AS_UNet"
    }
    if 32 in as_batch_sizes:
        as_guardrail = "- The final AS_UNet row uses the batch-32 rerun, so all five models in the main comparison share batch size 32. The legacy batch-16 AS_UNet results remain in the original group_asunet_seed* directories."
    else:
        as_guardrail = "- Existing AS_UNet runs used batch size 16, while the other completed models use batch size 32. The final table therefore does not yet have a perfectly matched AS_UNet batch-size setting."
    lines += [
        "",
        "## Interpretation Guardrails",
        "",
        "- This remains a single held-out external source group (c3); do not describe it as final cross-dataset generalization.",
        as_guardrail,
        "- External seed variance is large, so mean and standard deviation must be reported together; selecting the best seed is not valid.",
        "- ASK-UNet++ follows the components specified in the handoff memo because no canonical public implementation was located during this run.",
        "",
        "## Files",
        "",
        "- Per-seed metrics: D:\\landslide_unet_project\\reports\\group_model_seed_results.csv",
        "- Mean/std table: D:\\landslide_unet_project\\reports\\group_model_seed_summary.csv",
        "- Summary text: D:\\landslide_unet_project\\reports\\group_model_seed_summary.md",
        "- Figure: D:\\landslide_unet_project\\figures\\group_model_seed_summary.png",
        "- Checkpoints: D:\\landslide_unet_project\\outputs\\group_*_seed*",
        "- Logs: D:\\landslide_unet_project\\logs\\group_*_seed*.log",
    ]
    (REPORTS / "group_extended_experiment_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    REPORTS.mkdir(parents=True, exist_ok=True)
    rows = load_rows(args.allow_incomplete)
    if not rows:
        raise SystemExit("No completed grouped model metrics found")
    rows.sort(key=lambda row: ([name for name, _ in MODELS].index(row["model"]), row["seed"]))
    summary = summarize(rows)
    write_tables(rows, summary)
    write_figure(rows)
    write_report(rows, summary)
    print(f"completed rows: {len(rows)}")
    print(f"wrote: {REPORTS / 'group_model_seed_summary.md'}")
    print(f"wrote: {FIGURES / 'group_model_seed_summary.png'}")


if __name__ == "__main__":
    main()
