"""Summarize E3 model/seed replication of the epoch-wise overfitting pattern."""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

import matplotlib.pyplot as plt

import os as _repo_os
PROJECT = Path(_repo_os.environ.get("LANDSLIDE_PROJECT_ROOT", Path(__file__).resolve().parents[1])).resolve()
REPORTS = PROJECT / "reports"
FIGURES = PROJECT / "figures"
REGIONS = ["hokkaido_iburi_tobu", "lombok", "palu"]
REGION_LABELS = {"hokkaido_iburi_tobu": "Hokkaido", "lombok": "Lombok", "palu": "Palu"}
RUNS = [
    ("ResUNet", "resunet", 42, REPORTS / "p4_source_overfit_raw"),
    ("ResUNet", "resunet", 2026, REPORTS / "e3_raw" / "bench_v2_e3_epochwise_resunet_seed2026"),
    ("ResUNet", "resunet", 777, REPORTS / "e3_raw" / "bench_v2_e3_epochwise_resunet_seed777"),
    ("SegFormer-B0", "segformerb0", 42, REPORTS / "e3_raw" / "bench_v2_e3_epochwise_segformerb0_seed42"),
    ("SegFormer-B0", "segformerb0", 2026, REPORTS / "e3_raw" / "bench_v2_e3_epochwise_segformerb0_seed2026"),
    ("SegFormer-B0", "segformerb0", 777, REPORTS / "e3_raw" / "bench_v2_e3_epochwise_segformerb0_seed777"),
]
EPOCHS = [20, 30, 50]
MATERIAL_DECLINE = 0.005


def mean(values):
    values = list(values)
    return float(statistics.fmean(values)) if values else float("nan")


def load_run(model, slug, seed, raw_dir):
    rows = []
    for epoch in EPOCHS:
        path = raw_dir / f"epoch_{epoch:03d}.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        target_values = {
            region: float(data["regions"][region]["threshold_metrics"]["0.5"]["iou"])
            for region in REGIONS
        }
        rows.append({
            "model": model,
            "slug": slug,
            "seed": seed,
            "epoch": epoch,
            "source_val_iou": float(data["cas_val"]["iou"]),
            "target_macro_iou": mean(target_values.values()),
            **{f"{region}_iou": value for region, value in target_values.items()},
        })
    return rows

def main() -> None:
    rows = []
    for model, slug, seed, raw_dir in RUNS:
        rows.extend(load_run(model, slug, seed, raw_dir))
    with (REPORTS / "e3_epochwise_per_run.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    diagnostics = []
    for model, slug, seed, _ in RUNS:
        entries = [row for row in rows if row["model"] == model and row["seed"] == seed]
        source_peak = max(entries, key=lambda row: row["source_val_iou"])
        target_peak = max(entries, key=lambda row: row["target_macro_iou"])
        final = max(entries, key=lambda row: row["epoch"])
        decline = target_peak["target_macro_iou"] - final["target_macro_iou"]
        diagnostics.append({
            "model": model,
            "seed": seed,
            "source_peak_epoch": source_peak["epoch"],
            "source_peak_iou": source_peak["source_val_iou"],
            "target_peak_epoch": target_peak["epoch"],
            "target_peak_macro_iou": target_peak["target_macro_iou"],
            "epoch50_target_macro_iou": final["target_macro_iou"],
            "target_peak_to_final_drop": decline,
            "source_final_best": source_peak["epoch"] == 50,
            "target_early_peak": target_peak["epoch"] < 50,
            "material_decline": decline >= MATERIAL_DECLINE,
            "pattern_support": source_peak["epoch"] == 50 and target_peak["epoch"] < 50 and decline >= MATERIAL_DECLINE,
        })
    with (REPORTS / "e3_epochwise_diagnostics.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(diagnostics[0].keys()))
        writer.writeheader()
        writer.writerows(diagnostics)

    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), dpi=180)
    for axis, (model, slug, seed, _) in zip(axes.flatten(), RUNS):
        entries = [row for row in rows if row["model"] == model and row["seed"] == seed]
        epochs = [row["epoch"] for row in entries]
        axis.plot(epochs, [row["source_val_iou"] for row in entries], marker="o", linewidth=2, label="CAS val")
        axis.plot(epochs, [row["target_macro_iou"] for row in entries], marker="o", linewidth=2, label="Target Macro")
        axis.set_title(f"{model}, seed {seed}")
        axis.set_xlabel("Epoch")
        axis.set_ylabel("IoU")
        axis.grid(alpha=0.25)
        axis.legend(frameon=False)
    fig.tight_layout()
    figure_path = FIGURES / "e3_epochwise_replication.png"
    fig.savefig(figure_path, bbox_inches="tight")
    plt.close(fig)

    supported = sum(row["pattern_support"] for row in diagnostics)
    lines = [
        "# E3: Model and Seed Replication of Epoch-wise Source Overfitting",
        "",
        "The prespecified pattern is: CAS validation peaks at epoch 50, target Macro IoU peaks at epoch 20 or 30, and the target then declines by at least 0.005.",
        "",
        "## Per-Run Diagnostics",
        "",
        "| Model | Seed | Source peak epoch | Target peak epoch | Target peak IoU | Epoch-50 target IoU | Peak-to-final drop | Pattern supported |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in diagnostics:
        lines.append(f"| {row['model']} | {row['seed']} | {row['source_peak_epoch']} | {row['target_peak_epoch']} | {row['target_peak_macro_iou']:.4f} | {row['epoch50_target_macro_iou']:.4f} | {row['target_peak_to_final_drop']:+.4f} | {row['pattern_support']} |")
    lines += [
        "",
        "## Decision",
        "",
        f"The pattern is supported in {supported}/6 model-seed runs. The prespecified reviewer-response threshold was at least 4/6 runs.",
        "",
        ("The prespecified threshold is met. Retain the temporal source-overfitting pattern as a replicated benchmark property, while keeping the causal few-shot claim separate based on E2." if supported >= 4 else "The prespecified threshold is not met. The temporal source-overfitting pattern is not replicated across models and seeds. Downgrade this mechanism claim to exploratory and report model/seed heterogeneity."),
        "",
        f"Figure: {figure_path}",
    ]
    (REPORTS / "e3_epochwise_replication_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("diagnostics", diagnostics)
    print("supported", supported)


if __name__ == "__main__":
    main()
