"""Summarize and plot epoch-wise source-domain overfitting diagnostics."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import os as _repo_os
PROJECT = Path(_repo_os.environ.get("LANDSLIDE_PROJECT_ROOT", Path(__file__).resolve().parents[1])).resolve()
RAW_DIR = PROJECT / "reports" / "p4_source_overfit_raw"
REPORTS = PROJECT / "reports"
FIGURES = PROJECT / "figures"
REGIONS = ["hokkaido_iburi_tobu", "lombok", "palu"]
LABELS = {"hokkaido_iburi_tobu": "Hokkaido", "lombok": "Lombok", "palu": "Palu"}


def main() -> None:
    records = []
    for path in sorted(RAW_DIR.glob("epoch_*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        row = {"epoch": int(data["epoch"]), "cas_val_iou": float(data["cas_val"]["iou"]), "cas_val_dice": float(data["cas_val"]["dice"])}
        for region in REGIONS:
            group = data["regions"][region]
            row[f"{region}_iou"] = float(group["threshold_metrics"]["0.5"]["iou"])
            row[f"{region}_dice"] = float(group["threshold_metrics"]["0.5"]["dice"])
            row[f"{region}_bf1_2"] = float(group["f1_2_mean"])
            row[f"{region}_bf1_4"] = float(group["f1_4_mean"])
            row[f"{region}_hd95"] = float(group["hd95_mean"])
        records.append(row)
    records.sort(key=lambda row: row["epoch"])
    if not records:
        raise RuntimeError(f"No epoch records in {RAW_DIR}")
    fields = list(records[0].keys())
    with (REPORTS / "p4_source_overfit_curves.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)

    epochs = [row["epoch"] for row in records]
    cas = [row["cas_val_iou"] for row in records]
    cas_peak = max(records, key=lambda row: row["cas_val_iou"])
    diagnostics = []
    for region in REGIONS:
        target = [row[f"{region}_iou"] for row in records]
        target_peak = max(records, key=lambda row: row[f"{region}_iou"])
        correlation = float(np.corrcoef(cas, target)[0, 1])
        late_delta = target[-1] - target[0]
        peak_to_final = target_peak[f"{region}_iou"] - target[-1]
        diagnostics.append(
            {
                "region": region,
                "region_label": LABELS[region],
                "target_peak_epoch": target_peak["epoch"],
                "target_peak_iou": target_peak[f"{region}_iou"],
                "final_iou": target[-1],
                "first_to_final_delta": late_delta,
                "peak_to_final_drop": peak_to_final,
                "source_target_iou_correlation": correlation,
            }
        )

    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=180)
    axes[0].plot(epochs, cas, marker="o", linewidth=2, label="CAS validation")
    axes[0].set_title("Source-domain validation")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("IoU")
    axes[0].grid(alpha=0.25)
    for region in REGIONS:
        axes[1].plot(epochs, [row[f"{region}_iou"] for row in records], marker="o", linewidth=2, label=LABELS[region])
    axes[1].set_title("Target-domain zero-shot")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("IoU")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False)
    fig.tight_layout()
    figure_path = FIGURES / "p4_source_overfit_resunet_seed42.png"
    fig.savefig(figure_path, bbox_inches="tight")
    plt.close(fig)

    lines = [
        "# P4-A: Epoch-wise Source Overfitting Diagnostic",
        "",
        "Model: ResUNet, seed 42, 128x128, PolyGHMDiceLoss. Checkpoints evaluated every 10 epochs on CAS validation and the full Hokkaido/Lombok/Palu target regions.",
        "",
        "## Curve Summary",
        "",
        "| Epoch | CAS Val IoU | Hokkaido IoU | Lombok IoU | Palu IoU |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in records:
        lines.append(
            f"| {row['epoch']} | {row['cas_val_iou']:.4f} | {row['hokkaido_iburi_tobu_iou']:.4f} | "
            f"{row['lombok_iou']:.4f} | {row['palu_iou']:.4f} |"
        )
    lines += [
        "",
        "## Trend Diagnostic",
        "",
        "| Region | Target peak epoch | Peak IoU | Final IoU | First-to-final delta | Peak-to-final drop | Corr(source val, target) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in diagnostics:
        lines.append(
            f"| {row['region_label']} | {row['target_peak_epoch']} | {row['target_peak_iou']:.4f} | {row['final_iou']:.4f} | "
            f"{row['first_to_final_delta']:+.4f} | {row['peak_to_final_drop']:+.4f} | {row['source_target_iou_correlation']:+.3f} |"
        )
    lines += [
        "",
        f"CAS validation peak: epoch {cas_peak['epoch']} with IoU {cas_peak['cas_val_iou']:.4f}.",
        "",
        "A source-overfitting pattern is supported only when source validation continues to improve while one or more target-domain curves peak earlier and then decline. With only five checkpoints, correlations and peak timing are descriptive, not inferential tests.",
        "",
        f"Figure: {figure_path}",
    ]
    (REPORTS / "p4_source_overfit_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("epochs", epochs)
    print("cas peak", cas_peak["epoch"], cas_peak["cas_val_iou"])
    print("diagnostics", diagnostics)


if __name__ == "__main__":
    main()
