"""Summarize source-domain augmentation ablation."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt

import os as _repo_os
PROJECT = Path(_repo_os.environ.get("LANDSLIDE_PROJECT_ROOT", Path(__file__).resolve().parents[1])).resolve()
REPORTS = PROJECT / "reports"
FIGURES = PROJECT / "figures"
REGIONS = ["hokkaido_iburi_tobu", "lombok", "palu"]
RAW_SPECS = {
    "none": REPORTS / "p4_augmentation_raw" / "none",
    "current": REPORTS / "p4_source_overfit_raw",
    "strong": REPORTS / "p4_augmentation_raw" / "strong",
}


def main() -> None:
    records = []
    for mode, raw_dir in RAW_SPECS.items():
        for path in sorted(raw_dir.glob("epoch_*.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            region_values = {
                region: float(data["regions"][region]["threshold_metrics"]["0.5"]["iou"])
                for region in REGIONS
            }
            records.append({
                "mode": mode,
                "epoch": int(data["epoch"]),
                "cas_val_iou": float(data["cas_val"]["iou"]),
                "target_macro_iou": sum(region_values.values()) / len(region_values),
                **{f"{region}_iou": value for region, value in region_values.items()},
            })
    records.sort(key=lambda row: (list(RAW_SPECS).index(row["mode"]), row["epoch"]))
    with (REPORTS / "p4_augmentation_curves.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    summaries = []
    for mode in RAW_SPECS:
        entries = [row for row in records if row["mode"] == mode]
        source_best = max(entries, key=lambda row: row["cas_val_iou"])
        target_best = max(entries, key=lambda row: row["target_macro_iou"])
        final = max(entries, key=lambda row: row["epoch"])
        summaries.append({
            "mode": mode,
            "source_best_epoch": source_best["epoch"],
            "source_best_iou": source_best["cas_val_iou"],
            "target_best_epoch": target_best["epoch"],
            "target_best_macro_iou": target_best["target_macro_iou"],
            "epoch50_cas_iou": final["cas_val_iou"],
            "epoch50_target_macro_iou": final["target_macro_iou"],
            "epoch50_target_peak_drop": target_best["target_macro_iou"] - final["target_macro_iou"],
            "epoch50_source_target_gap": final["cas_val_iou"] - final["target_macro_iou"],
        })

    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=180)
    for mode in RAW_SPECS:
        entries = [row for row in records if row["mode"] == mode]
        epochs = [row["epoch"] for row in entries]
        axes[0].plot(epochs, [row["cas_val_iou"] for row in entries], marker="o", linewidth=2, label=mode)
        axes[1].plot(epochs, [row["target_macro_iou"] for row in entries], marker="o", linewidth=2, label=mode)
    axes[0].set_title("CAS validation")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("IoU")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False)
    axes[1].set_title("Target Macro IoU")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("IoU")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False)
    fig.tight_layout()
    figure_path = FIGURES / "p4_augmentation_resunet_seed42.png"
    fig.savefig(figure_path, bbox_inches="tight")
    plt.close(fig)

    lines = [
        "# P4-C: Source Augmentation Ablation",
        "",
        "Model: ResUNet, seed 42, 128x128, 100% source data. `current` is the original geometric augmentation used by the main benchmark. `strong` adds brightness/contrast, Gaussian blur, and scale perturbation.",
        "",
        "| Augmentation | Epoch-50 CAS IoU | Epoch-50 target Macro | Source-best epoch | Target-best epoch | Epoch-50 peak drop | Source-target gap |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['mode']} | {row['epoch50_cas_iou']:.4f} | {row['epoch50_target_macro_iou']:.4f} | "
            f"{row['source_best_epoch']} | {row['target_best_epoch']} | {row['epoch50_target_peak_drop']:+.4f} | {row['epoch50_source_target_gap']:.4f} |"
        )
    lines += [
        "",
        "Interpretation: strong augmentation is useful only if it improves target-domain peak or epoch-50 performance and reduces the post-peak drop without causing a large source-validation collapse.",
        "",
        f"Figure: {figure_path}",
    ]
    (REPORTS / "p4_augmentation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
