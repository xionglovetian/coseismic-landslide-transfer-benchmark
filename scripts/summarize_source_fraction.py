"""Summarize source-domain data fraction ablation at epoch checkpoints."""

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
LABELS = {"hokkaido_iburi_tobu": "Hokkaido", "lombok": "Lombok", "palu": "Palu"}
EPOCH50_KEYS = {"hokkaido_iburi_tobu": "epoch50_hokkaido_iou", "lombok": "epoch50_lombok_iou", "palu": "epoch50_palu_iou"}
RAW_SPECS = {
    25: REPORTS / "p4_source_fraction_raw" / "frac_25",
    50: REPORTS / "p4_source_fraction_raw" / "frac_50",
    75: REPORTS / "p4_source_fraction_raw" / "frac_75",
    100: REPORTS / "p4_source_overfit_raw",
}


def main() -> None:
    records = []
    for fraction, raw_dir in RAW_SPECS.items():
        for path in sorted(raw_dir.glob("epoch_*.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            row = {
                "fraction": fraction,
                "epoch": int(data["epoch"]),
                "cas_val_iou": float(data["cas_val"]["iou"]),
                "target_macro_iou": 0.0,
            }
            target_values = []
            for region in REGIONS:
                group = data["regions"][region]
                value = float(group["threshold_metrics"]["0.5"]["iou"])
                row[f"{region}_iou"] = value
                row[f"{region}_bf1_2"] = float(group["f1_2_mean"])
                row[f"{region}_hd95"] = float(group["hd95_mean"])
                target_values.append(value)
            row["target_macro_iou"] = sum(target_values) / len(target_values)
            records.append(row)
    records.sort(key=lambda row: (row["fraction"], row["epoch"]))
    if not records:
        raise RuntimeError("No source fraction records found")
    with (REPORTS / "p4_source_fraction_curves.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    final_rows = []
    for fraction in sorted(RAW_SPECS):
        entries = [row for row in records if row["fraction"] == fraction]
        source_best = max(entries, key=lambda row: row["cas_val_iou"])
        target_best = max(entries, key=lambda row: row["target_macro_iou"])
        final = max(entries, key=lambda row: row["epoch"])
        final_rows.append({
            "fraction": fraction,
            "source_best_epoch": source_best["epoch"],
            "source_best_iou": source_best["cas_val_iou"],
            "target_best_epoch": target_best["epoch"],
            "target_best_macro_iou": target_best["target_macro_iou"],
            "epoch50_cas_iou": final["cas_val_iou"],
            "epoch50_target_macro_iou": final["target_macro_iou"],
            "epoch50_hokkaido_iou": final["hokkaido_iburi_tobu_iou"],
            "epoch50_lombok_iou": final["lombok_iou"],
            "epoch50_palu_iou": final["palu_iou"],
        })

    FIGURES.mkdir(parents=True, exist_ok=True)
    fractions = [row["fraction"] for row in final_rows]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=180)
    axes[0].plot(fractions, [row["epoch50_cas_iou"] for row in final_rows], marker="o", linewidth=2)
    axes[0].set_title("CAS validation at epoch 50")
    axes[0].set_xlabel("Source training fraction (%)")
    axes[0].set_ylabel("IoU")
    axes[0].grid(alpha=0.25)
    axes[1].plot(fractions, [row["epoch50_target_macro_iou"] for row in final_rows], marker="o", linewidth=2, label="Macro")
    for region in REGIONS:
        key = EPOCH50_KEYS[region]
        axes[1].plot(fractions, [row[key] for row in final_rows], marker="o", linewidth=1.5, label=LABELS[region])
    axes[1].set_title("Target-domain zero-shot at epoch 50")
    axes[1].set_xlabel("Source training fraction (%)")
    axes[1].set_ylabel("IoU")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False)
    fig.tight_layout()
    figure_path = FIGURES / "p4_source_fraction_resunet_seed42.png"
    fig.savefig(figure_path, bbox_inches="tight")
    plt.close(fig)

    lines = [
        "# P4-B: Source-Domain Data Fraction Ablation",
        "",
        "Model: ResUNet, seed 42, 128x128. Training fractions use the same source train root and validation set; subsets are deterministic random samples with subset seed 42.",
        "",
        "| Source fraction | Epoch-50 CAS IoU | Epoch-50 Hokkaido | Epoch-50 Lombok | Epoch-50 Palu | Epoch-50 target macro | Source-best epoch | Target-best epoch |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in final_rows:
        lines.append(
            f"| {row['fraction']}% | {row['epoch50_cas_iou']:.4f} | {row['epoch50_hokkaido_iou']:.4f} | "
            f"{row['epoch50_lombok_iou']:.4f} | {row['epoch50_palu_iou']:.4f} | {row['epoch50_target_macro_iou']:.4f} | "
            f"{row['source_best_epoch']} | {row['target_best_epoch']} |"
        )
    lines += [
        "",
        "Interpretation rule: if source validation improves with fraction but target macro does not, additional source data changes source fit more than cross-event transfer. If both improve, source scaling is also a transfer intervention.",
        "",
        f"Figure: {figure_path}",
    ]
    (REPORTS / "p4_source_fraction_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(final_rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
