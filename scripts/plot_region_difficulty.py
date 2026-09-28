"""Plot external-region descriptors and available LORO performance."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import os as _repo_os
PROJECT = Path(_repo_os.environ.get("LANDSLIDE_PROJECT_ROOT", Path(__file__).resolve().parents[1])).resolve()
REPORTS = PROJECT / "reports"
FIGURES = PROJECT / "figures"
DIFFICULTY = REPORTS / "external_region_difficulty.csv"
LORO = REPORTS / "loro_benchmark_summary.csv"
OUTPUT = FIGURES / "external_region_difficulty.png"
METHOD_ORDER = [
    "CAS-only zero-shot",
    "Uniform multi-source",
    "Balanced multi-source",
    "Balanced + feature alignment",
]
METHOD_COLORS = {
    "CAS-only zero-shot": "#4C72B0",
    "Uniform multi-source": "#DD8452",
    "Balanced multi-source": "#55A868",
    "Balanced + feature alignment": "#C44E52",
}


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def fnum(value):
    return None if value in (None, "", "NA") else float(value)


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    difficulty = read_csv(DIFFICULTY)
    summary = read_csv(LORO)
    by_region_method = {
        (row["region"], row["method"]): row for row in summary
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8), constrained_layout=True)
    ax = axes[0]
    x = np.array([100 * fnum(row["foreground_fraction_mean"]) for row in difficulty])
    y = np.array([fnum(row["zero_shot_iou"]) for row in difficulty])
    sizes = np.array([max(5.0, fnum(row["n_samples"]) / 90.0) for row in difficulty])
    colors = np.array([fnum(row["perimeter_area_ratio_mean"]) for row in difficulty])
    scatter = ax.scatter(x, y, s=sizes, c=colors, cmap="viridis_r", edgecolor="black", linewidth=0.7, zorder=3)
    for row, x_value, y_value in zip(difficulty, x, y):
        ax.annotate(row["region_label"], (x_value, y_value), xytext=(6, 5), textcoords="offset points", fontsize=9)
        uniform_iou = fnum(row["uniform_iou"])
        if uniform_iou is not None:
            ax.annotate(
                "", xy=(x_value, uniform_iou), xytext=(x_value, y_value),
                arrowprops={"arrowstyle": "-|>", "color": "#DD8452", "lw": 1.5}, zorder=2,
            )
    ax.set_xlabel("Mean foreground fraction (%)")
    ax.set_ylabel("IoU")
    ax.set_title("Region descriptors and zero-shot difficulty")
    ax.grid(alpha=0.2)
    colorbar = fig.colorbar(scatter, ax=ax, pad=0.02)
    colorbar.set_label("Perimeter / sqrt(area)")

    ax = axes[1]
    regions = [row["region"] for row in difficulty]
    labels = [row["region_label"] for row in difficulty]
    x_positions = np.arange(len(regions))
    width = 0.18
    available_methods = [method for method in METHOD_ORDER if any((region, method) in by_region_method for region in regions)]
    for method_index, method in enumerate(available_methods):
        values = [
            fnum(by_region_method[(region, method)]["iou_mean"]) if (region, method) in by_region_method else np.nan
            for region in regions
        ]
        offset = (method_index - (len(available_methods) - 1) / 2) * width
        ax.bar(x_positions + offset, values, width, label=method, color=METHOD_COLORS[method])
    ax.set_xticks(x_positions, labels, rotation=15, ha="right")
    ax.set_ylabel("Region mean IoU")
    ax.set_ylim(0, max(0.6, np.nanmax([fnum(row["iou_mean"]) for row in summary]) * 1.15))
    ax.set_title("Available LORO benchmark results")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(fontsize=8, frameon=False)

    fig.suptitle("External landslide segmentation regions", fontsize=14)
    fig.savefig(OUTPUT, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
