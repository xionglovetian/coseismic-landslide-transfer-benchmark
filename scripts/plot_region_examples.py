"""Create a representative RGB/mask/overlay grid for external regions."""

from __future__ import annotations

import csv
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

import os as _repo_os
PROJECT = Path(_repo_os.environ.get("LANDSLIDE_PROJECT_ROOT", Path(__file__).resolve().parents[1])).resolve()
MANIFEST = PROJECT / "data" / "processed" / "external_regions_512" / "manifest_external.csv"
OUTPUT = PROJECT / "figures" / "external_region_examples.png"
REGIONS = [
    ("wenchuan", "Wenchuan"),
    ("jiuzhai_valley", "Jiuzhai Valley"),
    ("moxitaidi", "Moxitaidi"),
    ("longxi_river", "Longxi River"),
]


def main():
    with MANIFEST.open(encoding="utf-8-sig", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if int(row.get("eligible_external", 1)) == 1]

    fig, axes = plt.subplots(len(REGIONS), 3, figsize=(9.2, 12.4), constrained_layout=True)
    for row_index, (region, label) in enumerate(REGIONS):
        candidates = [row for row in rows if row["region"] == region]
        foreground = np.array([float(row["foreground_fraction"]) for row in candidates])
        median_fraction = float(np.median(foreground))
        selected = min(candidates, key=lambda row: abs(float(row["foreground_fraction"]) - median_fraction))

        image = cv2.cvtColor(cv2.imread(selected["image_path"]), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(selected["mask_path"], cv2.IMREAD_GRAYSCALE) > 0
        overlay = image.copy()
        overlay[mask] = (0.45 * overlay[mask] + 0.55 * np.array([255, 40, 40])).astype(np.uint8)

        axes[row_index, 0].imshow(image)
        axes[row_index, 0].set_title(f"{label}: RGB")
        axes[row_index, 1].imshow(mask, cmap="gray")
        axes[row_index, 1].set_title(f"Mask ({100 * float(selected['foreground_fraction']):.2f}% foreground)")
        axes[row_index, 2].imshow(overlay)
        axes[row_index, 2].set_title("Overlay")
        for column in range(3):
            axes[row_index, column].axis("off")

    fig.suptitle("Representative external-region samples near each region median foreground fraction", fontsize=13)
    fig.savefig(OUTPUT, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
