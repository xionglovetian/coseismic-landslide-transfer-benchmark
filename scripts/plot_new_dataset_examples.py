"""Plot representative RGB/mask/overlay examples from the newly added archives."""

from __future__ import annotations

import io
import posixpath
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import os as _repo_os
PROJECT = Path(_repo_os.environ.get("LANDSLIDE_PROJECT_ROOT", Path(__file__).resolve().parents[1])).resolve()
OUTPUT = PROJECT / "figures" / "new_dataset_examples.png"
ARCHIVES = [
    (r"C:\Users\ASUS\Desktop\数据集\Hokkaido Iburi-Tobu.zip", "Hokkaido Iburi-Tobu"),
    (r"C:\Users\ASUS\Desktop\数据集\Lombok.zip", "Lombok"),
    (r"C:\Users\ASUS\Desktop\数据集\palu.zip", "Palu"),
]


def main():
    fig, axes = plt.subplots(len(ARCHIVES), 3, figsize=(9.2, 9.8), constrained_layout=True)
    for row_index, (archive, label) in enumerate(ARCHIVES):
        with zipfile.ZipFile(archive) as archive_handle:
            mask_names = sorted(
                name for name in archive_handle.namelist()
                if name.startswith("mask/") and name.lower().endswith(".tif")
            )
            candidates = []
            for mask_name in mask_names:
                mask = np.asarray(Image.open(io.BytesIO(archive_handle.read(mask_name)))) > 0
                candidates.append((float(mask.mean()), mask_name))
            median_fraction = float(np.median([item[0] for item in candidates]))
            _, selected_mask = min(candidates, key=lambda item: abs(item[0] - median_fraction))
            stem = posixpath.splitext(posixpath.basename(selected_mask))[0]
            image = np.asarray(Image.open(io.BytesIO(archive_handle.read(f"img/{stem}.tif"))).convert("RGB"))
            mask = np.asarray(Image.open(io.BytesIO(archive_handle.read(selected_mask)))) > 0
        overlay = image.copy()
        overlay[mask] = (0.45 * overlay[mask] + 0.55 * np.array([255, 40, 40])).astype(np.uint8)

        axes[row_index, 0].imshow(image)
        axes[row_index, 0].set_title(f"{label}: RGB")
        axes[row_index, 1].imshow(mask, cmap="gray")
        axes[row_index, 1].set_title(f"Mask ({100 * mask.mean():.2f}% foreground)")
        axes[row_index, 2].imshow(overlay)
        axes[row_index, 2].set_title("Overlay")
        for column in range(3):
            axes[row_index, column].axis("off")

    fig.suptitle("Representative samples near each new dataset median foreground fraction", fontsize=13)
    fig.savefig(OUTPUT, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
