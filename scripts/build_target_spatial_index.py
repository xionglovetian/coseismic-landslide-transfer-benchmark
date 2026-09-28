"""Build spatial coordinates and connected components for CAS target-region chips.

Adjacent chips are identified from exact overlap of image+mask half tiles, which
is more reliable than filename order for the official stride-256 crop scheme.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict, deque
from pathlib import Path

import numpy as np
from PIL import Image

import os as _repo_os
PROJECT = Path(_repo_os.environ.get("LANDSLIDE_PROJECT_ROOT", Path(__file__).resolve().parents[1])).resolve()
DEFAULT_MANIFEST = PROJECT / "data" / "processed" / "benchmark_v2_regions_512" / "manifest_benchmark_v2.csv"
DEFAULT_OUTPUT = PROJECT / "data" / "processed" / "benchmark_v2_regions_512" / "target_spatial_index.csv"
TARGET_REGIONS = ["hokkaido_iburi_tobu", "lombok", "palu"]


def half_hashes(image: np.ndarray, mask: np.ndarray) -> dict[str, str]:
    parts = {
        "L": (image[:, :256], mask[:, :256]),
        "R": (image[:, 256:], mask[:, 256:]),
        "T": (image[:256, :], mask[:256, :]),
        "B": (image[256:, :], mask[256:, :]),
    }
    return {
        key: hashlib.sha256(image_part.tobytes() + mask_part.tobytes()).hexdigest()
        for key, (image_part, mask_part) in parts.items()
    }


def build_region(rows: list[dict]) -> list[dict]:
    half_index = {key: defaultdict(list) for key in "LRTB"}
    for index, row in enumerate(rows):
        image = np.asarray(Image.open(row["image_path"]).convert("RGB"))
        mask = np.asarray(Image.open(row["mask_path"]).convert("L"))
        if image.shape[:2] != (512, 512):
            raise ValueError(f"Expected 512x512 chip: {row['image_path']} {image.shape}")
        for key, digest in half_hashes(image, mask).items():
            half_index[key][digest].append(index)

    adjacency = defaultdict(list)
    for digest, rights in half_index["R"].items():
        lefts = half_index["L"].get(digest, [])
        if len(rights) == 1 and len(lefts) == 1 and rights[0] != lefts[0]:
            left, right = rights[0], lefts[0]
            adjacency[left].append((right, 1, 0))
            adjacency[right].append((left, -1, 0))
    for digest, bottoms in half_index["B"].items():
        tops = half_index["T"].get(digest, [])
        if len(bottoms) == 1 and len(tops) == 1 and bottoms[0] != tops[0]:
            top, bottom = bottoms[0], tops[0]
            adjacency[top].append((bottom, 0, 1))
            adjacency[bottom].append((top, 0, -1))

    coords: dict[int, tuple[int, int]] = {}
    component_ids: dict[int, int] = {}
    conflicts = 0
    component_sizes = {}
    for start in range(len(rows)):
        if start in coords:
            continue
        component_id = len(component_sizes)
        coords[start] = (0, 0)
        component_ids[start] = component_id
        queue = deque([start])
        members = []
        while queue:
            current = queue.popleft()
            members.append(current)
            for neighbor, dx, dy in adjacency[current]:
                candidate = (coords[current][0] + dx, coords[current][1] + dy)
                if neighbor not in coords:
                    coords[neighbor] = candidate
                    component_ids[neighbor] = component_id
                    queue.append(neighbor)
                elif coords[neighbor] != candidate:
                    conflicts += 1
        component_sizes[component_id] = len(members)
    if conflicts:
        raise RuntimeError(f"Detected {conflicts} inconsistent spatial-coordinate assignments")

    output = []
    for index, row in enumerate(rows):
        x, y = coords[index]
        component_id = component_ids[index]
        output.append(
            {
                "region": row["region"],
                "id": row["id"],
                "group_id": row["group_id"],
                "component_id": component_id,
                "component_size": component_sizes[component_id],
                "x": x,
                "y": y,
                "image_path": row["image_path"],
                "mask_path": row["mask_path"],
                "foreground_fraction": row.get("foreground_fraction", ""),
            }
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--regions", nargs="+", default=TARGET_REGIONS)
    args = parser.parse_args()

    rows = list(csv.DictReader(Path(args.manifest).open(encoding="utf-8-sig")))
    output_rows = []
    summary = {}
    for region in args.regions:
        region_rows = [row for row in rows if row["region"] == region and int(row.get("eligible_external", 1)) == 1]
        region_index = build_region(region_rows)
        output_rows.extend(region_index)
        component_sizes = Counter(row["component_size"] for row in region_index)
        summary[region] = {
            "n_samples": len(region_index),
            "n_components": len(component_sizes),
            "component_size_counts": {str(key): value for key, value in sorted(component_sizes.items())},
            "max_component": max(component_sizes),
        }
        print(region, summary[region], flush=True)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0].keys()))
        writer.writeheader()
        writer.writerows(output_rows)
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {output_path}")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
