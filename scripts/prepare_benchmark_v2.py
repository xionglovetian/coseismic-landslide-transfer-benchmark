"""Prepare the seven-region CAS benchmark v2 manifest and PNG data."""

from __future__ import annotations

import csv
import hashlib
import io
import posixpath
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

import os as _repo_os
PROJECT = Path(_repo_os.environ.get("LANDSLIDE_PROJECT_ROOT", Path(__file__).resolve().parents[1])).resolve()
EXISTING_MANIFEST = PROJECT / "data" / "processed" / "external_regions_512" / "manifest_external.csv"
OUTPUT_ROOT = PROJECT / "data" / "processed" / "benchmark_v2_regions_512"
OUTPUT_MANIFEST = OUTPUT_ROOT / "manifest_benchmark_v2.csv"

NEW_ARCHIVES = {
    "hokkaido_iburi_tobu": {
        "archive": Path(r"C:\Users\ASUS\Desktop\数据集\Hokkaido Iburi-Tobu.zip"),
        "region_name": "Hokkaido Iburi-Tobu",
        "acquisition": "2018.09-2018.10",
        "source": "Geospatial Information Authority of Japan",
        "sensor": "Satellite",
        "ground_resolution_m": 3,
        "authorization": "CC BY 4.0",
    },
    "lombok": {
        "archive": Path(r"C:\Users\ASUS\Desktop\数据集\Lombok.zip"),
        "region_name": "Lombok",
        "acquisition": "2019.05-2019.12",
        "source": "Digital Globe Open Data Program",
        "sensor": "WorldView-2/3",
        "ground_resolution_m": 5,
        "authorization": "CC BY-NC 4.0",
    },
    "palu": {
        "archive": Path(r"C:\Users\ASUS\Desktop\数据集\palu.zip"),
        "region_name": "Palu",
        "acquisition": "2021.01-2021.11",
        "source": "Digital Globe Open Data Program",
        "sensor": "WorldView-2/3",
        "ground_resolution_m": 5,
        "authorization": "CC BY-NC 4.0",
    },
}
EXISTING_METADATA = {
    "wenchuan": {"region_name": "Wenchuan", "acquisition": "2008.11-2008.12", "source": "U.S. Geological Survey", "sensor": "Landsat", "ground_resolution_m": 5, "authorization": "LP DAAC terms"},
    "jiuzhai_valley": {"region_name": "Jiuzhai Valley", "acquisition": "2017.08-2017.09", "source": "Sichuan Geomatics Center", "sensor": "UAV", "ground_resolution_m": 0.2, "authorization": "Derivative Works Licence"},
    "moxitaidi": {"region_name": "Moxitaidi", "acquisition": "2022.09-2022.10", "source": "Sichuan Geomatics Center", "sensor": "UAV", "ground_resolution_m": 0.6, "authorization": "Derivative Works Licence"},
    "longxi_river": {"region_name": "Longxi River", "acquisition": "2011.03-2011.05", "source": "Sichuan Geomatics Center", "sensor": "UAV", "ground_resolution_m": 0.5, "authorization": "Derivative Works Licence"},
}

FIELDS = [
    "dataset", "benchmark_version", "region", "region_name", "split", "id",
    "image_path", "mask_path", "source_archive", "source_image_member", "source_mask_member",
    "acquisition", "source", "sensor", "ground_resolution_m", "authorization",
    "group_id", "augmented", "eligible_external", "exclusion_reason",
    "width", "height", "foreground_fraction", "image_sha256", "mask_sha256",
]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_existing_rows():
    with EXISTING_MANIFEST.open(encoding="utf-8-sig", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if int(row.get("eligible_external", 1)) == 1]
    output = []
    for row in rows:
        metadata = EXISTING_METADATA[row["region"]]
        output.append({
            "dataset": "CAS_Landslide_Dataset",
            "benchmark_version": "2",
            "region": row["region"],
            "region_name": metadata["region_name"],
            "split": "external",
            "id": row["id"],
            "image_path": row["image_path"],
            "mask_path": row["mask_path"],
            "source_archive": row.get("source_archive", ""),
            "source_image_member": row.get("source_image_member", ""),
            "source_mask_member": row.get("source_mask_member", ""),
            "acquisition": metadata["acquisition"],
            "source": metadata["source"],
            "sensor": metadata["sensor"],
            "ground_resolution_m": metadata["ground_resolution_m"],
            "authorization": metadata["authorization"],
            "group_id": f"{row['region']}:{row['id']}",
            "augmented": row.get("augmented", "0"),
            "eligible_external": "1",
            "exclusion_reason": "",
            "width": row.get("width", ""),
            "height": row.get("height", ""),
            "foreground_fraction": row.get("foreground_fraction", ""),
            "image_sha256": row.get("image_sha256", ""),
            "mask_sha256": row.get("mask_sha256", ""),
        })
    return output


def prepare_new_region(region, spec):
    region_dir = OUTPUT_ROOT / region
    image_dir = region_dir / "images"
    mask_dir = region_dir / "masks"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    with zipfile.ZipFile(spec["archive"]) as archive:
        names = set(archive.namelist())
        stems = sorted(
            posixpath.splitext(posixpath.basename(name))[0]
            for name in names
            if name.startswith("mask/") and name.lower().endswith(".tif")
        )
        for stem in tqdm(stems, desc=f"prepare {region}"):
            image_member = f"img/{stem}.tif"
            mask_member = f"mask/{stem}.tif"
            if image_member not in names:
                continue
            image = Image.open(io.BytesIO(archive.read(image_member))).convert("RGB")
            mask = Image.open(io.BytesIO(archive.read(mask_member))).convert("L")
            mask_array = np.asarray(mask) > 0
            image_path = image_dir / f"{stem}.png"
            mask_path = mask_dir / f"{stem}.png"
            image.save(image_path)
            Image.fromarray((mask_array.astype(np.uint8) * 255), mode="L").save(mask_path)
            image_bytes = image_path.read_bytes()
            mask_bytes = mask_path.read_bytes()
            rows.append({
                "dataset": "CAS_Landslide_Dataset",
                "benchmark_version": "2",
                "region": region,
                "region_name": spec["region_name"],
                "split": "external",
                "id": stem,
                "image_path": str(image_path),
                "mask_path": str(mask_path),
                "source_archive": spec["archive"].name,
                "source_image_member": image_member,
                "source_mask_member": mask_member,
                "acquisition": spec["acquisition"],
                "source": spec["source"],
                "sensor": spec["sensor"],
                "ground_resolution_m": spec["ground_resolution_m"],
                "authorization": spec["authorization"],
                "group_id": f"{region}:{stem}",
                "augmented": "0",
                "eligible_external": "1",
                "exclusion_reason": "",
                "width": image.width,
                "height": image.height,
                "foreground_fraction": float(mask_array.mean()),
                "image_sha256": sha256_bytes(image_bytes),
                "mask_sha256": sha256_bytes(mask_bytes),
            })
    return rows


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    rows = load_existing_rows()
    for region, spec in NEW_ARCHIVES.items():
        rows.extend(prepare_new_region(region, spec))
    with OUTPUT_MANIFEST.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {OUTPUT_MANIFEST}")
    print(f"rows={len(rows)}")
    counts = {}
    for row in rows:
        counts[row["region"]] = counts.get(row["region"], 0) + 1
    for region in sorted(counts):
        print(region, counts[region])


if __name__ == "__main__":
    main()
