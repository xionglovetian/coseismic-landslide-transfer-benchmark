import argparse
import csv
import hashlib
import io
import re
from collections import Counter
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image
from tqdm import tqdm

import os as _repo_os
PROJECT = Path(_repo_os.environ.get("LANDSLIDE_PROJECT_ROOT", Path(__file__).resolve().parents[1])).resolve()
DEFAULT_SOURCE = Path(r"C:\Users\ASUS\Desktop\数据集")
DEFAULT_OUTPUT = PROJECT / "data" / "processed" / "external_regions_512"
REGION_SPECS = [
    ("longxi_river", "Longxi River", "*Longxi River*.zip"),
    ("wenchuan", "Wenchuan", "*汶川*178*.zip"),
    ("jiuzhai_valley", "Jiuzhai Valley", "*Jiuzhai valley*.zip"),
    ("moxitaidi", "Moxitaidi", "*Moxitaidi*.zip"),
]
EXCLUSIONS = {
    "moxitaidi": {
        "luding_UAV0108": "pHASH distance 8 to CAS training image",
        "luding_UAV0277": "pHASH distance 8 to CAS training image",
        "luding_UAV0353": "pHASH distance 8 to CAS training image",
        "luding_UAV0477": "pHASH distance 8 to CAS training image",
    }
}

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def safe_stem(name: str) -> str:
    stem = Path(name).name
    if stem.lower().endswith((".tif", ".tiff", ".png")):
        stem = stem.rsplit(".", 1)[0]
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", stem)

def png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True, compress_level=6)
    return buffer.getvalue()

def find_asset_map(zip_file: zipfile.ZipFile, kind: str):
    asset_map = {}
    for name in zip_file.namelist():
        normalized = name.replace("\\", "/")
        parts = [part.lower() for part in normalized.split("/")]
        if kind not in parts or not normalized.lower().endswith((".tif", ".tiff")):
            continue
        asset_map[Path(normalized).stem.lower()] = name
    return asset_map

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", default=str(DEFAULT_SOURCE))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    source_root = Path(args.source_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    for slug, _, _ in REGION_SPECS:
        (output_root / slug / "images").mkdir(parents=True, exist_ok=True)
        (output_root / slug / "masks").mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    global_image_hashes = {}
    global_mask_hashes = {}
    for slug, display_name, pattern in REGION_SPECS:
        matches = sorted(source_root.glob(pattern))
        if len(matches) != 1:
            raise RuntimeError(f"Expected one archive for {display_name}, found {len(matches)}: {matches}")
        archive = matches[0]
        exclusion_map = EXCLUSIONS.get(slug, {})
        with zipfile.ZipFile(archive) as zf:
            image_map = find_asset_map(zf, "img")
            mask_map = find_asset_map(zf, "mask")
            stems = sorted(set(image_map) & set(mask_map))
            unmatched_images = set(image_map) - set(mask_map)
            unmatched_masks = set(mask_map) - set(image_map)
            print(
                f"{display_name}: images={len(image_map)} masks={len(mask_map)} pairs={len(stems)} "
                f"unmatched_images={len(unmatched_images)} unmatched_masks={len(unmatched_masks)}",
                flush=True,
            )
            for stem in tqdm(stems, desc=display_name):
                image_member = image_map[stem]
                mask_member = mask_map[stem]
                original_stem = Path(image_member).stem
                output_name = f"{safe_stem(original_stem)}.png"
                image_path = output_root / slug / "images" / output_name
                mask_path = output_root / slug / "masks" / output_name
                excluded = original_stem in exclusion_map
                reason = exclusion_map.get(original_stem, "")
                if args.force or not image_path.exists() or not mask_path.exists():
                    with zf.open(image_member) as handle:
                        image = Image.open(handle).convert("RGB")
                    with zf.open(mask_member) as handle:
                        mask = Image.open(handle).convert("L")
                    if image.size != mask.size:
                        raise RuntimeError(f"Size mismatch: {image.size} vs {mask.size}")
                    mask_array = (np.asarray(mask) > 0).astype(np.uint8) * 255
                    image_data = png_bytes(image)
                    mask_data = png_bytes(Image.fromarray(mask_array, mode="L"))
                    image_path.write_bytes(image_data)
                    mask_path.write_bytes(mask_data)
                else:
                    image_data = image_path.read_bytes()
                    mask_data = mask_path.read_bytes()

                image_hash = sha256_bytes(image_data)
                mask_hash = sha256_bytes(mask_data)
                global_image_hashes.setdefault(image_hash, []).append((slug, output_name))
                global_mask_hashes.setdefault(mask_hash, []).append((slug, output_name))
                mask_array = np.asarray(Image.open(mask_path).convert("L"))
                width, height = Image.open(image_path).size
                manifest_rows.append({
                    "dataset": f"External_{slug}",
                    "split": "external",
                    "id": original_stem,
                    "image_path": str(image_path),
                    "mask_path": str(mask_path),
                    "region": slug,
                    "region_name": display_name,
                    "source_archive": archive.name,
                    "source_image_member": image_member,
                    "source_mask_member": mask_member,
                    "augmented": 0,
                    "eligible_external": int(not excluded),
                    "exclusion_reason": reason,
                    "width": width,
                    "height": height,
                    "foreground_fraction": float((mask_array > 0).mean()),
                    "image_sha256": image_hash,
                    "mask_sha256": mask_hash,
                })
    manifest_path = output_root / "manifest.csv"
    fields = list(manifest_rows[0].keys())
    with manifest_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(manifest_rows)

    eligible_rows = [row for row in manifest_rows if row["eligible_external"] == 1]
    eligible_manifest = output_root / "manifest_external.csv"
    with eligible_manifest.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(eligible_rows)

    summary_rows = []
    for slug, display_name, _ in REGION_SPECS:
        rows = [row for row in manifest_rows if row["region"] == slug]
        eligible = [row for row in rows if row["eligible_external"] == 1]
        duplicate_images = sum(1 for row in rows if len(global_image_hashes[row["image_sha256"]]) > 1)
        summary_rows.append({
            "region": slug,
            "region_name": display_name,
            "pairs": len(rows),
            "eligible_external_pairs": len(eligible),
            "excluded_pairs": len(rows) - len(eligible),
            "mean_foreground_fraction": float(np.mean([row["foreground_fraction"] for row in rows])) if rows else 0.0,
            "duplicate_image_hashes": duplicate_images,
        })
    summary_path = output_root / "dataset_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    print("manifest", manifest_path, "rows", len(manifest_rows), flush=True)
    print("eligible", eligible_manifest, "rows", len(eligible_rows), flush=True)
    print("summary", summary_path, flush=True)

if __name__ == "__main__":
    main()
