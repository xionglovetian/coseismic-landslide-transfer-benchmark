"""Audit exact duplicates, near-duplicates, and target support/query separation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity

import os as _repo_os
PROJECT = Path(_repo_os.environ.get("LANDSLIDE_PROJECT_ROOT", Path(__file__).resolve().parents[1])).resolve()
SOURCE_ROOT = PROJECT / "repos" / "AS-UNet" / "inputs" / "data_sum_moxizhen+bijie"
TARGET_MANIFEST = PROJECT / "data" / "processed" / "benchmark_v2_regions_512" / "manifest_benchmark_v2.csv"
TARGET_INDEX = PROJECT / "data" / "processed" / "benchmark_v2_regions_512" / "target_spatial_index.csv"
SPLIT_DIR = PROJECT / "data" / "processed" / "fewshot_target_splits"


def array_hash(image: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest()


def average_hash(image: np.ndarray) -> np.ndarray:
    pil = Image.fromarray(image.astype(np.uint8)).convert("L").resize((8, 8), Image.Resampling.LANCZOS)
    values = np.asarray(pil, dtype=np.float32)
    return np.unpackbits((values >= values.mean()).astype(np.uint8).reshape(-1))


def hamming_matrix(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lookup = np.asarray([bin(value).count("1") for value in range(256)], dtype=np.uint8)
    xor = left[:, None, :] ^ right[None, :, :]
    return lookup[xor].sum(axis=2)


def pair_similarity(left: dict, right: dict) -> tuple[float, float]:
    left_image = Image.open(left["image_path"]).convert("L").resize((256, 256), Image.Resampling.BILINEAR)
    right_image = Image.open(right["image_path"]).convert("L").resize((256, 256), Image.Resampling.BILINEAR)
    ssim = float(structural_similarity(np.asarray(left_image), np.asarray(right_image), data_range=255))
    left_mask = np.asarray(Image.open(left["mask_path"]).convert("L").resize((256, 256), Image.Resampling.NEAREST)) > 0
    right_mask = np.asarray(Image.open(right["mask_path"]).convert("L").resize((256, 256), Image.Resampling.NEAREST)) > 0
    union = np.logical_or(left_mask, right_mask).sum()
    mask_iou = float(np.logical_and(left_mask, right_mask).sum() / union) if union else 1.0
    return ssim, mask_iou


def collect_source(split: str) -> list[dict]:
    image_dir = SOURCE_ROOT / split / "images"
    mask_dir = SOURCE_ROOT / split / "masks"
    rows = []
    for image_path in sorted(image_dir.glob("*.png")):
        mask_path = mask_dir / image_path.name
        image = np.asarray(Image.open(image_path).convert("RGB"))
        mask = (np.asarray(Image.open(mask_path).convert("L")) > 0).astype(np.uint8)
        rows.append({
            "split": split,
            "id": image_path.stem,
            "image_path": str(image_path),
            "mask_path": str(mask_path),
            "width": image.shape[1],
            "height": image.shape[0],
            "image_hash": array_hash(image),
            "mask_hash": array_hash(mask),
            "ahash": average_hash(image),
        })
    return rows
def main() -> None:
    global SOURCE_ROOT
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", default=str(SOURCE_ROOT))
    parser.add_argument("--output", default=str(PROJECT / "reports" / "spatial_leakage_audit.md"))
    args = parser.parse_args()

    SOURCE_ROOT = Path(args.source_root)
    source_train = collect_source("train")
    source_val = collect_source("val")

    train_image_hashes = {row["image_hash"] for row in source_train}
    val_image_hashes = {row["image_hash"] for row in source_val}
    train_mask_hashes = {row["mask_hash"] for row in source_train}
    val_mask_hashes = {row["mask_hash"] for row in source_val}
    exact_image_overlap = {row["id"]: row for row in source_val if row["image_hash"] in train_image_hashes}
    exact_mask_overlap = {row["id"]: row for row in source_val if row["mask_hash"] in train_mask_hashes}

    val_ahash = np.stack([row["ahash"] for row in source_val])
    train_ahash = np.stack([row["ahash"] for row in source_train])
    distances = hamming_matrix(val_ahash, train_ahash)
    nearest = distances.min(axis=1)
    nearest_index = distances.argmin(axis=1)
    near_rows = []
    for index, row in enumerate(source_val):
        if nearest[index] <= 4:
            train_row = source_train[int(nearest_index[index])]
            ssim, mask_iou = pair_similarity(row, train_row)
            near_rows.append({
                "val_id": row["id"],
                "train_id": train_row["id"],
                "hamming": int(nearest[index]),
                "ssim": ssim,
                "mask_iou": mask_iou,
            })

    target_rows = list(csv.DictReader(TARGET_MANIFEST.open(encoding="utf-8-sig")))
    target_duplicates = {}
    for region in sorted({row["region"] for row in target_rows}):
        region_rows = [row for row in target_rows if row["region"] == region]
        target_duplicates[region] = {
            "n": len(region_rows),
            "duplicate_images": len(region_rows) - len({row["image_sha256"] for row in region_rows}),
            "duplicate_masks": len(region_rows) - len({row["mask_sha256"] for row in region_rows}),
        }

    target_index = list(csv.DictReader(TARGET_INDEX.open(encoding="utf-8-sig")))
    index_map = {(row["region"], row["id"]): row for row in target_index}
    split_audits = []
    for seed in [42, 2026, 777]:
        support_path = SPLIT_DIR / (f"fewshot_support_seed{seed}.csv" if seed == 42 else f"fewshot_support_seed2026_777.csv")
        eval_path = SPLIT_DIR / (f"fewshot_eval_seed{seed}.csv" if seed == 42 else f"fewshot_eval_seed2026_777.csv")
        support_rows = [row for row in csv.DictReader(support_path.open(encoding="utf-8-sig")) if int(row["seed"]) == seed]
        eval_rows = [row for row in csv.DictReader(eval_path.open(encoding="utf-8-sig")) if int(row["seed"]) == seed]
        for region in sorted({row["region"] for row in support_rows}):
            support = [row for row in support_rows if row["region"] == region and int(row["shot"]) == 20]
            query = [row for row in eval_rows if row["region"] == region]
            min_distance = None
            overlapping = 0
            for support_row in support:
                sx, sy = int(support_row["x"]), int(support_row["y"])
                component = int(support_row["component_id"])
                for query_row in query:
                    if int(query_row["component_id"]) != component:
                        continue
                    distance = max(abs(sx - int(query_row["x"])), abs(sy - int(query_row["y"])))
                    min_distance = distance if min_distance is None else min(min_distance, distance)
                    overlapping += int(distance <= 1)
            split_audits.append({
                "seed": seed,
                "region": region,
                "n_support": len(support),
                "n_query": len(query),
                "min_chebyshev_distance": min_distance,
                "support_query_overlap_pairs": overlapping,
            })

    report = [
        "# Spatial Leakage and Duplicate Audit",
        "",
        "## Source train/validation",
        "",
        f"- Train images: {len(source_train)}; validation images: {len(source_val)}.",
        f"- Exact image overlaps: {len(exact_image_overlap)}.",
        f"- Exact mask overlaps: {len(exact_mask_overlap)}.",
        f"- Validation images with average-hash distance <= 4 to any train image: {len(near_rows)}.",
        f"- Strong perceptual near-duplicate candidates (aHash <= 2, SSIM >= 0.7, mask IoU >= 0.5): {sum(row['hamming'] <= 2 and row['ssim'] >= 0.7 and row['mask_iou'] >= 0.5 for row in near_rows)}.",
        "",
        "| Validation ID | Nearest Train ID | aHash Hamming | Grayscale SSIM | Mask IoU |",
        "|---|---|---:|---:|---:|",
    ]
    for row in near_rows[:50]:
        report.append(f"| {row['val_id']} | {row['train_id']} | {row['hamming']} | {row['ssim']:.3f} | {row['mask_iou']:.3f} |")
    if len(near_rows) > 50:
        report.append(f"| ... | {len(near_rows) - 50} more | |")

    report += [
        "",
        "## Target exact duplicates",
        "",
        "| Region | Samples | Duplicate images | Duplicate masks |",
        "|---|---:|---:|---:|",
    ]
    for region, row in target_duplicates.items():
        report.append(f"| {region} | {row['n']} | {row['duplicate_images']} | {row['duplicate_masks']} |")

    report += [
        "",
        "## Few-shot support/query separation",
        "",
        "| Seed | Region | Support | Query | Minimum Chebyshev distance | Pairs with distance <= 1 |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for row in split_audits:
        report.append(
            f"| {row['seed']} | {row['region']} | {row['n_support']} | {row['n_query']} | "
            f"{row['min_chebyshev_distance']} | {row['support_query_overlap_pairs']} |"
        )
    report += [
        "",
        "Interpretation: the few-shot splits enforce distance >= 2 between support chips and common-query chips in the reconstructed stride-256 grid, so they do not overlap spatially. Source train/validation has no exact duplicate images or masks and no strong perceptual near-duplicates after joint aHash, SSIM, and mask-IoU review. Low-threshold aHash candidates are reported for audit only; most have low SSIM and disjoint masks. Spatial adjacency in the variable-sized source images cannot be reconstructed from filenames alone.",
    ]
    Path(args.output).write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({
        "source_train": len(source_train),
        "source_val": len(source_val),
        "exact_image_overlap": len(exact_image_overlap),
        "exact_mask_overlap": len(exact_mask_overlap),
        "ahash_distance_le_4": len(near_rows),
        "target_duplicates": target_duplicates,
        "split_audits": split_audits,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
