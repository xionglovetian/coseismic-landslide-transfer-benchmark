"""Eval-only tile metrics and calibration for frozen Route 1 checkpoints.

This script performs inference only. It never optimizes model weights. Outputs
per-tile confusion counts, boundary metrics, ECE/Brier inputs, and risk-coverage
bins for E1 final checkpoints and E4 support/adaptation checkpoints.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
from PIL import Image
from scipy.ndimage import distance_transform_edt
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import os as _repo_os
PROJECT = Path(_repo_os.environ.get("LANDSLIDE_PROJECT_ROOT", Path(__file__).resolve().parents[1])).resolve()
sys.path.insert(0, str(PROJECT))
from src.models import build_model

MANIFEST = PROJECT / "data/processed/benchmark_v2_regions_512/manifest_benchmark_v2.csv"
SPLITS = PROJECT / "data/processed/fewshot_target_splits"
BUFFERED = SPLITS / "buffered"
RAW_E1 = PROJECT / "reports/e1_raw"
RAW_E4 = PROJECT / "reports/e4_query_raw"
OUT = PROJECT / "reports/route1_evalonly"
OUT.mkdir(parents=True, exist_ok=True)

E1_REGIMES = ["cas-single", "cas-multistream", "pooled"]
E4_MODES = ["full", "decoder-only"]
E4_BUFFERS = [0, 256, 512]
SEEDS = [42, 2026, 777]
REGIONS = ["hokkaido_iburi_tobu", "lombok", "palu"]
THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7]
CONFIDENCE_EDGES = np.linspace(0.5, 1.0, 21, dtype=np.float64)


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


class IndexedManifestDataset(Dataset):
    def __init__(self, rows, input_size):
        self.rows = rows
        self.transform = A.Compose([A.Resize(input_size, input_size)])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        image = np.array(Image.open(row["image_path"]).convert("RGB"))
        mask = (np.array(Image.open(row["mask_path"]).convert("L")) > 0).astype(np.float32)
        transformed = self.transform(image=image, mask=mask)
        image = transformed["image"].astype(np.float32) / 255.0
        mask = transformed["mask"].astype(np.float32)
        return torch.from_numpy(image.transpose(2, 0, 1)), torch.from_numpy(mask[None]), index


def extract_boundary(mask):
    mask_u8 = mask.astype(np.uint8)
    eroded = cv2.erode(mask_u8, np.ones((3, 3), np.uint8), iterations=1)
    return mask_u8.astype(bool) & ~eroded.astype(bool)


def boundary_metrics(pred, target):
    target_boundary = extract_boundary(target)
    if not target_boundary.any():
        return None
    pred_boundary = extract_boundary(pred)
    diagonal = float(np.hypot(*target.shape))
    if not pred_boundary.any():
        return {"hd95": diagonal, "f1_2": 0.0, "f1_4": 0.0}
    distance_to_target = distance_transform_edt(~target_boundary)
    distance_to_pred = distance_transform_edt(~pred_boundary)
    pred_to_target = distance_to_target[pred_boundary]
    target_to_pred = distance_to_pred[target_boundary]
    result = {"hd95": max(float(np.percentile(pred_to_target, 95)), float(np.percentile(target_to_pred, 95)))}
    for tolerance in (2, 4):
        precision = float(np.mean(pred_to_target <= tolerance))
        recall = float(np.mean(target_to_pred <= tolerance))
        result[f"f1_{tolerance}"] = 2 * precision * recall / (precision + recall + 1e-12)
    return result


def write_csv(path, rows):
    if not rows:
        raise RuntimeError(path)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with Path(path).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def calibrate_batch(probabilities, target):
    confidence = np.maximum(probabilities, 1.0 - probabilities)
    predicted = probabilities >= 0.5
    errors = predicted != target
    total = int(confidence.size)
    brier = float(np.sum((probabilities - target) ** 2))
    bins = []
    for lower, upper in zip(CONFIDENCE_EDGES[:-1], CONFIDENCE_EDGES[1:]):
        if upper >= 1.0:
            mask = (confidence >= lower) & (confidence <= 1.0)
        else:
            mask = (confidence >= lower) & (confidence < upper)
        count = int(mask.sum())
        error_count = int(errors[mask].sum())
        confidence_sum = float(confidence[mask].sum())
        bins.append((float(lower), float(upper), count, error_count, confidence_sum))
    return total, brier, bins


def evaluate_task(task, args):
    model = build_model(task["model_name"]).to(args.device)
    checkpoint = torch.load(task["checkpoint"], map_location=args.device, weights_only=False)
    model.load_state_dict(checkpoint.get("state_dict", checkpoint))
    model.eval()
    outputs = []
    for query in task["queries"]:
        query_hash = sha256_text(",".join(sorted(row["id"] for row in query["rows"])))
        stem = f"{task['task_key']}__b{query['buffer_m']}__q{query_hash}"
        paths = {
            "tiles": OUT / f"{stem}__tiles.csv",
            "summary": OUT / f"{stem}__calibration.json",
            "bins": OUT / f"{stem}__bins.csv",
            "risk": OUT / f"{stem}__risk_coverage.csv",
        }
        if all(path.exists() for path in paths.values()) and not args.overwrite:
            print("SKIP", stem, flush=True)
            continue
        dataset = IndexedManifestDataset(query["rows"], args.input_size)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
        tile_rows = []
        bin_totals = defaultdict(lambda: [0, 0, 0.0])
        total_pixels = 0
        brier_sum = 0.0
        start = time.time()
        with torch.inference_mode():
            for image, mask, indices in tqdm(loader, desc=stem, leave=False):
                image = image.to(args.device, non_blocking=True)
                mask = mask.to(args.device, non_blocking=True)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=args.amp and args.device.type == "cuda"):
                    probabilities = torch.sigmoid(model(image).float())
                probabilities_np = probabilities[:, 0].cpu().numpy()
                target_np = (mask[:, 0] >= 0.5).cpu().numpy()
                total_pixels += int(target_np.size)
                brier_sum += float(np.sum((probabilities_np - target_np) ** 2))
                _, _, bins = calibrate_batch(probabilities_np, target_np)
                for lower, upper, count, error_count, confidence_sum in bins:
                    bucket = bin_totals[(lower, upper)]
                    bucket[0] += count
                    bucket[1] += error_count
                    bucket[2] += confidence_sum
                batch_start = len(tile_rows)
                for threshold in THRESHOLDS:
                    pred = probabilities_np >= threshold
                    tp = np.sum(pred & target_np, axis=(1, 2))
                    fp = np.sum(pred & ~target_np, axis=(1, 2))
                    fn = np.sum(~pred & target_np, axis=(1, 2))
                    tn = np.sum(~pred & ~target_np, axis=(1, 2))
                    for local_idx, row_index in enumerate(indices.tolist()):
                        source_row = query["rows"][row_index]
                        tile_rows.append({
                            **task["metadata"],
                            "buffer_m": query["buffer_m"],
                            "region": source_row["region"],
                            "id": source_row["id"],
                            "threshold": threshold,
                            "tp": int(tp[local_idx]),
                            "fp": int(fp[local_idx]),
                            "fn": int(fn[local_idx]),
                            "tn": int(tn[local_idx]),
                        })
                preds_05 = probabilities_np >= 0.5
                for local_idx, row_index in enumerate(indices.tolist()):
                    metrics = boundary_metrics(preds_05[local_idx], target_np[local_idx])
                    if metrics is None:
                        continue
                    threshold_index = THRESHOLDS.index(0.5)
                    tile_rows[batch_start + threshold_index * len(indices) + local_idx].update(metrics)
        bin_rows = []
        ece = 0.0
        high_conf_count = 0
        high_conf_error = 0
        for lower, upper in sorted(bin_totals):
            count, error_count, confidence_sum = bin_totals[(lower, upper)]
            mean_confidence = confidence_sum / count if count else float("nan")
            error_rate = error_count / count if count else float("nan")
            if count:
                ece += count / total_pixels * abs((1.0 - error_rate) - mean_confidence)
            if lower >= 0.9:
                high_conf_count += count
                high_conf_error += error_count
            bin_rows.append({
                **task["metadata"],
                "buffer_m": query["buffer_m"],
                "bin_lower": lower,
                "bin_upper": upper,
                "pixels": count,
                "errors": error_count,
                "mean_confidence": mean_confidence,
                "error_rate": error_rate,
            })
        risk_rows = []
        cumulative_pixels = 0
        cumulative_errors = 0
        for lower, upper in sorted(bin_totals, reverse=True):
            count, error_count, _ = bin_totals[(lower, upper)]
            cumulative_pixels += count
            cumulative_errors += error_count
            risk_rows.append({
                **task["metadata"],
                "buffer_m": query["buffer_m"],
                "confidence_threshold": lower,
                "coverage": cumulative_pixels / max(total_pixels, 1),
                "selective_risk": cumulative_errors / max(cumulative_pixels, 1),
                "selected_pixels": cumulative_pixels,
                "errors": cumulative_errors,
            })
        summary = {
            **task["metadata"],
            "buffer_m": query["buffer_m"],
            "n_eval": len(query["rows"]),
            "pixels": total_pixels,
            "brier": brier_sum / max(total_pixels, 1),
            "ece": ece,
            "high_confidence_pixels": high_conf_count,
            "high_confidence_errors": high_conf_error,
            "high_confidence_error_rate": high_conf_error / max(high_conf_count, 1),
            "elapsed_seconds": time.time() - start,
        }
        write_csv(paths["tiles"], tile_rows)
        write_csv(paths["bins"], bin_rows)
        write_csv(paths["risk"], risk_rows)
        paths["summary"].write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        outputs.append(summary)
        print("DONE", stem, "n_eval", len(query["rows"]), "elapsed_s", round(summary["elapsed_seconds"], 1), flush=True)
    del model
    if args.device.type == "cuda":
        torch.cuda.empty_cache()
    return outputs


def load_manifest_rows():
    rows = read_csv(MANIFEST)
    rows = [row for row in rows if int(row.get("eligible_external", 1)) == 1]
    return [row for row in rows if row["region"] in REGIONS]


def build_e1_tasks():
    rows = load_manifest_rows()
    by_region = defaultdict(list)
    for row in rows:
        by_region[row["region"]].append(row)
    tasks = []
    for regime in E1_REGIMES:
        for seed in SEEDS:
            checkpoint = PROJECT / "outputs" / ("bench_v2_e1_" + regime + "_seed" + str(seed)) / "checkpoints" / "step_1120.pth"
            if not checkpoint.exists():
                raise FileNotFoundError(checkpoint)
            task_key = "e1__" + regime + "__seed" + str(seed)
            tasks.append({
                "task_key": task_key,
                "experiment": "E1",
                "model_name": "BottleneckLiteASKUNetPlusPlus",
                "checkpoint": str(checkpoint),
                "queries": [{"buffer_m": "na", "rows": rows}],
                "metadata": {"experiment": "E1", "run_id": task_key, "regime": regime, "role": "intervention", "mode": "", "seed": seed, "step": 1120},
            })
    return tasks


def parse_e4_label(label):
    if label.startswith("source_"):
        return "source", "source", int(label.split("_")[1].replace("seed", ""))
    parts = label.split("_")
    return "adapted", parts[1], int(parts[-1].replace("seed", ""))


def build_e4_tasks():
    registry = {}
    for buffer_m in E4_BUFFERS:
        folder = RAW_E4 / ("buffer" + str(buffer_m))
        for path in sorted(folder.glob("*.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            role, mode, seed = parse_e4_label(data["checkpoint_label"])
            region = data["region"]
            checkpoint = data["checkpoint"]
            key = (checkpoint, region)
            if key not in registry:
                registry[key] = {
                    "checkpoint": checkpoint,
                    "role": role,
                    "mode": mode,
                    "seed": seed,
                    "region": region,
                    "queries": [],
                }
            if buffer_m == 0:
                eval_csv = SPLITS / ("fewshot_eval_seed42.csv" if seed == 42 else "fewshot_eval_seed2026_777.csv")
                rows = [row for row in read_csv(eval_csv) if int(row["seed"]) == seed and row["region"] == region]
            else:
                eval_csv = BUFFERED / ("eval_seed" + str(seed) + "_" + region + "_buffer" + str(buffer_m) + ".csv")
                rows = read_csv(eval_csv)
            registry[key]["queries"].append({"buffer_m": buffer_m, "rows": rows})
    tasks = []
    for checkpoint_key, spec in registry.items():
        # Deduplicate identical query sets, which occur for 256 and 512 m in the
        # current target splits. Keep the lower buffer label as the canonical row.
        unique = {}
        for query in spec["queries"]:
            query_hash = sha256_text(",".join(sorted(row["id"] for row in query["rows"])))
            if query_hash not in unique:
                unique[query_hash] = query
        task_key = "e4__" + spec["role"] + "__" + spec["mode"] + "__seed" + str(spec["seed"]) + "__" + spec["region"]
        tasks.append({
            "task_key": task_key,
            "experiment": "E4",
            "model_name": "ResUNet",
            "checkpoint": spec["checkpoint"],
            "queries": list(unique.values()),
            "metadata": {
                "experiment": "E4",
                "run_id": task_key,
                "regime": "",
                "role": spec["role"],
                "mode": spec["mode"],
                "seed": spec["seed"],
                "step": 400 if spec["role"] == "adapted" else 0,
                "region": spec["region"],
            },
        })
    return tasks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=["all", "e1", "e4"], default="all")
    parser.add_argument("--input-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tasks = []
    if args.experiment in {"all", "e1"}:
        tasks.extend(build_e1_tasks())
    if args.experiment in {"all", "e4"}:
        tasks.extend(build_e4_tasks())
    print("device", args.device, "tasks", len(tasks), "queries", sum(len(task["queries"]) for task in tasks), flush=True)
    summaries = []
    for task in tasks:
        summaries.extend(evaluate_task(task, args))
    print(json.dumps({"completed_or_cached_queries": len(summaries), "output_dir": str(OUT)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
