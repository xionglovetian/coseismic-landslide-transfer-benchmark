"""Evaluate confidence calibration and high-confidence error rates."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import os as _repo_os
PROJECT = Path(_repo_os.environ.get("LANDSLIDE_PROJECT_ROOT", Path(__file__).resolve().parents[1])).resolve()
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "scripts"))

from evaluate_external_benchmark import ManifestDataset
from src.datasets import build_dataset
from src.models import build_model

REGIONS = ["hokkaido_iburi_tobu", "lombok", "palu"]
BIN_EDGES = [0.5, 0.7, 0.9, 0.99, 1.000001]


def empty_stats():
    bins = []
    for lower, upper in zip(BIN_EDGES[:-1], BIN_EDGES[1:]):
        bins.append({"lower": lower, "upper": min(upper, 1.0), "count": 0, "error": 0, "confidence_sum": 0.0, "positive_sum": 0})
    return {"count": 0, "error": 0, "confidence_sum": 0.0, "positive_sum": 0, "bins": bins}


def update_stats(stats, probabilities, targets):
    confidence = np.maximum(probabilities, 1.0 - probabilities)
    predicted = probabilities >= 0.5
    errors = predicted != targets
    stats["count"] += int(confidence.size)
    stats["error"] += int(errors.sum())
    stats["confidence_sum"] += float(confidence.sum())
    stats["positive_sum"] += int(predicted.sum())
    for item in stats["bins"]:
        mask = (confidence >= item["lower"]) & (confidence < item["upper"])
        item["count"] += int(mask.sum())
        item["error"] += int(errors[mask].sum())
        item["confidence_sum"] += float(confidence[mask].sum())
        item["positive_sum"] += int(predicted[mask].sum())


def finalize(stats):
    result = {
        "pixels": stats["count"],
        "error_rate": stats["error"] / max(stats["count"], 1),
        "mean_confidence": stats["confidence_sum"] / max(stats["count"], 1),
        "positive_rate": stats["positive_sum"] / max(stats["count"], 1),
        "high_confidence_pixels": 0,
        "high_confidence_fraction": 0.0,
        "high_confidence_error_rate": 0.0,
        "bins": [],
    }
    for item in stats["bins"]:
        record = {
            "lower": item["lower"],
            "upper": item["upper"],
            "pixels": item["count"],
            "pixel_fraction": item["count"] / max(stats["count"], 1),
            "error_rate": item["error"] / max(item["count"], 1),
            "mean_confidence": item["confidence_sum"] / max(item["count"], 1),
        }
        result["bins"].append(record)
        if item["lower"] >= 0.9:
            result["high_confidence_pixels"] += item["count"]
            result["high_confidence_error_rate"] += item["error"]
    result["high_confidence_fraction"] = result["high_confidence_pixels"] / max(stats["count"], 1)
    result["high_confidence_error_rate"] /= max(result["high_confidence_pixels"], 1)
    return result


def run_loader(model, loader, device, amp):
    stats = empty_stats()
    model.eval()
    with torch.inference_mode():
        for image, mask, _ in loader:
            image = image.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=amp and device.type == "cuda"):
                probabilities = torch.sigmoid(model(image).float()).cpu().numpy()[:, 0]
            targets = (mask.numpy()[:, 0] >= 0.5)
            update_stats(stats, probabilities, targets)
    return finalize(stats)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model", default="ResUNet")
    parser.add_argument("--train-root", default=str(PROJECT / "repos" / "AS-UNet" / "inputs" / "data_sum_moxizhen+bijie"))
    parser.add_argument("--manifest", default=str(PROJECT / "data" / "processed" / "benchmark_v2_regions_512" / "manifest_benchmark_v2.csv"))
    parser.add_argument("--input-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--output", default=str(PROJECT / "reports" / "p4_confidence_calibration.json"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = build_model(args.model).to(device)
    model.load_state_dict(checkpoint.get("state_dict", checkpoint))

    source_dataset = build_dataset(args.train_root, "val", args.input_size, train=False)
    source_loader = DataLoader(source_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    results = {"source_val": run_loader(model, source_loader, device, args.amp)}

    manifest_rows = list(csv.DictReader(Path(args.manifest).open(encoding="utf-8-sig")))
    for region in REGIONS:
        rows = [row for row in manifest_rows if row["region"] == region]
        dataset = ManifestDataset(rows, args.input_size)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
        results[region] = run_loader(model, loader, device, args.amp)

    Path(args.output).write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: {k: v for k, v in value.items() if k != "bins"} for key, value in results.items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
