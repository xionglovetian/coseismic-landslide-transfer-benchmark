"""Evaluate a few-shot checkpoint on a specified query split."""

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

from evaluate_external_benchmark import ManifestDataset, evaluate_checkpoint
from src.models import build_model

BIN_EDGES = [0.5, 0.7, 0.9, 0.99, 1.000001]


def empty_stats():
    return {"count": 0, "error": 0, "confidence_sum": 0.0, "bins": [{"lower": lower, "upper": min(upper, 1.0), "count": 0, "error": 0, "confidence_sum": 0.0} for lower, upper in zip(BIN_EDGES[:-1], BIN_EDGES[1:])]}


def update_stats(stats, probabilities, targets):
    confidence = np.maximum(probabilities, 1.0 - probabilities)
    predicted = probabilities >= 0.5
    errors = predicted != targets
    stats["count"] += int(confidence.size)
    stats["error"] += int(errors.sum())
    stats["confidence_sum"] += float(confidence.sum())
    for item in stats["bins"]:
        mask = (confidence >= item["lower"]) & (confidence < item["upper"])
        item["count"] += int(mask.sum())
        item["error"] += int(errors[mask].sum())
        item["confidence_sum"] += float(confidence[mask].sum())


def finalize(stats):
    result = {
        "pixels": stats["count"],
        "error_rate": stats["error"] / max(stats["count"], 1),
        "mean_confidence": stats["confidence_sum"] / max(stats["count"], 1),
        "high_confidence_error_rate": 0.0,
        "high_confidence_pixels": 0,
        "bins": [],
    }
    for item in stats["bins"]:
        result["bins"].append({"lower": item["lower"], "upper": item["upper"], "pixels": item["count"], "error_rate": item["error"] / max(item["count"], 1), "mean_confidence": item["confidence_sum"] / max(item["count"], 1)})
        if item["lower"] >= 0.9:
            result["high_confidence_pixels"] += item["count"]
            result["high_confidence_error_rate"] += item["error"]
    result["high_confidence_error_rate"] /= max(result["high_confidence_pixels"], 1)
    return result

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model", default="ResUNet")
    parser.add_argument("--region", required=True)
    parser.add_argument("--eval-csv", required=True)
    parser.add_argument("--eval-seed", type=int, required=True)
    parser.add_argument("--checkpoint-label", required=True)
    parser.add_argument("--input-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rows = [row for row in csv.DictReader(Path(args.eval_csv).open(encoding="utf-8-sig")) if row.get("region") == args.region and int(row.get("seed", args.eval_seed)) == args.eval_seed]
    if not rows:
        raise RuntimeError("No evaluation rows selected")
    dataset = ManifestDataset(rows, args.input_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = build_model(args.model).to(device)
    model.load_state_dict(checkpoint.get("state_dict", checkpoint))

    metrics = evaluate_checkpoint(model, loader, [args.region], device, args.amp)
    stats = empty_stats()
    model.eval()
    with torch.inference_mode():
        for image, mask, _ in loader:
            image = image.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=args.amp and device.type == "cuda"):
                probabilities = torch.sigmoid(model(image).float()).cpu().numpy()[:, 0]
            targets = mask.numpy()[:, 0] >= 0.5
            update_stats(stats, probabilities, targets)
    result = {
        "checkpoint_label": args.checkpoint_label,
        "checkpoint": args.checkpoint,
        "region": args.region,
        "eval_seed": args.eval_seed,
        "n_eval": len(rows),
        "threshold_metrics": metrics[args.region]["threshold_metrics"],
        "boundary": {key: value for key, value in metrics[args.region].items() if key != "threshold_metrics"},
        "calibration": finalize(stats),
    }
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"label": args.checkpoint_label, "region": args.region, "n_eval": len(rows), "iou": metrics[args.region]["threshold_metrics"]["0.5"]["iou"], "high_conf_error": result["calibration"]["high_confidence_error_rate"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
