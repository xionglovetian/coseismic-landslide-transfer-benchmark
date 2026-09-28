"""Evaluate a checkpoint by stitching overlapping 512-pixel chips."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import os as _repo_os
PROJECT = Path(_repo_os.environ.get("LANDSLIDE_PROJECT_ROOT", Path(__file__).resolve().parents[1])).resolve()
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "scripts"))

from evaluate_external_benchmark import ManifestDataset
from src.models import build_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model", default="ResUNet")
    parser.add_argument("--region", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-csv", required=True)
    parser.add_argument("--checkpoint-label", required=True)
    parser.add_argument("--input-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rows = [row for row in csv.DictReader(Path(args.eval_csv).open(encoding="utf-8-sig")) if row["region"] == args.region and int(row["seed"]) == args.seed]
    if not rows:
        raise RuntimeError("No query rows")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = build_model(args.model).to(device)
    model.load_state_dict(checkpoint.get("state_dict", checkpoint))
    model.eval()
    grouped = defaultdict(list)
    for row in rows:
        grouped[int(row["component_id"])].append(row)

    total = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    component_results = []
    with torch.inference_mode():
        for component_id, component_rows in sorted(grouped.items()):
            min_x = min(int(row["x"]) for row in component_rows)
            min_y = min(int(row["y"]) for row in component_rows)
            max_x = max(int(row["x"]) for row in component_rows)
            max_y = max(int(row["y"]) for row in component_rows)
            width = (max_x - min_x) * 256 + 512
            height = (max_y - min_y) * 256 + 512
            probability_sum = np.zeros((height, width), dtype=np.float32)
            target_mask = np.zeros((height, width), dtype=np.uint8)
            coverage = np.zeros((height, width), dtype=np.uint16)
            component_dataset = ManifestDataset(component_rows, args.input_size)
            component_loader = DataLoader(component_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
            component_batches = [component_rows[i:i + args.batch_size] for i in range(0, len(component_rows), args.batch_size)]
            for batch_rows, (image, mask, _) in zip(component_batches, component_loader):
                image = image.to(device, non_blocking=True)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=args.amp and device.type == "cuda"):
                    probabilities = torch.sigmoid(model(image).float()).cpu()
                up = F.interpolate(probabilities, size=(512, 512), mode="bilinear", align_corners=False)[:, 0].numpy()
                target_up = F.interpolate(mask.float(), size=(512, 512), mode="nearest")[:, 0].numpy() >= 0.5
                for index, row in enumerate(batch_rows):
                    x = (int(row["x"]) - min_x) * 256
                    y = (int(row["y"]) - min_y) * 256
                    write_mask = coverage[y:y + 512, x:x + 512] == 0
                    target_mask[y:y + 512, x:x + 512][write_mask] = target_up[index][write_mask].astype(np.uint8)
                    probability_sum[y:y + 512, x:x + 512] += up[index].astype(np.float32)
                    coverage[y:y + 512, x:x + 512] += 1
            valid = coverage > 0
            prediction = (probability_sum / np.maximum(coverage, 1)) >= 0.5
            target = target_mask.astype(bool)
            tp = int(np.logical_and(prediction & valid, target & valid).sum())
            fp = int(np.logical_and(prediction & valid, ~target & valid).sum())
            fn = int(np.logical_and(~prediction & valid, target & valid).sum())
            tn = int(np.logical_and(~prediction & valid, ~target & valid).sum())
            for key, value in {"tp": tp, "fp": fp, "fn": fn, "tn": tn}.items():
                total[key] += value
            component_results.append({"component_id": component_id, "n_chips": len(component_rows), "width": width, "height": height, "valid_pixels": int(valid.sum()), "tp": tp, "fp": fp, "fn": fn, "tn": tn})
            print("component", component_id, "done", flush=True)
    eps = 1e-12
    tp, fp, fn, tn = total["tp"], total["fp"], total["fn"], total["tn"]
    result = {
        "checkpoint_label": args.checkpoint_label,
        "region": args.region,
        "seed": args.seed,
        "n_chips": len(rows),
        "metrics": {
            "iou": tp / (tp + fp + fn + eps),
            "dice": 2 * tp / (2 * tp + fp + fn + eps),
            "precision": tp / (tp + fp + eps),
            "recall": tp / (tp + fn + eps),
            "accuracy": (tp + tn) / (tp + tn + fp + fn + eps),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        },
        "components": component_results,
    }
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"checkpoint_label": result["checkpoint_label"], "region": result["region"], "seed": result["seed"], "n_chips": result["n_chips"], **result["metrics"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
