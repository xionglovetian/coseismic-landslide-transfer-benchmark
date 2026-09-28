"""Evaluate one trained checkpoint on CAS validation and selected target regions."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import os as _repo_os
PROJECT = Path(_repo_os.environ.get("LANDSLIDE_PROJECT_ROOT", Path(__file__).resolve().parents[1])).resolve()
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "scripts"))

from evaluate_external_benchmark import ManifestDataset, evaluate_checkpoint
from src.datasets import build_dataset
from src.metrics import SegmentationMetrics
from src.models import build_model

DEFAULT_REGIONS = [
    "hokkaido_iburi_tobu",
    "lombok",
    "palu",
    "wenchuan",
    "longxi_river",
    "jiuzhai_valley",
]


def evaluate_source(model, loader, device, amp):
    metrics = SegmentationMetrics()
    model.eval()
    with torch.inference_mode():
        for image, mask, _ in loader:
            image = image.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=amp and device.type == "cuda"):
                logits = model(image).float()
            metrics.update(logits, mask)
    return metrics.compute()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--input-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--train-root", default=str(PROJECT / "repos" / "AS-UNet" / "inputs" / "data_sum_moxizhen+bijie"))
    parser.add_argument("--manifest", default=str(PROJECT / "data" / "processed" / "benchmark_v2_regions_512" / "manifest_benchmark_v2.csv"))
    parser.add_argument("--target-regions", nargs="+", default=DEFAULT_REGIONS)
    args = parser.parse_args()

    output_path = Path(args.output)
    if output_path.exists() and not args.force:
        print(f"SKIP {output_path}", flush=True)
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)

    rows = [
        row
        for row in csv.DictReader(Path(args.manifest).open(encoding="utf-8-sig"))
        if row["region"] in args.target_regions and int(row.get("eligible_external", 1)) == 1
    ]
    found = sorted({row["region"] for row in rows})
    missing = sorted(set(args.target_regions) - set(found))
    if missing:
        raise RuntimeError(f"Missing target regions: {missing}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = build_model(args.model).to(device)
    model.load_state_dict(checkpoint.get("state_dict", checkpoint))

    source_dataset = build_dataset(args.train_root, "val", args.input_size, train=False)
    source_loader = DataLoader(
        source_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    target_dataset = ManifestDataset(rows, args.input_size)
    target_loader = DataLoader(
        target_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    start = time.time()
    source_validation = evaluate_source(model, source_loader, device, args.amp)
    target_groups = evaluate_checkpoint(model, target_loader, found, device, args.amp)
    threshold_05 = {
        region: target_groups[region]["threshold_metrics"]["0.5"]
        for region in found
    }
    macro_iou = sum(threshold_05[region]["iou"] for region in found) / len(found)

    result = {
        "model": args.model,
        "checkpoint": str(checkpoint_path),
        "input_size": args.input_size,
        "source_validation": source_validation,
        "target_regions": found,
        "target_macro_iou": macro_iou,
        "target_threshold_05": threshold_05,
        "target_full": target_groups,
        "elapsed_seconds": time.time() - start,
    }
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output_path),
                "source_val_iou": source_validation["iou"],
                "target_macro_iou": macro_iou,
                "elapsed_seconds": result["elapsed_seconds"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
