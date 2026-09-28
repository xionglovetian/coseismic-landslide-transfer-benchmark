"""Evaluate epoch checkpoints on CAS validation and three target regions."""

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

REGIONS = ["hokkaido_iburi_tobu", "lombok", "palu"]


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
    parser.add_argument("--model", default="ResUNet")
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--epochs", type=int, nargs="+", default=[10, 20, 30, 40, 50])
    parser.add_argument("--train-root", default=str(PROJECT / "repos" / "AS-UNet" / "inputs" / "data_sum_moxizhen+bijie"))
    parser.add_argument("--manifest", default=str(PROJECT / "data" / "processed" / "benchmark_v2_regions_512" / "manifest_benchmark_v2.csv"))
    parser.add_argument("--regions", nargs="+", default=REGIONS)
    parser.add_argument("--input-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--output-dir", default=str(PROJECT / "reports" / "p4_source_overfit_raw"))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_dataset = build_dataset(args.train_root, "val", args.input_size, train=False)
    source_loader = DataLoader(source_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    rows = [row for row in csv.DictReader(Path(args.manifest).open(encoding="utf-8-sig")) if row["region"] in args.regions]
    target_loaders = {}
    for region in args.regions:
        region_rows = [row for row in rows if row["region"] == region]
        dataset = ManifestDataset(region_rows, args.input_size)
        target_loaders[region] = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for epoch in args.epochs:
        checkpoint_path = Path(args.checkpoint_dir) / f"epoch_{epoch:03d}.pth"
        output_path = output_dir / f"epoch_{epoch:03d}.json"
        if output_path.exists() and not args.force:
            print(f"SKIP epoch {epoch}", flush=True)
            continue
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model = build_model(args.model).to(device)
        model.load_state_dict(checkpoint.get("state_dict", checkpoint))
        start = time.time()
        cas_val = evaluate_source(model, source_loader, device, args.amp)
        groups = {}
        for region, loader in target_loaders.items():
            groups.update(evaluate_checkpoint(model, loader, [region], device, args.amp))
        result = {
            "model": args.model,
            "epoch": epoch,
            "checkpoint": str(checkpoint_path),
            "input_size": args.input_size,
            "cas_val": cas_val,
            "regions": groups,
            "elapsed_seconds": time.time() - start,
        }
        output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"DONE epoch {epoch} elapsed_min={(time.time() - start) / 60:.2f}", flush=True)
        del model, checkpoint
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
