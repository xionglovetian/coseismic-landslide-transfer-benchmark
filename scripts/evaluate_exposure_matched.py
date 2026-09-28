"""Evaluate exposure-matched step checkpoints on CAS val and three targets."""

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
from src.models import build_model
from train_multisource_dg import evaluate

REGIONS = ["hokkaido_iburi_tobu", "lombok", "palu"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--steps", type=int, nargs="+", default=[0, 280, 560, 840, 1120])
    parser.add_argument("--model", default="BottleneckLiteASKUNetPlusPlus")
    parser.add_argument("--input-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    source_dataset = build_dataset(PROJECT / "repos" / "AS-UNet" / "inputs" / "data_sum_moxizhen+bijie", "val", args.input_size, train=False)
    source_loader = DataLoader(source_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    criterion = torch.nn.BCEWithLogitsLoss()
    manifest_path = PROJECT / "data" / "processed" / "benchmark_v2_regions_512" / "manifest_benchmark_v2.csv"
    manifest_rows = list(csv.DictReader(manifest_path.open(encoding="utf-8-sig")))
    target_loaders = {}
    for region in REGIONS:
        rows = [row for row in manifest_rows if row["region"] == region]
        target_loaders[region] = DataLoader(ManifestDataset(rows, args.input_size), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    for step in args.steps:
        checkpoint_path = run_dir / "checkpoints" / f"step_{step:04d}.pth"
        output_path = output_dir / f"step_{step:04d}.json"
        if not checkpoint_path.exists():
            raise FileNotFoundError(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model = build_model(args.model).to(device)
        model.load_state_dict(checkpoint.get("state_dict", checkpoint))
        start = time.time()
        source_metrics = evaluate(model, source_loader, criterion, device, args.amp)
        targets = {}
        for region, loader in target_loaders.items():
            targets.update(evaluate_checkpoint(model, loader, [region], device, args.amp))
        output = {"regime": run_dir.name, "step": step, "checkpoint": str(checkpoint_path), "source_validation": source_metrics, "target": targets, "elapsed_seconds": time.time() - start}
        output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
        print("DONE", run_dir.name, step, flush=True)
        del model, checkpoint
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
