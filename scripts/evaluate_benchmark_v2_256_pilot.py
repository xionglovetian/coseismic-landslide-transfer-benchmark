"""Evaluate the four 256x256 seed-42 pilot checkpoints on benchmark v2."""

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
from src.models import build_model

RUN_SPECS = [
    ("ResUNet", "ResUNet", "resunet"),
    ("DeepLabV3+", "DeepLabV3Plus", "deeplabv3plus"),
    ("SegFormer-B0", "SegFormerB0", "segformerb0"),
    ("Bottleneck-LiteASK", "BottleneckLiteASKUNetPlusPlus", "bottleneckliteask"),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        default=str(PROJECT / "data" / "processed" / "benchmark_v2_regions_512" / "manifest_benchmark_v2.csv"),
    )
    parser.add_argument("--input-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--raw-dir",
        default=str(PROJECT / "reports" / "benchmark_v2_256_pilot_raw"),
    )
    args = parser.parse_args()

    rows = [
        row
        for row in csv.DictReader(Path(args.manifest).open(encoding="utf-8-sig"))
        if int(row.get("eligible_external", 1)) == 1
    ]
    regions = sorted({row["region"] for row in rows})
    dataset = ManifestDataset(rows, args.input_size)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw_dir = Path(args.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    for model_label, model_name, slug in RUN_SPECS:
        run_dir = PROJECT / "outputs" / f"bench_v2_256_pilot_{slug}_seed{args.seed}"
        checkpoint_path = run_dir / "best_model.pth"
        output_path = raw_dir / f"bench_v2_256_pilot_{slug}_seed{args.seed}.json"
        if output_path.exists() and not args.force:
            print(f"SKIP {model_label} seed={args.seed}", flush=True)
            continue
        if not checkpoint_path.exists():
            raise FileNotFoundError(checkpoint_path)

        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model = build_model(model_name).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        print(f"START {model_label} seed={args.seed} input={args.input_size}", flush=True)
        start = time.time()
        groups = evaluate_checkpoint(model, loader, regions, device, args.amp)
        record = {
            "model": model_label,
            "seed": args.seed,
            "run_dir": str(run_dir),
            "input_size": args.input_size,
            "regions": regions,
            "num_samples": len(rows),
            "elapsed_seconds": time.time() - start,
            "groups": groups,
        }
        output_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        print(
            f"DONE {model_label} seed={args.seed} elapsed_min={(time.time() - start) / 60:.2f}",
            flush=True,
        )
        del model, checkpoint
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
