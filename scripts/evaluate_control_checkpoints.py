"""Evaluate CAS-retrain20 control checkpoints on the seven-region benchmark."""

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
from src.models import build_model
from evaluate_external_benchmark import ManifestDataset, evaluate_checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(PROJECT / "data" / "processed" / "benchmark_v2_regions_512" / "manifest_benchmark_v2.csv"))
    parser.add_argument("--input-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 2026, 777])
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--raw-dir", default=str(PROJECT / "reports" / "benchmark_v2_control_raw"))
    args = parser.parse_args()

    rows = [row for row in csv.DictReader(Path(args.manifest).open(encoding="utf-8-sig")) if int(row.get("eligible_external", 1)) == 1]
    regions = sorted({row["region"] for row in rows})
    dataset = ManifestDataset(rows, args.input_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw_dir = Path(args.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    for seed in args.seeds:
        run_dir = PROJECT / "outputs" / f"bench_v2_cas_retrain20_seed{seed}"
        checkpoint_path = run_dir / "best_model.pth"
        output_path = raw_dir / f"cas_retrain20_seed{seed}.json"
        if output_path.exists():
            print("SKIP", output_path, flush=True)
            continue
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model = build_model("BottleneckLiteASKUNetPlusPlus").to(device)
        model.load_state_dict(checkpoint["state_dict"])
        start = time.time()
        groups = evaluate_checkpoint(model, loader, regions, device, args.amp)
        record = {
            "model": "CAS-retrain20",
            "seed": seed,
            "run_dir": str(run_dir),
            "regions": regions,
            "num_samples": len(rows),
            "elapsed_seconds": time.time() - start,
            "groups": groups,
        }
        output_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        print("DONE", seed, output_path, flush=True)
        del model, checkpoint
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
