"""Write per-tile segmentation metrics for one checkpoint."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import albumentations as A
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import os as _repo_os
PROJECT = Path(_repo_os.environ.get("LANDSLIDE_PROJECT_ROOT", Path(__file__).resolve().parents[1])).resolve()
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "scripts"))

from evaluate_external_benchmark import boundary_metrics
from src.models import build_model

DEFAULT_MANIFEST = PROJECT / "data" / "processed" / "benchmark_v2_regions_512" / "manifest_benchmark_v2.csv"
DEFAULT_REGIONS = [
    "hokkaido_iburi_tobu",
    "lombok",
    "palu",
    "wenchuan",
    "longxi_river",
    "jiuzhai_valley",
]


class TileDataset(Dataset):
    def __init__(self, rows, input_size):
        self.rows = rows
        self.transform = A.Compose([A.Resize(input_size, input_size)])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        image = np.asarray(Image.open(row["image_path"]).convert("RGB"))
        mask = (np.asarray(Image.open(row["mask_path"]).convert("L")) > 0).astype(np.float32)
        transformed = self.transform(image=image, mask=mask)
        image = transformed["image"].astype(np.float32) / 255.0
        mask = transformed["mask"].astype(np.float32)
        return torch.from_numpy(image.transpose(2, 0, 1)), torch.from_numpy(mask[None]), row["region"], row["id"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--regions", nargs="+", default=DEFAULT_REGIONS)
    parser.add_argument("--input-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--regime", required=True)
    parser.add_argument("--role", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--step", type=int, default=1120)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    output_path = Path(args.output)
    if output_path.exists() and not args.force:
        print(f"SKIP {output_path}", flush=True)
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = [
        row for row in csv.DictReader(Path(args.manifest).open(encoding="utf-8-sig"))
        if row["region"] in args.regions and int(row.get("eligible_external", 1)) == 1
    ]
    found = sorted({row["region"] for row in rows})
    missing = sorted(set(args.regions) - set(found))
    if missing:
        raise RuntimeError(f"Missing regions: {missing}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = build_model(args.model).to(device)
    model.load_state_dict(checkpoint.get("state_dict", checkpoint))
    model.eval()
    loader = DataLoader(TileDataset(rows, args.input_size), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    start = time.time()
    fieldnames = ["experiment", "run_id", "regime", "role", "seed", "step", "buffer_m", "region", "id", "threshold", "tp", "fp", "fn", "tn", "hd95", "f1_2", "f1_4"]
    with output_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        with torch.inference_mode():
            for image, mask, regions, identifiers in tqdm(loader, desc=args.run_id, leave=False):
                image = image.to(device, non_blocking=True)
                mask = mask.to(device, non_blocking=True)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=args.amp and device.type == "cuda"):
                    probabilities = torch.sigmoid(model(image).float())
                target = (mask >= 0.5)
                pred = (probabilities >= args.threshold)
                tp = (pred & target).flatten(1).sum(1).cpu().tolist()
                fp = (pred & ~target).flatten(1).sum(1).cpu().tolist()
                fn = (~pred & target).flatten(1).sum(1).cpu().tolist()
                tn = (~pred & ~target).flatten(1).sum(1).cpu().tolist()
                pred_np = pred.cpu().numpy()[:, 0].astype(bool)
                target_np = target.cpu().numpy()[:, 0].astype(bool)
                for index, region in enumerate(regions):
                    boundary = boundary_metrics(pred_np[index], target_np[index]) or {}
                    writer.writerow({
                        "experiment": "E1",
                        "run_id": args.run_id,
                        "regime": args.regime,
                        "role": args.role,
                        "seed": args.seed,
                        "step": args.step,
                        "buffer_m": "na",
                        "region": region,
                        "id": identifiers[index],
                        "threshold": args.threshold,
                        "tp": int(tp[index]),
                        "fp": int(fp[index]),
                        "fn": int(fn[index]),
                        "tn": int(tn[index]),
                        "hd95": boundary.get("hd95", ""),
                        "f1_2": boundary.get("f1_2", ""),
                        "f1_4": boundary.get("f1_4", ""),
                    })
    metadata = {
        "model": args.model,
        "checkpoint": args.checkpoint,
        "run_id": args.run_id,
        "regime": args.regime,
        "role": args.role,
        "seed": args.seed,
        "step": args.step,
        "regions": found,
        "threshold": args.threshold,
        "rows": len(rows),
        "output": str(output_path),
        "elapsed_seconds": time.time() - start,
    }
    output_path.with_suffix(".json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
