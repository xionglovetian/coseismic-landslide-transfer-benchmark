import argparse
import csv
import json
import sys
import time
from collections import defaultdict
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
from src.models import build_model

SEEDS = [42, 2026, 777]
REGIONS = ["a1", "a2", "a3", "b", "c1", "c2", "c3"]
THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7]
RUN_SPECS = [
    ("UNet", "UNet", "group_unet"),
    ("NestedUNet", "NestedUNet", "group_nestedunet"),
    ("AS_UNet", "AS_UNet", "group_asunet32"),
    ("U-Net++", "UNetPlusPlus", "group_unetpp"),
    ("ASK-UNet++", "ASKUNetPlusPlus", "group_askunetpp"),
    ("Bottleneck-LiteASK", "BottleneckLiteASKUNetPlusPlus", "group_bottleneckliteaskunetpp"),
]


class ManifestDataset(Dataset):
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
        return torch.from_numpy(image.transpose(2, 0, 1)), torch.from_numpy(mask[None]), row["region"]


def make_counts():
    return {region: {threshold: {"tp": 0, "fp": 0, "fn": 0} for threshold in THRESHOLDS} for region in REGIONS}


def finalize(counts):
    eps = 1e-12
    tp, fp, fn = float(counts["tp"]), float(counts["fp"]), float(counts["fn"])
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    return {
        "iou": tp / (tp + fp + fn + eps),
        "dice": 2 * tp / (2 * tp + fp + fn + eps),
        "precision": precision,
        "recall": recall,
    }


def evaluate(model, loader, device, amp):
    counts = make_counts()
    model.eval()
    with torch.inference_mode():
        for image, mask, regions in tqdm(loader, desc="threshold", leave=False):
            image = image.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=amp and device.type == "cuda"):
                probabilities = torch.sigmoid(model(image).float())
            target = (mask >= 0.5).float()
            for threshold in THRESHOLDS:
                pred = (probabilities >= threshold).float()
                tp = (pred * target).flatten(1).sum(1).cpu().tolist()
                fp = (pred * (1 - target)).flatten(1).sum(1).cpu().tolist()
                fn = ((1 - pred) * target).flatten(1).sum(1).cpu().tolist()
                for i, region in enumerate(regions):
                    bucket = counts[region][threshold]
                    bucket["tp"] += int(tp[i])
                    bucket["fp"] += int(fp[i])
                    bucket["fn"] += int(fn[i])
    return {
        region: {str(threshold): finalize(counts[region][threshold]) for threshold in THRESHOLDS}
        for region in REGIONS
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(PROJECT / "data" / "splits" / "resunet_bfa_group_manifest.csv"))
    parser.add_argument("--input-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--run-prefix", default=None)
    parser.add_argument("--run-model", default=None)
    parser.add_argument("--run-label", default="Custom")
    args = parser.parse_args()

    raw_dir = PROJECT / "reports" / "threshold_sensitivity_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(Path(args.manifest).open(encoding="utf-8-sig")))
    rows = [row for row in rows if row["region"] in REGIONS and int(row["augmented"]) == 0]
    dataset = ManifestDataset(rows, args.input_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, persistent_workers=args.num_workers > 0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    run_specs = RUN_SPECS
    if args.run_prefix:
        if not args.run_model:
            raise SystemExit("--run-prefix requires --run-model")
        run_specs = [(args.run_label, args.run_model, args.run_prefix)]
    for model_label, model_name, run_prefix in run_specs:
        if args.models and model_name not in args.models:
            continue
        for seed in args.seeds:
            output_path = raw_dir / f"{run_prefix}_seed{seed}.json"
            if output_path.exists() and not args.force:
                print(f"SKIP {model_label} seed={seed}", flush=True)
                continue
            run_dir = PROJECT / "outputs" / f"{run_prefix}_seed{seed}"
            checkpoint = torch.load(run_dir / "best_model.pth", map_location=device, weights_only=False)
            model = build_model(model_name).to(device)
            model.load_state_dict(checkpoint["state_dict"])
            print(f"START {model_label} seed={seed}", flush=True)
            start = time.time()
            groups = evaluate(model, loader, device, args.amp)
            result = {
                "model": model_label,
                "seed": seed,
                "thresholds": THRESHOLDS,
                "num_original_samples": len(rows),
                "elapsed_seconds": time.time() - start,
                "groups": groups,
            }
            output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"DONE {model_label} seed={seed} elapsed_min={(time.time() - start) / 60:.2f}", flush=True)
            del model, checkpoint
            if device.type == "cuda":
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
