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
        return (
            torch.from_numpy(image.transpose(2, 0, 1)),
            torch.from_numpy(mask[None]),
            row["region"],
            int(row["augmented"]),
        )


def empty_counts():
    return {"tp": 0, "fp": 0, "fn": 0, "tn": 0}


def finalize_counts(counts):
    eps = 1e-12
    tp = float(counts["tp"])
    fp = float(counts["fp"])
    fn = float(counts["fn"])
    tn = float(counts["tn"])
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    return {
        "iou": tp / (tp + fp + fn + eps),
        "dice": 2 * tp / (2 * tp + fp + fn + eps),
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "accuracy": (tp + tn) / (tp + tn + fp + fn + eps),
        "tp": counts["tp"],
        "fp": counts["fp"],
        "fn": counts["fn"],
        "tn": counts["tn"],
    }


def evaluate_checkpoint(model, loader, device, amp):
    counts = {
        "all": {region: empty_counts() for region in REGIONS},
        "original": {region: empty_counts() for region in REGIONS},
    }
    model.eval()
    with torch.inference_mode():
        for image, mask, regions, augmented in tqdm(loader, desc="evaluate", leave=False):
            image = image.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=amp and device.type == "cuda"):
                logits = model(image)
            pred = (torch.sigmoid(logits.float()) >= 0.5).float()
            target = (mask >= 0.5).float()
            tp = (pred * target).flatten(1).sum(1).cpu().tolist()
            fp = (pred * (1 - target)).flatten(1).sum(1).cpu().tolist()
            fn = ((1 - pred) * target).flatten(1).sum(1).cpu().tolist()
            tn = ((1 - pred) * (1 - target)).flatten(1).sum(1).cpu().tolist()
            for i, region in enumerate(regions):
                if region not in counts["all"]:
                    continue
                values = (int(tp[i]), int(fp[i]), int(fn[i]), int(tn[i]))
                for subset in ("all", "original") if int(augmented[i]) == 0 else ("all",):
                    bucket = counts[subset][region]
                    bucket["tp"] += values[0]
                    bucket["fp"] += values[1]
                    bucket["fn"] += values[2]
                    bucket["tn"] += values[3]
    return {
        subset: {region: finalize_counts(group_counts) for region, group_counts in per_subset.items()}
        for subset, per_subset in counts.items()
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

    raw_dir = PROJECT / "reports" / "cross_group_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(Path(args.manifest).open(encoding="utf-8-sig")))
    rows = [row for row in rows if row["region"] in REGIONS]
    dataset = ManifestDataset(rows, args.input_size)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
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
            checkpoint_path = run_dir / "best_model.pth"
            metrics_path = run_dir / "metrics.json"
            if not checkpoint_path.exists():
                raise FileNotFoundError(checkpoint_path)
            model = build_model(model_name).to(device)
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint["state_dict"])
            start = time.time()
            print(f"START {model_label} seed={seed}", flush=True)
            group_metrics = evaluate_checkpoint(model, loader, device, args.amp)
            result = {
                "model": model_label,
                "seed": seed,
                "run_dir": str(run_dir),
                "best_epoch": json.loads(metrics_path.read_text(encoding="utf-8"))["best_epoch"],
                "input_size": args.input_size,
                "batch_size": args.batch_size,
                "elapsed_seconds": time.time() - start,
                "num_samples": len(rows),
                "groups": group_metrics,
            }
            output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"DONE {model_label} seed={seed} elapsed_min={(time.time() - start) / 60:.2f}", flush=True)
            del model, checkpoint
            if device.type == "cuda":
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
