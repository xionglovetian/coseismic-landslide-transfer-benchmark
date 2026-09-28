import argparse
import csv
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
from PIL import Image
from scipy.ndimage import distance_transform_edt
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import os as _repo_os
PROJECT = Path(_repo_os.environ.get("LANDSLIDE_PROJECT_ROOT", Path(__file__).resolve().parents[1])).resolve()
sys.path.insert(0, str(PROJECT))
from src.models import build_model

SEEDS = [42, 2026, 777]
THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7]
TOLERANCES = [2, 4]
RUN_SPECS = [
    ("UNet", "UNet", "group_unet"),
    ("NestedUNet", "NestedUNet", "group_nestedunet"),
    ("AS_UNet", "AS_UNet", "group_asunet32"),
    ("U-Net++", "UNetPlusPlus", "group_unetpp"),
    ("ASK-UNet++", "ASKUNetPlusPlus", "group_askunetpp"),
    ("Bottleneck-LiteASK", "BottleneckLiteASKUNetPlusPlus", "group_bottleneckliteaskunetpp"),
    ("ResUNet", "ResUNet", "bench_v2_resunet"),
    ("DeepLabV3+", "DeepLabV3Plus", "bench_v2_deeplabv3plus"),
    ("SegFormer-B0", "SegFormerB0", "bench_v2_segformerb0"),
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

def empty_counts():
    return {"tp": 0.0, "fp": 0.0, "fn": 0.0, "tn": 0.0}

def finalize_counts(counts):
    eps = 1e-12
    tp, fp, fn, tn = counts["tp"], counts["fp"], counts["fn"], counts["tn"]
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    return {
        "iou": tp / (tp + fp + fn + eps),
        "dice": 2 * tp / (2 * tp + fp + fn + eps),
        "f1": 2 * precision * recall / (precision + recall + eps),
        "precision": precision,
        "recall": recall,
        "accuracy": (tp + tn) / (tp + tn + fp + fn + eps),
    }

def extract_boundary(mask):
    mask_u8 = mask.astype(np.uint8)
    eroded = cv2.erode(mask_u8, np.ones((3, 3), np.uint8), iterations=1)
    return mask_u8.astype(bool) & ~eroded.astype(bool)

def boundary_metrics(pred, target):
    target_boundary = extract_boundary(target)
    if not target_boundary.any():
        return None
    pred_boundary = extract_boundary(pred)
    diagonal = float(np.hypot(*target.shape))
    if not pred_boundary.any():
        return {"hd95": diagonal, "f1_2": 0.0, "f1_4": 0.0, "precision_2": 0.0, "recall_2": 0.0, "precision_4": 0.0, "recall_4": 0.0}
    distance_to_target = distance_transform_edt(~target_boundary)
    distance_to_pred = distance_transform_edt(~pred_boundary)
    pred_to_target = distance_to_target[pred_boundary]
    target_to_pred = distance_to_pred[target_boundary]
    result = {"hd95": max(float(np.percentile(pred_to_target, 95)), float(np.percentile(target_to_pred, 95)))}
    for tolerance in TOLERANCES:
        precision = float(np.mean(pred_to_target <= tolerance))
        recall = float(np.mean(target_to_pred <= tolerance))
        f1 = 2 * precision * recall / (precision + recall + 1e-12)
        result[f"precision_{tolerance}"] = precision
        result[f"recall_{tolerance}"] = recall
        result[f"f1_{tolerance}"] = f1
    return result

def evaluate_checkpoint(model, loader, regions, device, amp):
    pixel_counts = {region: {threshold: empty_counts() for threshold in THRESHOLDS} for region in regions}
    boundary_values = {region: defaultdict(list) for region in regions}
    boundary_skipped = defaultdict(int)
    model.eval()
    with torch.inference_mode():
        for image, mask, batch_regions in tqdm(loader, desc="external-benchmark", leave=False):
            image = image.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=amp and device.type == "cuda"):
                probabilities = torch.sigmoid(model(image).float())
            target = (mask >= 0.5).float()
            pred_by_threshold = {}
            for threshold in THRESHOLDS:
                pred_by_threshold[threshold] = (probabilities >= threshold).float()
            for threshold in THRESHOLDS:
                pred = pred_by_threshold[threshold]
                tp = (pred * target).flatten(1).sum(1).cpu().tolist()
                fp = (pred * (1 - target)).flatten(1).sum(1).cpu().tolist()
                fn = ((1 - pred) * target).flatten(1).sum(1).cpu().tolist()
                tn = ((1 - pred) * (1 - target)).flatten(1).sum(1).cpu().tolist()
                for i, region in enumerate(batch_regions):
                    bucket = pixel_counts[region][threshold]
                    bucket["tp"] += int(tp[i]); bucket["fp"] += int(fp[i]); bucket["fn"] += int(fn[i]); bucket["tn"] += int(tn[i])
            predictions = pred_by_threshold[0.5].cpu().numpy()[:, 0].astype(bool)
            targets = target.cpu().numpy()[:, 0].astype(bool)
            for i, region in enumerate(batch_regions):
                metrics = boundary_metrics(predictions[i], targets[i])
                if metrics is None:
                    boundary_skipped[region] += 1
                    continue
                for key, value in metrics.items():
                    boundary_values[region][key].append(value)

    output = {}
    for region in regions:
        output[region] = {
            "threshold_metrics": {str(threshold): finalize_counts(pixel_counts[region][threshold]) for threshold in THRESHOLDS},
            "boundary_n_scored": len(boundary_values[region].get("hd95", [])),
            "boundary_n_skipped_empty_target": boundary_skipped[region],
        }
        for key, entries in boundary_values[region].items():
            output[region][f"{key}_mean"] = float(np.mean(entries))
            output[region][f"{key}_std"] = float(np.std(entries, ddof=1)) if len(entries) > 1 else 0.0
    return output

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(PROJECT / "data" / "processed" / "external_regions_512" / "manifest_external.csv"))
    parser.add_argument("--input-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--raw-dir", default=str(PROJECT / "reports" / "external_regions_raw"))
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(Path(args.manifest).open(encoding="utf-8-sig")))
    rows = [row for row in rows if int(row.get("eligible_external", 1)) == 1]
    regions = sorted({row["region"] for row in rows})
    print("regions", regions, "samples", len(rows), flush=True)
    dataset = ManifestDataset(rows, args.input_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, persistent_workers=args.num_workers > 0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    for model_label, model_name, run_prefix in RUN_SPECS:
        if args.models and model_name not in args.models:
            continue
        for seed in args.seeds:
            output_path = raw_dir / f"{run_prefix}_seed{seed}.json"
            if output_path.exists() and not args.force:
                print(f"SKIP {model_label} seed={seed}", flush=True)
                continue
            run_dir = PROJECT / "outputs" / f"{run_prefix}_seed{seed}"
            checkpoint_path = run_dir / "best_model.pth"
            if not checkpoint_path.exists():
                raise FileNotFoundError(checkpoint_path)
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            model = build_model(model_name).to(device)
            model.load_state_dict(checkpoint["state_dict"])
            print(f"START {model_label} seed={seed}", flush=True)
            start = time.time()
            group_metrics = evaluate_checkpoint(model, loader, regions, device, args.amp)
            result = {
                "model": model_label,
                "seed": seed,
                "run_dir": str(run_dir),
                "input_size": args.input_size,
                "regions": regions,
                "num_samples": len(rows),
                "elapsed_seconds": time.time() - start,
                "groups": group_metrics,
            }
            output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"DONE {model_label} seed={seed} elapsed_min={(time.time() - start) / 60:.2f}", flush=True)
            del model, checkpoint
            if device.type == "cuda":
                torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
