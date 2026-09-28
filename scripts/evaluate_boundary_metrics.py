import argparse
import csv
import json
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
REGIONS = ["a1", "a2", "a3", "b", "c1", "c2", "c3"]
RUN_SPECS = [
    ("UNet", "UNet", "group_unet"),
    ("NestedUNet", "NestedUNet", "group_nestedunet"),
    ("AS_UNet", "AS_UNet", "group_asunet32"),
    ("U-Net++", "UNetPlusPlus", "group_unetpp"),
    ("ASK-UNet++", "ASKUNetPlusPlus", "group_askunetpp"),
    ("Bottleneck-LiteASK", "BottleneckLiteASKUNetPlusPlus", "group_bottleneckliteaskunetpp"),
]
TOLERANCES = [2.0, 4.0]


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


def extract_boundary(mask):
    mask_u8 = mask.astype(np.uint8)
    eroded = cv2.erode(mask_u8, np.ones((3, 3), np.uint8), iterations=1)
    return mask_u8.astype(bool) & ~eroded.astype(bool)


def boundary_metrics(pred, target, tolerances=TOLERANCES):
    target_boundary = extract_boundary(target)
    if not target_boundary.any():
        return None
    pred_boundary = extract_boundary(pred)
    diagonal = float(np.hypot(*target.shape))
    if not pred_boundary.any():
        result = {"hd95": diagonal, "precision_2": 0.0, "recall_2": 0.0, "f1_2": 0.0, "precision_4": 0.0, "recall_4": 0.0, "f1_4": 0.0}
        return result

    distance_to_target = distance_transform_edt(~target_boundary)
    distance_to_pred = distance_transform_edt(~pred_boundary)
    pred_to_target = distance_to_target[pred_boundary]
    target_to_pred = distance_to_pred[target_boundary]
    result = {
        "hd95": max(
            float(np.percentile(pred_to_target, 95)),
            float(np.percentile(target_to_pred, 95)),
        )
    }
    for tolerance in tolerances:
        precision = float(np.mean(pred_to_target <= tolerance))
        recall = float(np.mean(target_to_pred <= tolerance))
        f1 = 2 * precision * recall / (precision + recall + 1e-12)
        key = str(int(tolerance))
        result[f"precision_{key}"] = precision
        result[f"recall_{key}"] = recall
        result[f"f1_{key}"] = f1
    return result


def evaluate_checkpoint(model, loader, device, amp):
    values = {
        region: defaultdict(list)
        for region in REGIONS
    }
    skipped = defaultdict(int)
    model.eval()
    with torch.inference_mode():
        for image, mask, regions in tqdm(loader, desc="boundary", leave=False):
            image = image.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=amp and device.type == "cuda"):
                logits = model(image)
            predictions = (torch.sigmoid(logits.float()) >= 0.5).cpu().numpy()[:, 0]
            targets = (mask >= 0.5).cpu().numpy()[:, 0]
            for i, region in enumerate(regions):
                metrics = boundary_metrics(predictions[i], targets[i])
                if metrics is None:
                    skipped[region] += 1
                    continue
                for key, value in metrics.items():
                    values[region][key].append(value)

    output = {}
    for region in REGIONS:
        region_values = values[region]
        output[region] = {
            "n_scored": len(region_values.get("hd95", [])),
            "n_skipped_empty_target": skipped[region],
        }
        for key, entries in region_values.items():
            output[region][f"{key}_mean"] = float(np.mean(entries))
            output[region][f"{key}_std"] = float(np.std(entries, ddof=1)) if len(entries) > 1 else 0.0
            output[region][f"{key}_median"] = float(np.median(entries))
    return output


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

    raw_dir = PROJECT / "reports" / "boundary_metrics_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(Path(args.manifest).open(encoding="utf-8-sig")))
    rows = [row for row in rows if row["region"] in REGIONS and int(row["augmented"]) == 0]
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
            model = build_model(model_name).to(device)
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint["state_dict"])
            print(f"START {model_label} seed={seed}", flush=True)
            start = time.time()
            group_metrics = evaluate_checkpoint(model, loader, device, args.amp)
            result = {
                "model": model_label,
                "seed": seed,
                "run_dir": str(run_dir),
                "input_size": args.input_size,
                "num_original_samples": len(rows),
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
