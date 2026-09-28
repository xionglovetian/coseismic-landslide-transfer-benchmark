"""Evaluate lightweight domain-shift perturbations on target regions."""

from __future__ import annotations

import argparse
import csv
import io
import sys
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

import os as _repo_os
PROJECT = Path(_repo_os.environ.get("LANDSLIDE_PROJECT_ROOT", Path(__file__).resolve().parents[1])).resolve()
sys.path.insert(0, str(PROJECT))

from src.models import build_model

REGIONS = ["hokkaido_iburi_tobu", "lombok", "palu"]
MODES = ["none", "brightness_low", "brightness_high", "blur1", "blur2", "jpeg30", "jpeg60", "downscale64"]


class PerturbedDataset(Dataset):
    def __init__(self, rows, input_size, mode):
        self.rows = rows
        self.resize = A.Resize(input_size, input_size)
        self.mode = mode

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        image = np.asarray(Image.open(row["image_path"]).convert("RGB"))
        mask = (np.asarray(Image.open(row["mask_path"]).convert("L")) > 0).astype(np.float32)
        transformed = self.resize(image=image, mask=mask)
        image = transformed["image"].astype(np.float32)
        mask = transformed["mask"]
        if self.mode == "brightness_low":
            image = image * 0.75
        elif self.mode == "brightness_high":
            image = np.clip(image * 1.25, 0, 255)
        elif self.mode == "blur1":
            image = cv2.GaussianBlur(image, (5, 5), 1.0)
        elif self.mode == "blur2":
            image = cv2.GaussianBlur(image, (7, 7), 2.0)
        elif self.mode in {"jpeg30", "jpeg60"}:
            quality = 30 if self.mode == "jpeg30" else 60
            buffer = io.BytesIO()
            Image.fromarray(image.astype(np.uint8)).save(buffer, format="JPEG", quality=quality)
            buffer.seek(0)
            image = np.asarray(Image.open(buffer).convert("RGB"), dtype=np.float32)
        elif self.mode == "downscale64":
            small = cv2.resize(image, (64, 64), interpolation=cv2.INTER_AREA)
            image = cv2.resize(small, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
        image = np.clip(image, 0, 255).astype(np.float32) / 255.0
        return torch.from_numpy(image.transpose(2, 0, 1)), torch.from_numpy(mask[None]), row["region"]


def evaluate(model, loader, device, amp):
    tp = fp = fn = tn = 0
    model.eval()
    with torch.inference_mode():
        for image, mask, _ in loader:
            image = image.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=amp and device.type == "cuda"):
                probabilities = torch.sigmoid(model(image).float()).cpu()
            predicted = probabilities >= 0.5
            target = mask >= 0.5
            tp += int((predicted & target).sum())
            fp += int((predicted & ~target).sum())
            fn += int((~predicted & target).sum())
            tn += int((~predicted & ~target).sum())
    eps = 1e-12
    return {"iou": tp / (tp + fp + fn + eps), "dice": 2 * tp / (2 * tp + fp + fn + eps), "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model", default="ResUNet")
    parser.add_argument("--manifest", default=str(PROJECT / "data" / "processed" / "benchmark_v2_regions_512" / "manifest_benchmark_v2.csv"))
    parser.add_argument("--input-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--output", default=str(PROJECT / "reports" / "p4_domain_shift.csv"))
    args = parser.parse_args()

    manifest = list(csv.DictReader(Path(args.manifest).open(encoding="utf-8-sig")))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = build_model(args.model).to(device)
    model.load_state_dict(checkpoint.get("state_dict", checkpoint))

    records = []
    baseline = {}
    for region in REGIONS:
        rows = [row for row in manifest if row["region"] == region]
        for mode in MODES:
            dataset = PerturbedDataset(rows, args.input_size, mode)
            loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
            result = evaluate(model, loader, device, args.amp)
            record = {"region": region, "mode": mode, **result}
            records.append(record)
            if mode == "none":
                baseline[region] = result["iou"]
            print(region, mode, result["iou"], flush=True)
    for row in records:
        row["delta_iou"] = row["iou"] - baseline[row["region"]]
    with Path(args.output).open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    print("wrote", args.output)


if __name__ == "__main__":
    main()
