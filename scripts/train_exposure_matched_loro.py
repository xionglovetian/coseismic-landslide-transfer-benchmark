"""Exposure-matched leave-one-region-out trainer for the six-region extension."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

import os as _repo_os
PROJECT = Path(_repo_os.environ.get("LANDSLIDE_PROJECT_ROOT", Path(__file__).resolve().parents[1])).resolve()
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "scripts"))

from src.models import build_loss, build_model
from train_multisource_dg import MultiDomainDataset, build_rows, evaluate

DEFAULT_MANIFEST = PROJECT / "data" / "processed" / "benchmark_v2_regions_512" / "manifest_benchmark_v2.csv"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def write_rows_csv(rows: list[dict], path: Path) -> None:
    fields = ["dataset", "region", "split", "id", "image_path", "mask_path", "domain", "domain_name"]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{key: row[key] for key in fields} for row in rows])


def build_loro_rows(target_region: str, manifest_path: Path):
    rows, source_regions = build_rows(
        target_region=target_region,
        manifest_path=manifest_path,
        exclude_regions=[],
        validation_mode="cas_only",
    )
    cas_train = [row for row in rows if row.get("dataset") == "CAS" and row["split"] == "train"]
    cas_val = [row for row in rows if row["split"] == "val"]
    pooled_train = [row for row in rows if row["split"] == "train"]
    return cas_train, cas_val, pooled_train, source_regions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-region", required=True)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--model", default="BottleneckLiteASKUNetPlusPlus")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--steps", type=int, default=1120)
    parser.add_argument("--checkpoint-interval", type=int, default=280)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--input-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--init-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    if args.steps % args.checkpoint_interval != 0:
        raise ValueError("steps must be divisible by checkpoint interval")

    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(vars(args), ensure_ascii=False, indent=2), encoding="utf-8")

    cas_train, cas_val, pooled_train, source_regions = build_loro_rows(args.target_region, Path(args.manifest))
    if args.target_region in source_regions:
        raise RuntimeError(f"Leakage: target region appears in source_regions: {args.target_region}")
    write_rows_csv(pooled_train, output_dir / "pooled_train_rows.csv")
    write_rows_csv(cas_val, output_dir / "cas_val_rows.csv")

    domain_counts = Counter(row["domain"] for row in pooled_train)
    weights = [1.0 / domain_counts[row["domain"]] for row in pooled_train]
    sampler = WeightedRandomSampler(weights, num_samples=args.steps * args.batch_size, replacement=True)
    train_dataset = MultiDomainDataset(pooled_train, args.input_size, train=True)
    val_dataset = MultiDomainDataset(cas_val, args.input_size, train=False)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    if len(train_loader) != args.steps:
        raise RuntimeError(f"Expected {args.steps} steps, got {len(train_loader)}")

    composition = {
        "target_region": args.target_region,
        "source_regions": source_regions,
        "domain_counts": dict(domain_counts),
        "total_source_rows": len(pooled_train),
        "domain_slot_mass": 1.0 / len(domain_counts),
        "expected_exposure_per_domain": args.steps * args.batch_size / len(domain_counts),
    }
    (output_dir / "source_composition.json").write_text(json.dumps(composition, ensure_ascii=False, indent=2), encoding="utf-8")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    init_path = Path(args.init_checkpoint)
    checkpoint = torch.load(init_path, map_location=device, weights_only=False)
    model = build_model(args.model).to(device)
    model.load_state_dict(checkpoint.get("state_dict", checkpoint))
    criterion = build_loss("PolyGHMDiceLoss").to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps, eta_min=args.min_lr)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    torch.save({"model": args.model, "state_dict": model.state_dict(), "step": 0, "args": vars(args)}, checkpoint_dir / "step_0000.pth")

    history = []
    running_loss = 0.0
    count = 0
    start = time.time()
    model.train()
    pbar = tqdm(train_loader, desc=f"LORO {args.target_region} seed {args.seed}")
    for step, (image, mask, _) in enumerate(pbar, start=1):
        image = image.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=args.amp and device.type == "cuda"):
            logits = model(image)
            loss = criterion(logits, mask)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        running_loss += float(loss.item()) * image.size(0)
        count += image.size(0)
        pbar.set_postfix(loss=f"{running_loss / max(count, 1):.4f}")
        if step % 100 == 0:
            print(json.dumps({"step": step, "train_loss": running_loss / max(count, 1), "lr": optimizer.param_groups[0]["lr"], "elapsed_seconds": time.time() - start}), flush=True)
        if step % args.checkpoint_interval == 0:
            torch.save({"model": args.model, "state_dict": model.state_dict(), "step": step, "args": vars(args)}, checkpoint_dir / f"step_{step:04d}.pth")
            val_metrics = evaluate(model, val_loader, criterion, device, args.amp)
            history.append({"step": step, "train_loss": running_loss / max(count, 1), "lr": optimizer.param_groups[0]["lr"], "source_val_iou": val_metrics["iou"], "source_val_dice": val_metrics["dice"], "seconds": time.time() - start})
            print(json.dumps(history[-1]), flush=True)
        if step >= args.steps:
            break

    torch.save({"model": args.model, "state_dict": model.state_dict(), "step": args.steps, "args": vars(args)}, output_dir / "final_model.pth")
    with (output_dir / "history.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)
    final_metrics = evaluate(model, val_loader, criterion, device, args.amp)
    (output_dir / "metrics.json").write_text(json.dumps({"regime": "pooled_loro", "target_region": args.target_region, "seed": args.seed, "steps": args.steps, "source_regions": source_regions, "domain_counts": dict(domain_counts), "source_validation": final_metrics, "init_checkpoint": str(init_path)}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"complete": True, "target_region": args.target_region, "seed": args.seed, "source_val_iou": final_metrics["iou"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
