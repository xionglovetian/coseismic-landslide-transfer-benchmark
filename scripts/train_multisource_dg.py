import argparse
import csv
import hashlib
import random
import sys
import time
from collections import Counter
from pathlib import Path

import albumentations as A
import numpy as np
import torch
from torch import nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm

import os as _repo_os
PROJECT = Path(_repo_os.environ.get("LANDSLIDE_PROJECT_ROOT", Path(__file__).resolve().parents[1])).resolve()
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "scripts"))
from src.models import build_model, build_loss
from src.metrics import SegmentationMetrics
from evaluate_external_benchmark import ManifestDataset, evaluate_checkpoint

CAS_ROOT = PROJECT / "repos" / "AS-UNet" / "inputs" / "data_sum_moxizhen+bijie"
EXTERNAL_MANIFEST = PROJECT / "data" / "processed" / "external_regions_512" / "manifest_external.csv"

class MultiDomainDataset(Dataset):
    def __init__(self, rows, input_size, train):
        self.rows = rows
        transforms = [A.Resize(input_size, input_size)]
        if train:
            transforms += [A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5), A.RandomRotate90(p=0.5)]
        self.transform = A.Compose(transforms)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        image = np.array(Image.open(row["image_path"]).convert("RGB"))
        mask = (np.array(Image.open(row["mask_path"]).convert("L")) > 0).astype(np.float32)
        transformed = self.transform(image=image, mask=mask)
        image = transformed["image"].astype(np.float32) / 255.0
        mask = transformed["mask"].astype(np.float32)
        return torch.from_numpy(image.transpose(2, 0, 1)), torch.from_numpy(mask[None]), int(row["domain"])


def stable_holdout(identifier, ratio=10):
    value = int(hashlib.sha1(identifier.encode("utf-8")).hexdigest()[:8], 16)
    return value % ratio == 0


def build_rows(target_region, manifest_path=EXTERNAL_MANIFEST, exclude_regions=None, validation_mode="holdout"):
    rows = []
    for split in ["train", "val"]:
        for image_path in sorted((CAS_ROOT / split / "images").glob("*.png")):
            mask_path = CAS_ROOT / split / "masks" / image_path.name
            rows.append({
                "dataset": "CAS", "region": "cas", "split": split, "id": image_path.stem,
                "image_path": str(image_path), "mask_path": str(mask_path), "domain": 0,
            })
    excluded = set(exclude_regions or [])
    external = list(csv.DictReader(Path(manifest_path).open(encoding="utf-8-sig")))
    source_regions = sorted({row["region"] for row in external if row["region"] != target_region and row["region"] not in excluded})
    domain_map = {region: index + 1 for index, region in enumerate(source_regions)}
    for row in external:
        region = row["region"]
        if region == target_region or region in excluded:
            continue
        split = "train" if validation_mode == "cas_only" else ("val" if stable_holdout(f"{region}:{row['id']}") else "train")
        rows.append({
            "dataset": row["dataset"], "region": region, "split": split, "id": row["id"],
            "image_path": row["image_path"], "mask_path": row["mask_path"], "domain": domain_map[region],
        })
    for row in rows:
        row["domain_name"] = "cas" if row["domain"] == 0 else source_regions[row["domain"] - 1]
    return rows, source_regions


class FeatureStatisticsAlignment(nn.Module):
    def __init__(self, weight=0.2):
        super().__init__()
        self.weight = weight

    def forward(self, features, domains):
        stats = []
        domain_ids = []
        for domain in sorted(set(int(value) for value in domains.tolist())):
            mask = domains == domain
            if int(mask.sum().item()) == 0:
                continue
            selected = features[mask].float()
            mean = selected.mean(dim=(2, 3))
            std = selected.var(dim=(2, 3), unbiased=False).clamp_min(1e-6).sqrt()
            stats.append(torch.cat([mean, std], dim=1))
            domain_ids.append(domain)
        if len(stats) < 2:
            return features.new_zeros(())
        loss = features.new_zeros((), dtype=torch.float32)
        pairs = 0
        for i in range(len(stats)):
            for j in range(i + 1, len(stats)):
                loss = loss + (stats[i].mean(0) - stats[j].mean(0)).pow(2).mean()
                pairs += 1
        return self.weight * loss / max(pairs, 1)


def evaluate(model, loader, criterion, device, amp):
    model.eval()
    metrics = SegmentationMetrics()
    loss_sum = 0.0
    count = 0
    with torch.no_grad():
        for image, mask, _ in loader:
            image = image.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=amp and device.type == "cuda"):
                logits = model(image)
                loss = criterion(logits, mask)
            metrics.update(logits.float(), mask)
            loss_sum += loss.item() * image.size(0)
            count += image.size(0)
    result = metrics.compute()
    result["loss"] = loss_sum / max(count, 1)
    return result


def save_manifest(rows, path):
    fields = ["dataset", "region", "split", "id", "image_path", "mask_path", "domain", "domain_name"]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{key: row[key] for key in fields} for row in rows])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-region", default="longxi_river")
    parser.add_argument("--model", default="BottleneckLiteASKUNetPlusPlus")
    parser.add_argument("--init-checkpoint", default=str(PROJECT / "outputs" / "group_bottleneckliteaskunetpp_seed42" / "best_model.pth"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--input-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alignment-weight", type=float, default=0.2)
    parser.add_argument("--sampling", choices=["balanced", "uniform"], default="balanced")
    parser.add_argument("--output-dir", default=str(PROJECT / "outputs" / "dg_longxi_balanced_mmd_seed42"))
    parser.add_argument("--manifest", default=str(EXTERNAL_MANIFEST))
    parser.add_argument("--exclude-regions", nargs="*", default=None)
    parser.add_argument("--validation-mode", choices=["holdout", "cas_only"], default="holdout")
    parser.add_argument("--eval-regions", nargs="*", default=None)
    parser.add_argument("--split-name", default=None)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows, source_regions = build_rows(args.target_region, args.manifest, args.exclude_regions, args.validation_mode)
    train_rows = [row for row in rows if row["split"] == "train"]
    val_rows = [row for row in rows if row["split"] == "val"]
    split_name = args.split_name or f"multisource_exclude_{args.target_region}"
    manifest_path = PROJECT / "data" / "splits" / f"{split_name}.csv"
    save_manifest(rows, manifest_path)
    counts = Counter(row["domain"] for row in train_rows)
    train_dataset = MultiDomainDataset(train_rows, args.input_size, train=True)
    val_dataset = MultiDomainDataset(val_rows, args.input_size, train=False)
    if args.sampling == "balanced":
        weights = [1.0 / counts[row["domain"]] for row in train_rows]
        sampler = WeightedRandomSampler(weights, num_samples=len(train_rows), replacement=True)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    else:
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    print("source regions", source_regions, "train", len(train_rows), "val", len(val_rows), "domain counts", dict(counts), flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.model).to(device)
    checkpoint = torch.load(args.init_checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    criterion = build_loss("PolyGHMDiceLoss").to(device)
    alignment = FeatureStatisticsAlignment(args.alignment_weight).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

    captured = {}
    def capture_bottleneck(module, inputs, output):
        captured["feature"] = output
    hook = model.conv4_0.register_forward_hook(capture_bottleneck) if args.alignment_weight > 0 else None

    history = []
    best_iou = -1.0
    best_path = output_dir / "best_model.pth"
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_base = 0.0
        running_align = 0.0
        count = 0
        start = time.time()
        pbar = tqdm(train_loader, desc=f"DG epoch {epoch}/{args.epochs}")
        for step, (image, mask, domains) in enumerate(pbar):
            image = image.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            domains = domains.to(device, non_blocking=True)
            if step % args.grad_accum_steps == 0:
                optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
                logits = model(image)
                base_loss = criterion(logits, mask)
                align_loss = alignment(captured["feature"], domains) if args.alignment_weight > 0 else logits.new_zeros(())
                loss = base_loss + align_loss
            scaler.scale(loss / args.grad_accum_steps).backward()
            if ((step + 1) % args.grad_accum_steps == 0) or (step + 1 == len(train_loader)):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            running_base += base_loss.item() * image.size(0)
            running_align += align_loss.item() * image.size(0)
            count += image.size(0)
            pbar.set_postfix(base=f"{running_base / max(count, 1):.4f}", align=f"{running_align / max(count, 1):.4f}")
        scheduler.step()
        val_metrics = evaluate(model, val_loader, criterion, device, torch.cuda.is_available())
        record = {
            "epoch": epoch,
            "base_loss": running_base / max(count, 1),
            "alignment_loss": running_align / max(count, 1),
            "val_iou": val_metrics["iou"],
            "val_dice": val_metrics["dice"],
            "seconds": time.time() - start,
        }
        history.append(record)
        print(record, flush=True)
        if val_metrics["iou"] > best_iou:
            best_iou = val_metrics["iou"]
            torch.save({"model": args.model, "state_dict": model.state_dict(), "epoch": epoch, "val_metrics": val_metrics, "args": vars(args)}, best_path)
    if hook is not None:
        hook.remove()
    with (output_dir / "history.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    eval_regions = sorted(args.eval_regions or [args.target_region])
    target_rows = [row for row in csv.DictReader(Path(args.manifest).open(encoding="utf-8-sig")) if row["region"] in eval_regions and int(row.get("eligible_external", 1)) == 1]
    target_dataset = ManifestDataset(target_rows, args.input_size)
    target_loader = DataLoader(target_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    target_metrics = evaluate_checkpoint(model, target_loader, eval_regions, device, torch.cuda.is_available())
    method_name = "balanced_mmd" if args.alignment_weight > 0 else ("balanced_noalign" if args.sampling == "balanced" else "uniform_noalign")
    summary = {"method": method_name, "target_region": args.target_region, "eval_regions": eval_regions, "init_checkpoint": str(args.init_checkpoint), "seed": args.seed, "sampling": args.sampling, "alignment_weight": args.alignment_weight, "best_epoch": checkpoint["epoch"], "source_validation": checkpoint["val_metrics"], "target": target_metrics}
    (output_dir / "target_metrics.json").write_text(__import__("json").dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary, flush=True)


if __name__ == "__main__":
    main()
