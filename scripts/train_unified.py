import argparse, csv, json, random, sys, time
from pathlib import Path
from collections import OrderedDict

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

import os as _repo_os
PROJECT = Path(_repo_os.environ.get("LANDSLIDE_PROJECT_ROOT", Path(__file__).resolve().parents[1])).resolve()
sys.path.insert(0, str(PROJECT))
from src.datasets import build_dataset
from src.models import build_model, build_loss
from src.metrics import SegmentationMetrics


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def evaluate(model, loader, criterion, device, amp=False):
    model.eval()
    metrics = SegmentationMetrics()
    loss_sum = 0.0
    count = 0
    with torch.no_grad():
        for image, mask, _ in loader:
            image = image.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                logits = model(image)
                loss = criterion(logits, mask)
            metrics.update(logits.float(), mask)
            loss_sum += loss.item() * image.size(0)
            count += image.size(0)
    result = metrics.compute()
    result["loss"] = loss_sum / max(count, 1)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="AS_UNet", choices=["UNet", "NestedUNet", "AS_UNet", "UNetPlusPlus", "ASKUNetPlusPlus", "LiteASKUNetPlusPlus", "DeepLiteASKUNetPlusPlus", "BottleneckLiteASKUNetPlusPlus", "ResUNet", "DeepLabV3Plus", "SegFormerB0"])
    parser.add_argument("--loss", default="PolyGHMDiceLoss", choices=["BCEDiceLoss", "PolyGHMDiceLoss", "BoundaryAwarePolyGHMDiceLoss"])
    parser.add_argument("--train-root", default=str(PROJECT / "repos" / "AS-UNet" / "inputs" / "data_sum_moxizhen+bijie"))
    parser.add_argument("--external-root", default=str(PROJECT / "data" / "processed" / "resunet_bfa_grouped_224"))
    parser.add_argument("--output-dir", default=str(PROJECT / "outputs" / "group_asunet"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--input-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-interval", type=int, default=5)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--checkpoint-interval", type=int, default=0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--skip-external-eval", action="store_true")
    parser.add_argument("--init-checkpoint", default=None)
    parser.add_argument("--train-fraction", type=float, default=1.0)
    parser.add_argument("--subset-seed", type=int, default=42)
    parser.add_argument("--augmentation", choices=["none", "current", "strong"], default="current")
    args = parser.parse_args()
    if not 0.0 < args.train_fraction <= 1.0:
        raise ValueError("--train-fraction must be in (0, 1]")

    seed_everything(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(vars(args), ensure_ascii=False, indent=2), encoding="utf-8")

    train_dataset = build_dataset(args.train_root, "train", args.input_size, train=True, augmentation=args.augmentation)
    full_train_size = len(train_dataset)
    subset_indices = list(range(full_train_size))
    if args.train_fraction < 1.0:
        subset_size = max(1, int(round(full_train_size * args.train_fraction)))
        subset_indices = sorted(random.Random(args.subset_seed).sample(range(full_train_size), subset_size))
        train_dataset = Subset(train_dataset, subset_indices)
        (output_dir / "train_subset.json").write_text(
            json.dumps({"full_train_size": full_train_size, "subset_size": subset_size, "fraction": args.train_fraction, "indices": subset_indices}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    val_dataset = build_dataset(args.train_root, "val", args.input_size, train=False)
    external_dataset = None if args.skip_external_eval else build_dataset(args.external_root, "test", args.input_size, train=False)
    external_text = "skipped" if external_dataset is None else str(len(external_dataset))
    print(f"train={len(train_dataset)} val={len(val_dataset)} external={external_text}", flush=True)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    external_loader = None if external_dataset is None else DataLoader(external_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.model).to(device)
    if args.init_checkpoint:
        init_checkpoint = torch.load(args.init_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(init_checkpoint.get("state_dict", init_checkpoint))
    criterion = build_loss(args.loss).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1), eta_min=1e-5)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)

    history = []
    best_iou = -1.0
    best_path = output_dir / "best_model.pth"
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        count = 0
        start = time.time()
        pbar = tqdm(train_loader, desc=f"seed {args.seed} epoch {epoch}/{args.epochs}")
        optimizer.zero_grad(set_to_none=True)
        for step, (image, mask, _) in enumerate(pbar):
            image = image.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=args.amp):
                logits = model(image)
                loss = criterion(logits, mask)
            scaler.scale(loss / args.grad_accum_steps).backward()
            should_step = ((step + 1) % args.grad_accum_steps == 0) or (step + 1 == len(train_loader))
            if should_step:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            running_loss += loss.item() * image.size(0)
            count += image.size(0)
            pbar.set_postfix(loss=f"{running_loss / max(count, 1):.4f}")
        scheduler.step()
        train_loss = running_loss / max(count, 1)
        should_validate = (epoch % args.val_interval == 0) or (epoch == args.epochs)
        if should_validate:
            val_metrics = evaluate(model, val_loader, criterion, device, amp=args.amp)
            row = OrderedDict(epoch=epoch, train_loss=train_loss, lr=optimizer.param_groups[0]["lr"], seconds=time.time() - start, **{f"val_{k}": v for k, v in val_metrics.items()})
            history.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
            if val_metrics["iou"] > best_iou:
                best_iou = val_metrics["iou"]
                torch.save({"model": args.model, "state_dict": model.state_dict(), "epoch": epoch, "val_metrics": val_metrics, "args": vars(args)}, best_path)
                print(f"saved best model: {best_path}", flush=True)
        else:
            history.append(OrderedDict(epoch=epoch, train_loss=train_loss, lr=optimizer.param_groups[0]["lr"], seconds=time.time() - start))
        if args.checkpoint_interval > 0 and (epoch % args.checkpoint_interval == 0 or epoch == args.epochs):
            interval_path = output_dir / f"epoch_{epoch:03d}.pth"
            torch.save({"model": args.model, "state_dict": model.state_dict(), "epoch": epoch, "args": vars(args)}, interval_path)
            print(f"saved interval checkpoint: {interval_path}", flush=True)

    with (output_dir / "history.csv").open("w", newline="", encoding="utf-8-sig") as f:
        fieldnames = sorted({k for row in history for k in row.keys()})
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    final_val = evaluate(model, val_loader, criterion, device, amp=args.amp)
    summary = {
        "model": args.model,
        "loss": args.loss,
        "seed": args.seed,
        "best_epoch": checkpoint["epoch"],
        "cas_val": final_val,
    }
    if external_loader is not None:
        summary["resunet_bfa_external_test"] = evaluate(model, external_loader, criterion, device, amp=args.amp)
    (output_dir / "metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
