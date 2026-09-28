"""Train and evaluate one 128x128 target-domain few-shot adaptation run."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import albumentations as A
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

import os as _repo_os
PROJECT = Path(_repo_os.environ.get("LANDSLIDE_PROJECT_ROOT", Path(__file__).resolve().parents[1])).resolve()
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "scripts"))

from evaluate_external_benchmark import evaluate_checkpoint
from src.models import build_loss, build_model

MODEL_SPECS = {
    "ResUNet": ("ResUNet", "resunet"),
    "SegFormerB0": ("SegFormerB0", "segformerb0"),
}


class ManifestChipDataset(Dataset):
    def __init__(self, rows: list[dict], input_size: int, train: bool):
        self.rows = rows
        if train:
            transforms = [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
            ]
        else:
            transforms = [A.Resize(input_size, input_size)]
        self.transform = A.Compose(transforms)
        self.cache = None
        if train:
            self.cache = []
            resize = A.Resize(input_size, input_size)
            for row in self.rows:
                image = np.asarray(Image.open(row["image_path"]).convert("RGB"))
                mask = (np.asarray(Image.open(row["mask_path"]).convert("L")) > 0).astype(np.float32)
                resized = resize(image=image, mask=mask)
                self.cache.append((resized["image"], resized["mask"]))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        if self.cache is None:
            image = np.asarray(Image.open(row["image_path"]).convert("RGB"))
            mask = (np.asarray(Image.open(row["mask_path"]).convert("L")) > 0).astype(np.float32)
        else:
            image, mask = self.cache[index]
        transformed = self.transform(image=image, mask=mask)
        image = transformed["image"].astype(np.float32) / 255.0
        mask = transformed["mask"].astype(np.float32)
        return torch.from_numpy(image.transpose(2, 0, 1)), torch.from_numpy(mask[None]), row["region"]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_rows(path: Path) -> list[dict]:
    return list(csv.DictReader(path.open(encoding="utf-8-sig")))


def select_rows(rows: list[dict], region: str, seed: int, shot: int | None = None) -> list[dict]:
    selected = [row for row in rows if row["region"] == region and int(row["seed"]) == seed]
    if shot is not None:
        selected = [row for row in selected if int(row["shot"]) == shot]
    return selected


def set_train_mode(model: torch.nn.Module, mode: str) -> None:
    if mode == "full":
        model.train()
        return
    if mode == "decoder-only":
        model.train()
        model.encoder.eval()
        return
    raise ValueError(mode)
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(MODEL_SPECS), required=True)
    parser.add_argument("--mode", choices=["full", "decoder-only", "zero-shot"], required=True)
    parser.add_argument("--region", choices=["hokkaido_iburi_tobu", "lombok", "palu"], required=True)
    parser.add_argument("--shot", type=int, choices=[5, 10, 20], default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--support-seed", type=int, default=None)
    parser.add_argument("--eval-seed", type=int, default=None)
    parser.add_argument("--input-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--init-checkpoint", default=None)
    parser.add_argument("--support-csv", default=str(PROJECT / "data" / "processed" / "fewshot_target_splits" / "fewshot_support_seed42.csv"))
    parser.add_argument("--eval-csv", default=str(PROJECT / "data" / "processed" / "fewshot_target_splits" / "fewshot_eval_seed42.csv"))
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    if args.mode == "zero-shot" and args.shot is not None:
        raise ValueError("zero-shot should not specify --shot")
    if args.mode != "zero-shot" and args.shot is None:
        raise ValueError("Adapted modes require --shot")

    seed_everything(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(vars(args), ensure_ascii=False, indent=2), encoding="utf-8")

    support_seed = args.seed if args.support_seed is None else args.support_seed
    eval_seed = args.seed if args.eval_seed is None else args.eval_seed
    support_rows = [] if args.mode == "zero-shot" else select_rows(load_rows(Path(args.support_csv)), args.region, support_seed, args.shot)
    eval_rows = select_rows(load_rows(Path(args.eval_csv)), args.region, eval_seed)
    if not eval_rows:
        raise RuntimeError("No evaluation rows found")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name, slug = MODEL_SPECS[args.model]
    checkpoint_path = Path(args.init_checkpoint) if args.init_checkpoint else (
        PROJECT / "outputs" / f"bench_v2_{slug}_seed{args.seed}" / "best_model.pth"
    )
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = build_model(model_name).to(device)
    model.load_state_dict(checkpoint.get("state_dict", checkpoint))

    history = []
    if args.mode != "zero-shot":
        if not support_rows:
            raise RuntimeError("No support rows found")
        train_dataset = ManifestChipDataset(support_rows, args.input_size, train=True)
        train_loader = DataLoader(
            train_dataset,
            batch_size=min(args.batch_size, len(train_dataset)),
            shuffle=True,
            num_workers=0,
            pin_memory=True,
            drop_last=False,
        )
        if args.mode == "decoder-only":
            for name, parameter in model.named_parameters():
                parameter.requires_grad = name.startswith("decoder.") or name.startswith("segmentation_head.")
        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        if not trainable:
            raise RuntimeError("No trainable parameters")
        criterion = build_loss("PolyGHMDiceLoss").to(device)
        optimizer = torch.optim.Adam(trainable, lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(args.max_steps, 1),
            eta_min=args.min_lr,
        )
        scaler = torch.amp.GradScaler("cuda", enabled=args.amp)
        data_iterator = iter(train_loader)
        start = time.time()
        for step in range(1, args.max_steps + 1):
            try:
                image, mask, _ = next(data_iterator)
            except StopIteration:
                data_iterator = iter(train_loader)
                image, mask, _ = next(data_iterator)
            image = image.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            set_train_mode(model, args.mode)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=args.amp and device.type == "cuda"):
                logits = model(image)
                loss = criterion(logits, mask)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            if step == 1 or step % 25 == 0 or step == args.max_steps:
                history.append(
                    {
                        "step": step,
                        "train_loss": float(loss.item()),
                        "lr": float(optimizer.param_groups[0]["lr"]),
                        "elapsed_seconds": time.time() - start,
                    }
                )
        torch.save(
            {
                "model": model_name,
                "state_dict": model.state_dict(),
                "args": vars(args),
                "support_rows": [row["id"] for row in support_rows],
            },
            output_dir / "final_model.pth",
        )
        with (output_dir / "history.csv").open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
            writer.writeheader()
            writer.writerows(history)
    eval_dataset = ManifestChipDataset(eval_rows, args.input_size, train=False)
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=min(8, len(eval_dataset)),
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    model.eval()
    groups = evaluate_checkpoint(model, eval_loader, [args.region], device, args.amp)
    result = {
        "model": args.model,
        "model_name": model_name,
        "mode": args.mode,
        "region": args.region,
        "seed": args.seed,
        "support_seed": support_seed,
        "eval_seed": eval_seed,
        "shot": args.shot,
        "input_size": args.input_size,
        "init_checkpoint": str(checkpoint_path),
        "n_support": len(support_rows),
        "n_eval": len(eval_rows),
        "max_steps": 0 if args.mode == "zero-shot" else args.max_steps,
        "groups": groups,
    }
    (output_dir / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "groups"}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
