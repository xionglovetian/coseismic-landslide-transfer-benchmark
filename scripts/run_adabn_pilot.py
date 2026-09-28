import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

import os as _repo_os
PROJECT = Path(_repo_os.environ.get("LANDSLIDE_PROJECT_ROOT", Path(__file__).resolve().parents[1])).resolve()
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "scripts"))
from src.models import build_model
from evaluate_external_benchmark import ManifestDataset, evaluate_checkpoint

MODELS = [
    ("UNet", "UNet", "group_unet"),
    ("Bottleneck-LiteASK", "BottleneckLiteASKUNetPlusPlus", "group_bottleneckliteaskunetpp"),
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(PROJECT / "data" / "processed" / "external_regions_512" / "manifest_external.csv"))
    parser.add_argument("--input-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    output_dir = PROJECT / "reports" / "external_adabn_raw"
    output_dir.mkdir(parents=True, exist_ok=True)
    all_rows = list(csv.DictReader(Path(args.manifest).open(encoding="utf-8-sig")))
    all_rows = [row for row in all_rows if int(row.get("eligible_external", 1)) == 1]
    regions = sorted({row["region"] for row in all_rows})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    for model_label, model_name, run_prefix in MODELS:
        if args.models and model_name not in args.models:
            continue
        for seed in args.seeds:
            for region in regions:
                output_path = output_dir / f"{run_prefix}_seed{seed}_{region}_alpha{args.alpha:.2f}.json"
                if output_path.exists() and not args.force:
                    print(f"SKIP {model_label} seed={seed} {region}", flush=True)
                    continue
                rows = [row for row in all_rows if row["region"] == region]
                dataset = ManifestDataset(rows, args.input_size)
                loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, persistent_workers=args.num_workers > 0)
                checkpoint = torch.load(PROJECT / "outputs" / f"{run_prefix}_seed{seed}" / "best_model.pth", map_location=device, weights_only=False)
                model = build_model(model_name).to(device)
                model.load_state_dict(checkpoint["state_dict"])

                bn_layers = [module for module in model.modules() if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d))]
                source_stats = [(module.running_mean.clone(), module.running_var.clone(), module.num_batches_tracked.clone()) for module in bn_layers]
                for module in bn_layers:
                    module.reset_running_stats()
                    module.momentum = 0.1
                model.train()
                start = time.time()
                with torch.no_grad():
                    for image, _, _ in loader:
                        image = image.to(device, non_blocking=True)
                        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=args.amp and device.type == "cuda"):
                            model(image)
                if args.alpha < 1.0:
                    for module, (source_mean, source_var, source_batches) in zip(bn_layers, source_stats):
                        target_mean = module.running_mean.clone()
                        target_var = module.running_var.clone()
                        module.running_mean.copy_((1.0 - args.alpha) * source_mean + args.alpha * target_mean)
                        module.running_var.copy_((1.0 - args.alpha) * source_var + args.alpha * target_var)
                        module.num_batches_tracked.copy_(source_batches)
                model.eval()
                eval_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, persistent_workers=args.num_workers > 0)
                metrics = evaluate_checkpoint(model, eval_loader, [region], device, args.amp)
                result = {
                    "model": model_label,
                    "seed": seed,
                    "region": region,
                    "method": "adabn",
                    "alpha": args.alpha,
                    "num_adaptation_samples": len(rows),
                    "elapsed_seconds": time.time() - start,
                    "groups": metrics,
                }
                output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
                print(f"DONE {model_label} seed={seed} {region} elapsed_min={(time.time()-start)/60:.2f}", flush=True)
                del model, checkpoint
                if device.type == "cuda":
                    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
