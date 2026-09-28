import argparse
import csv
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import os as _repo_os
PROJECT = Path(_repo_os.environ.get("LANDSLIDE_PROJECT_ROOT", Path(__file__).resolve().parents[1])).resolve()
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "scripts"))
from src.models import build_model
from evaluate_external_benchmark import ManifestDataset, evaluate_checkpoint

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--region", default="longxi_river")
parser.add_argument("--method", required=True)
args = parser.parse_args()
manifest = PROJECT / "data" / "processed" / "external_regions_512" / "manifest_external.csv"
rows = [row for row in csv.DictReader(manifest.open(encoding="utf-8-sig")) if row["region"] == args.region]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
model = build_model("BottleneckLiteASKUNetPlusPlus").to(device)
model.load_state_dict(checkpoint["state_dict"])
model.eval()
loader = DataLoader(ManifestDataset(rows, 128), batch_size=32, shuffle=False, num_workers=0, pin_memory=True)
metrics = evaluate_checkpoint(model, loader, [args.region], device, True)
result = {"method": args.method, "target_region": args.region, "best_epoch": checkpoint["epoch"], "source_validation": checkpoint["val_metrics"], "target": metrics}
Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({"method": args.method, "iou": metrics[args.region]["threshold_metrics"]["0.5"]["iou"], "dice": metrics[args.region]["threshold_metrics"]["0.5"]["dice"], "bf1_2": metrics[args.region]["f1_2_mean"], "hd95": metrics[args.region]["hd95_mean"]}, ensure_ascii=False))
