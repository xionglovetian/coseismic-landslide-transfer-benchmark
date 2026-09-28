"""Summarize the Q2 X1 resolution experiment."""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

import os as _repo_os
PROJECT = Path(_repo_os.environ.get("LANDSLIDE_PROJECT_ROOT", Path(__file__).resolve().parents[1])).resolve()
REPORTS = PROJECT / "reports"
SEEDS = [42, 2026, 777]
MODELS = [("ResUNet", "resunet"), ("SegFormer-B0", "segformerb0")]


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def mean(values):
    return sum(values) / len(values)


def sd(values):
    return statistics.stdev(values) if len(values) > 1 else float("nan")


rows = []
for label, slug in MODELS:
    for seed in SEEDS:
        if slug == "resunet":
            base = load(REPORTS / "q2_x2_augmentation_raw" / f"q2_x2_aug_none_resunet_seed{seed}.json")
        else:
            base = load(REPORTS / "q2_x1_128ref_eval_raw" / f"q2_x1_128ref_none_segformerb0_seed{seed}_input128.json")
        a1a = load(REPORTS / "q2_x1_resolution_raw" / f"q2_x1_256_none_{slug}_seed{seed}.json")
        a1c = load(REPORTS / "q2_x1_a1c_raw" / f"q2_x1_a1c_none_{slug}_seed{seed}.json")
        row = {
            "model": label,
            "seed": seed,
            "source_128": base["source_validation"]["iou"],
            "source_256_retrained": a1a["source_validation"]["iou"],
            "source_128weights_at256": a1c["source_validation"]["iou"],
            "target_128": base["target_macro_iou"],
            "target_256_retrained": a1a["target_macro_iou"],
            "target_128weights_at256": a1c["target_macro_iou"],
        }
        row["target_delta_256_minus_128"] = row["target_256_retrained"] - row["target_128"]
        row["target_delta_a1c_minus_128"] = row["target_128weights_at256"] - row["target_128"]
        rows.append(row)

out_dir = REPORTS / "q2_x1_summary"
out_dir.mkdir(parents=True, exist_ok=True)
with (out_dir / "per_run.csv").open("w", newline="", encoding="utf-8-sig") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

grouped = []
for label, _ in MODELS:
    subset = [row for row in rows if row["model"] == label]
    grouped.append({
        "model": label,
        "n_seeds": len(subset),
        "target_128_mean": mean([row["target_128"] for row in subset]),
        "target_256_mean": mean([row["target_256_retrained"] for row in subset]),
        "target_a1c_mean": mean([row["target_128weights_at256"] for row in subset]),
        "delta_256_minus_128_mean": mean([row["target_delta_256_minus_128"] for row in subset]),
        "delta_256_minus_128_sd": sd([row["target_delta_256_minus_128"] for row in subset]),
        "delta_a1c_minus_128_mean": mean([row["target_delta_a1c_minus_128"] for row in subset]),
        "delta_a1c_minus_128_sd": sd([row["target_delta_a1c_minus_128"] for row in subset]),
    })
with (out_dir / "grouped.csv").open("w", newline="", encoding="utf-8-sig") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(grouped[0].keys()))
    writer.writeheader()
    writer.writerows(grouped)

print(json.dumps({"per_run": len(rows), "output_dir": str(out_dir)}, ensure_ascii=False))
