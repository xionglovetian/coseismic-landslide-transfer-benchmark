"""Summarize the Q2 X2 augmentation grid."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import os as _repo_os
PROJECT = Path(_repo_os.environ.get("LANDSLIDE_PROJECT_ROOT", Path(__file__).resolve().parents[1])).resolve()
DEFAULT_RAW = PROJECT / "reports" / "q2_x2_augmentation_raw"
DEFAULT_OUT = PROJECT / "reports" / "q2_x2_augmentation"
EXPECTED = [
    ("ResUNet", "resunet"),
    ("BottleneckLiteASKUNetPlusPlus", "bottleneckliteask"),
]
PRIMARY_MODELS = ["ResUNet", "BottleneckLiteASKUNetPlusPlus"]
SEEDS = [42, 2026, 777]
AUGMENTATIONS = ["current", "none", "strong"]
REGIONS = [
    "hokkaido_iburi_tobu",
    "lombok",
    "palu",
    "wenchuan",
    "longxi_river",
    "jiuzhai_valley",
]


def mean(values):
    return sum(values) / len(values) if values else float("nan")


def sd(values):
    return statistics.stdev(values) if len(values) > 1 else float("nan")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default=str(DEFAULT_RAW))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT))
    args = parser.parse_args()
    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = {}
    for model, slug in EXPECTED:
        for seed in SEEDS:
            for augmentation in AUGMENTATIONS:
                name = f"q2_x2_aug_{augmentation}_{slug}_seed{seed}.json"
                path = raw_dir / name
                if not path.exists():
                    continue
                data = json.loads(path.read_text(encoding="utf-8"))
                records[(model, seed, augmentation)] = data

    per_run = []
    for model, _ in EXPECTED:
        for seed in SEEDS:
            for augmentation in AUGMENTATIONS:
                data = records.get((model, seed, augmentation))
                if data is None:
                    continue
                row = {
                    "model": model,
                    "seed": seed,
                    "augmentation": augmentation,
                    "source_val_iou": data["source_validation"]["iou"],
                    "target_macro_iou": data["target_macro_iou"],
                }
                for region in REGIONS:
                    row[f"target_{region}_iou"] = data["target_threshold_05"][region]["iou"]
                per_run.append(row)
    write_csv(output_dir / "per_run.csv", per_run)

    grouped = []
    for model, _ in EXPECTED:
        for augmentation in AUGMENTATIONS:
            rows = [row for row in per_run if row["model"] == model and row["augmentation"] == augmentation]
            if not rows:
                continue
            grouped.append(
                {
                    "model": model,
                    "augmentation": augmentation,
                    "n_seeds": len(rows),
                    "source_val_iou_mean": mean([row["source_val_iou"] for row in rows]),
                    "source_val_iou_sd": sd([row["source_val_iou"] for row in rows]),
                    "target_macro_iou_mean": mean([row["target_macro_iou"] for row in rows]),
                    "target_macro_iou_sd": sd([row["target_macro_iou"] for row in rows]),
                }
            )
    write_csv(output_dir / "grouped.csv", grouped)

    paired = []
    for model, _ in EXPECTED:
        for seed in SEEDS:
            current = records.get((model, seed, "current"))
            if current is None:
                continue
            for augmentation in ("none", "strong"):
                candidate = records.get((model, seed, augmentation))
                if candidate is None:
                    continue
                paired.append(
                    {
                        "model": model,
                        "seed": seed,
                        "comparison": f"{augmentation}-current",
                        "source_val_iou_delta": candidate["source_validation"]["iou"] - current["source_validation"]["iou"],
                        "target_macro_iou_delta": candidate["target_macro_iou"] - current["target_macro_iou"],
                    }
                )
    write_csv(output_dir / "paired_deltas.csv", paired)

    summary_by_comparison = []
    for comparison in ("none-current", "strong-current"):
        rows = [row for row in paired if row["comparison"] == comparison]
        if not rows:
            continue
        summary_by_comparison.append(
            {
                "comparison": comparison,
                "n_pairs": len(rows),
                "source_val_iou_delta_mean": mean([row["source_val_iou_delta"] for row in rows]),
                "source_val_iou_delta_sd": sd([row["source_val_iou_delta"] for row in rows]),
                "target_macro_iou_delta_mean": mean([row["target_macro_iou_delta"] for row in rows]),
                "target_macro_iou_delta_sd": sd([row["target_macro_iou_delta"] for row in rows]),
            }
        )
    write_csv(output_dir / "paired_summary.csv", summary_by_comparison)

    primary_paired = []
    for comparison in ("none-current", "strong-current"):
        rows = [row for row in paired if row["comparison"] == comparison and row["model"] in PRIMARY_MODELS]
        if not rows:
            continue
        primary_paired.append(
            {
                "comparison": comparison,
                "n_pairs": len(rows),
                "source_val_iou_delta_mean": mean([row["source_val_iou_delta"] for row in rows]),
                "source_val_iou_delta_sd": sd([row["source_val_iou_delta"] for row in rows]),
                "target_macro_iou_delta_mean": mean([row["target_macro_iou_delta"] for row in rows]),
                "target_macro_iou_delta_sd": sd([row["target_macro_iou_delta"] for row in rows]),
            }
        )
    write_csv(output_dir / "primary_paired_summary.csv", primary_paired)

    lines = [
        "# Q2 X2 Augmentation Results",
        "",
        "Primary configuration selection is based on source validation only. Target results are reported after that decision.",
        "",
        "## Per-model means",
        "",
        "| Model | Augmentation | Seeds | Source val IoU | Target 6-region Macro IoU |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in grouped:
        lines.append(
            f"| {row['model']} | {row['augmentation']} | {row['n_seeds']} | "
            f"{row['source_val_iou_mean']:.4f} +/- {row['source_val_iou_sd']:.4f} | "
            f"{row['target_macro_iou_mean']:.4f} +/- {row['target_macro_iou_sd']:.4f} |"
        )
    lines += ["", "## Paired deltas", "", "| Comparison | Pairs | Source val delta | Target macro delta |", "|---|---:|---:|---:|"]
    for row in summary_by_comparison:
        lines.append(
            f"| {row['comparison']} | {row['n_pairs']} | "
            f"{row['source_val_iou_delta_mean']:+.4f} +/- {row['source_val_iou_delta_sd']:.4f} | "
            f"{row['target_macro_iou_delta_mean']:+.4f} +/- {row['target_macro_iou_delta_sd']:.4f} |"
        )
    lines += [
        "",
        "## Prespecified selection rule",
        "",
        "`none` replaces `current` only if it is non-inferior for both architectures at the -0.010 source-validation IoU margin and the overall six-pair source delta is non-negative.",
        "",
    ]
    if len(records) == 18:
        by_model_aug = {(row["model"], row["augmentation"]): row for row in grouped}
        noninferior = True
        for model in PRIMARY_MODELS:
            current = by_model_aug.get((model, "current"))
            none = by_model_aug.get((model, "none"))
            if current is None or none is None or none["source_val_iou_mean"] - current["source_val_iou_mean"] < -0.010:
                noninferior = False
        overall = next((row for row in primary_paired if row["comparison"] == "none-current"), None)
        choose_none = bool(noninferior and overall and overall["source_val_iou_delta_mean"] >= 0)
        lines.append(f"Decision: **{'none' if choose_none else 'current'}**.")
    else:
        lines.append(f"Incomplete grid: {len(records)}/18 result files present; no final decision is emitted.")
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"records": len(records), "output_dir": str(output_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
