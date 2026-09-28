"""Compare 128x128 and 256x256 zero-shot benchmark results for Gate A."""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

import os as _repo_os
PROJECT = Path(_repo_os.environ.get("LANDSLIDE_PROJECT_ROOT", Path(__file__).resolve().parents[1])).resolve()
REPORTS = PROJECT / "reports"
RAW_128 = REPORTS / "benchmark_v2_zero_shot_raw"
RAW_256 = REPORTS / "benchmark_v2_256_pilot_raw"
OUT_PREFIX = "benchmark_v2_resolution_gate_a"

REGION_ORDER = [
    "wenchuan",
    "jiuzhai_valley",
    "moxitaidi",
    "longxi_river",
    "hokkaido_iburi_tobu",
    "lombok",
    "palu",
]
REGION_LABELS = {
    "wenchuan": "Wenchuan",
    "jiuzhai_valley": "Jiuzhai Valley",
    "moxitaidi": "Moxitaidi",
    "longxi_river": "Longxi River",
    "hokkaido_iburi_tobu": "Hokkaido",
    "lombok": "Lombok",
    "palu": "Palu",
}
TARGET_REGIONS = ["hokkaido_iburi_tobu", "lombok", "palu"]
OTHER_REGIONS = [region for region in REGION_ORDER if region not in TARGET_REGIONS]
FILE_PREFIX_128 = {
    "ResUNet": "bench_v2_resunet",
    "DeepLabV3+": "bench_v2_deeplabv3plus",
    "SegFormer-B0": "bench_v2_segformerb0",
    "Bottleneck-LiteASK": "group_bottleneckliteaskunetpp",
}
FILE_PREFIX_256 = {
    "ResUNet": "bench_v2_256_pilot_resunet",
    "DeepLabV3+": "bench_v2_256_pilot_deeplabv3plus",
    "SegFormer-B0": "bench_v2_256_pilot_segformerb0",
    "Bottleneck-LiteASK": "bench_v2_256_pilot_bottleneckliteask",
}
MODELS = list(FILE_PREFIX_128)


def mean(values):
    return statistics.fmean(float(value) for value in values)


def read_resolution(resolution: int):
    raw_dir = RAW_128 if resolution == 128 else RAW_256
    prefixes = FILE_PREFIX_128 if resolution == 128 else FILE_PREFIX_256
    records = []
    for model, prefix in prefixes.items():
        path = raw_dir / f"{prefix}_seed42.json"
        if not path.exists():
            raise FileNotFoundError(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        if int(data["seed"]) != 42:
            raise ValueError(f"Expected seed 42 in {path}")
        if int(data["input_size"]) != resolution:
            raise ValueError(f"Expected input_size={resolution} in {path}")
        for region in REGION_ORDER:
            group = data["groups"][region]
            threshold = group["threshold_metrics"]["0.5"]
            records.append(
                {
                    "model": model,
                    "resolution": resolution,
                    "region": region,
                    "region_label": REGION_LABELS[region],
                    "iou": float(threshold["iou"]),
                    "dice": float(threshold["dice"]),
                    "bf1_2": float(group["f1_2_mean"]),
                    "bf1_4": float(group["f1_4_mean"]),
                    "hd95": float(group["hd95_mean"]),
                }
            )
    return records


def write_csv(path: Path, rows, fields) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def fmt(value, digits=4):
    return f"{float(value):.{digits}f}"


def main() -> None:
    rows = read_resolution(128) + read_resolution(256)
    by_model_region = {
        (row["model"], row["region"], row["resolution"]): row for row in rows
    }

    per_model_region = []
    for model in MODELS:
        for region in REGION_ORDER:
            row_128 = by_model_region[(model, region, 128)]
            row_256 = by_model_region[(model, region, 256)]
            record = {
                "model": model,
                "region": region,
                "region_label": REGION_LABELS[region],
                "target_region": region in TARGET_REGIONS,
            }
            for metric in ["iou", "dice", "bf1_2", "bf1_4", "hd95"]:
                record[f"{metric}_128"] = row_128[metric]
                record[f"{metric}_256"] = row_256[metric]
                record[f"delta_{metric}"] = row_256[metric] - row_128[metric]
            per_model_region.append(record)

    aggregate_regions = []
    for region in REGION_ORDER:
        entries_128 = [row for row in rows if row["resolution"] == 128 and row["region"] == region]
        entries_256 = [row for row in rows if row["resolution"] == 256 and row["region"] == region]
        record = {
            "region": region,
            "region_label": REGION_LABELS[region],
            "target_region": region in TARGET_REGIONS,
            "n_models": len(entries_128),
        }
        for metric in ["iou", "dice", "bf1_2", "bf1_4", "hd95"]:
            value_128 = mean(row[metric] for row in entries_128)
            value_256 = mean(row[metric] for row in entries_256)
            record[f"{metric}_128"] = value_128
            record[f"{metric}_256"] = value_256
            record[f"delta_{metric}"] = value_256 - value_128
        # Boundary tolerances are expressed in input pixels. Normalize both
        # resolutions to a 128-grid equivalent: 2 px at 128 = 4 px at 256;
        # HD95 is scaled to the 256 grid by multiplying the 128 value by 2.
        record["bf1_2_norm_128"] = record["bf1_2_128"]
        record["bf1_2_norm_256"] = record["bf1_4_256"]
        record["delta_bf1_2_norm"] = record["bf1_2_norm_256"] - record["bf1_2_norm_128"]
        record["hd95_norm_128"] = record["hd95_128"] * 2.0
        record["hd95_norm_256"] = record["hd95_256"]
        record["delta_hd95_norm"] = record["hd95_norm_256"] - record["hd95_norm_128"]
        aggregate_regions.append(record)

    per_model_macro = []
    for model in MODELS:
        record = {"model": model, "n_regions": len(REGION_ORDER)}
        for metric in ["iou", "dice", "bf1_2", "bf1_4", "hd95"]:
            value_128 = mean(row[metric] for row in rows if row["model"] == model and row["resolution"] == 128)
            value_256 = mean(row[metric] for row in rows if row["model"] == model and row["resolution"] == 256)
            record[f"{metric}_128"] = value_128
            record[f"{metric}_256"] = value_256
            record[f"delta_{metric}"] = value_256 - value_128
        per_model_macro.append(record)

    region_delta = {row["region"]: row for row in aggregate_regions}
    target_iou_deltas = {region: region_delta[region]["delta_iou"] for region in TARGET_REGIONS}
    improving_targets = [region for region, delta in target_iou_deltas.items() if delta > 0]
    best_target = max(TARGET_REGIONS, key=lambda region: target_iou_deltas[region])
    macro_iou_128 = mean(row["iou"] for row in rows if row["resolution"] == 128)
    macro_iou_256 = mean(row["iou"] for row in rows if row["resolution"] == 256)
    macro_iou_delta = macro_iou_256 - macro_iou_128
    min_other_delta = min(region_delta[region]["delta_iou"] for region in OTHER_REGIONS)

    target_bf1_deltas = {region: region_delta[region]["delta_bf1_2_norm"] for region in TARGET_REGIONS}
    target_hd95_deltas = {region: region_delta[region]["delta_hd95_norm"] for region in TARGET_REGIONS}
    bf1_improved = sum(delta > 0 for delta in target_bf1_deltas.values())
    hd95_nonworse = sum(delta <= 0 for delta in target_hd95_deltas.values())
    boundary_pass = (
        bf1_improved >= 2
        and mean(target_bf1_deltas.values()) >= -0.005
        and hd95_nonworse >= 2
        and mean(target_hd95_deltas.values()) <= 2.0
    )

    checks = {
        "至少两个困难区域 IoU 改善": len(improving_targets) >= 2,
        "至少一个困难区域 IoU 绝对提升 >= 0.01": target_iou_deltas[best_target] >= 0.01,
        "七区域 Macro IoU 不下降": macro_iou_delta >= 0.0,
        "其他区域 IoU 无 >0.02 崩溃": min_other_delta > -0.02,
        "困难区域 BF1/HD95 无系统性恶化": boundary_pass,
    }
    gate_a_pass = all(checks.values())

    write_csv(
        REPORTS / f"{OUT_PREFIX}_region.csv",
        aggregate_regions,
        list(aggregate_regions[0].keys()),
    )
    write_csv(
        REPORTS / f"{OUT_PREFIX}_per_model_region.csv",
        per_model_region,
        list(per_model_region[0].keys()),
    )
    write_csv(
        REPORTS / f"{OUT_PREFIX}_per_model_macro.csv",
        per_model_macro,
        list(per_model_macro[0].keys()),
    )

    lines = [
        "# Benchmark v2: 128 vs 256 Resolution Gate A",
        "",
        "Protocol: seed 42, four pilot architectures, identical seven-region manifest and threshold 0.5. Region values are means across models; Macro IoU is the unweighted mean over seven regional means.",
        "",
        f"**Decision: Gate A {'PASS' if gate_a_pass else 'FAIL'}**",
        "",
        f"**Action:** {'Continue Gate B for 256.' if gate_a_pass else 'Stop 256 full training. Keep 128 as the main experiment and 256 only as a resolution-sensitivity analysis; proceed to 128 few-shot adaptation.'}",
        "",
        "This seed-42 four-model pilot is a go/no-go decision check, not a replacement for the existing three-seed 128 benchmark.",
        "",
        "## Gate Checks",
        "",
        "| Check | Result | Evidence |",
        "|---|---:|---|",
    ]
    evidence = {
        "至少两个困难区域 IoU 改善": f"{len(improving_targets)}/3 improved: {', '.join(REGION_LABELS[r] for r in improving_targets) or 'none'}",
        "至少一个困难区域 IoU 绝对提升 >= 0.01": f"best={REGION_LABELS[best_target]} {target_iou_deltas[best_target]:+.4f}",
        "七区域 Macro IoU 不下降": f"{macro_iou_256:.4f} - {macro_iou_128:.4f} = {macro_iou_delta:+.4f}",
        "其他区域 IoU 无 >0.02 崩溃": f"worst={REGION_LABELS[min(OTHER_REGIONS, key=lambda r: region_delta[r]['delta_iou'])]} {min_other_delta:+.4f}",
        "困难区域 BF1/HD95 无系统性恶化": f"normalized BF1@2 improved {bf1_improved}/3, mean delta {mean(target_bf1_deltas.values()):+.4f}; normalized HD95 non-worse {hd95_nonworse}/3, mean delta {mean(target_hd95_deltas.values()):+.2f}",
    }
    for check, passed in checks.items():
        lines.append(f"| {check} | {'PASS' if passed else 'FAIL'} | {evidence[check]} |")

    lines += [
        "",
        "## Regional Aggregate",
        "",
        "| Region | IoU 128 | IoU 256 | Delta IoU | BF1@2* 128 | BF1@2* 256 | Delta BF1@2* | HD95* 128 | HD95* 256 | Delta HD95* |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate_regions:
        lines.append(
            f"| {row['region_label']}{' *' if row['target_region'] else ''} | {fmt(row['iou_128'])} | {fmt(row['iou_256'])} | "
            f"{row['delta_iou']:+.4f} | {fmt(row['bf1_2_norm_128'])} | {fmt(row['bf1_2_norm_256'])} | {row['delta_bf1_2_norm']:+.4f} | "
            f"{fmt(row['hd95_norm_128'], 2)} | {fmt(row['hd95_norm_256'], 2)} | {row['delta_hd95_norm']:+.2f} |"
        )

    lines += [
        "",
        "* Starred regions are the Gate A target regions Hokkaido, Lombok, and Palu.",
        "* Boundary columns are resolution-normalized (denoted by an asterisk). BF1@2 compares native 128 px tolerance 2 with 256 px tolerance 4; HD95 rescales the 128-grid value by 2 to the 256 grid. Raw BF1/HD95 are retained in the CSV files but are not directly comparable across input resolutions.",
        "",
        "## Per-Model Macro",
        "",
        "| Model | Macro IoU 128 | Macro IoU 256 | Delta IoU | Delta BF1@2 (raw) | Delta HD95 (raw) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in per_model_macro:
        lines.append(
            f"| {row['model']} | {fmt(row['iou_128'])} | {fmt(row['iou_256'])} | {row['delta_iou']:+.4f} | "
            f"{row['delta_bf1_2']:+.4f} | {row['delta_hd95']:+.2f} |"
        )

    lines += [
        "",
        "## Operational Notes",
        "",
        "- `Other-region collapse` is operationalized as an aggregate IoU loss greater than 0.02; it is a guardrail, not a claim threshold.",
        "- `No systematic BF1/HD95 degradation` is evaluated on resolution-normalized boundary metrics. It requires at least two of three target regions to improve in normalized BF1@2 and be non-worse in normalized HD95, while target means remain within BF1 -0.005 and HD95 +2 normalized pixels.",
        "- Per-model and per-region source values, including raw and normalized boundary metrics, are retained in the companion CSV files for audit.",
    ]
    (REPORTS / f"{OUT_PREFIX}_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Gate A:", "PASS" if gate_a_pass else "FAIL")
    print("checks:", checks)
    print("target IoU deltas:", target_iou_deltas)
    print("macro IoU delta:", macro_iou_delta)


if __name__ == "__main__":
    main()
