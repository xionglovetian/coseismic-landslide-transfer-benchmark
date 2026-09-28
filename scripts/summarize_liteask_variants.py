import csv
import json
from pathlib import Path

import os as _repo_os
PROJECT = Path(_repo_os.environ.get("LANDSLIDE_PROJECT_ROOT", Path(__file__).resolve().parents[1])).resolve()
REPORTS = PROJECT / "reports"
OUTPUTS = PROJECT / "outputs"

variants = [
    ("LiteASK-UNet++", "group_liteaskunetpp"),
    ("DeepLiteASK-UNet++", "group_deepliteaskunetpp"),
    ("Bottleneck-LiteASK", "group_bottleneckliteaskunetpp"),
]
with (REPORTS / "model_complexity.csv").open(encoding="utf-8-sig", newline="") as handle:
    complexity = {row["model"]: row for row in csv.DictReader(handle)}
with (REPORTS / "group_model_seed_summary.csv").open(encoding="utf-8-sig", newline="") as handle:
    main_summary = list(csv.DictReader(handle))

rows = []
for label, slug in variants:
    metrics = json.loads((OUTPUTS / f"{slug}_seed42" / "metrics.json").read_text(encoding="utf-8"))
    comp = complexity[label]
    rows.append({
        "model": label,
        "seed42_cas_iou": metrics["cas_val"]["iou"],
        "seed42_c3_iou": metrics["resunet_bfa_external_test"]["iou"],
        "seed42_c3_dice": metrics["resunet_bfa_external_test"]["dice"],
        "params_million": float(comp["params_million"]),
        "gflops": float(comp["gflops_b1_128"]),
        "fps_b32": float(comp["fps_b32_bf16"]),
    })

with (REPORTS / "liteask_variant_screening.csv").open("w", newline="", encoding="utf-8-sig") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

candidate_cas = next(row for row in main_summary if row["model"] == "Bottleneck-LiteASK" and row["split"] == "cas_val" and row["metric"] == "iou")
candidate_ext = next(row for row in main_summary if row["model"] == "Bottleneck-LiteASK" and row["split"] == "external" and row["metric"] == "iou")
unetpp_cas = next(row for row in main_summary if row["model"] == "U-Net++" and row["split"] == "cas_val" and row["metric"] == "iou")
unetpp_ext = next(row for row in main_summary if row["model"] == "U-Net++" and row["split"] == "external" and row["metric"] == "iou")

lines = [
    "# LiteASK Variant Screening",
    "",
    "Date: 2026-09-21",
    "",
    "## Seed 42 Screening",
    "",
    "| Model | CAS IoU | c3 IoU | c3 Dice | Params (M) | GFLOPs | FPS b32 |",
    "|---|---:|---:|---:|---:|---:|---:|",
]
for row in rows:
    lines.append(
        f'| {row["model"]} | {row["seed42_cas_iou"]:.4f} | {row["seed42_c3_iou"]:.4f} | '
        f'{row["seed42_c3_dice"]:.4f} | {row["params_million"]:.3f} | {row["gflops"]:.3f} | {row["fps_b32"]:.1f} |'
    )
lines += [
    "",
    "## Three-Seed Candidate Result",
    "",
    "| Model | CAS IoU | c3 IoU |",
    "|---|---:|---:|",
    f'| Bottleneck-LiteASK | {float(candidate_cas["mean"]):.4f} ± {float(candidate_cas["std"]):.4f} | {float(candidate_ext["mean"]):.4f} ± {float(candidate_ext["std"]):.4f} |',
    f'| U-Net++ | {float(unetpp_cas["mean"]):.4f} ± {float(unetpp_cas["std"]):.4f} | {float(unetpp_ext["mean"]):.4f} ± {float(unetpp_ext["std"]):.4f} |',
    "",
    "## Decision",
    "",
    "- Full-depth and deep-only LiteSK improve CAS but do not preserve the c3 advantage of heavy ASK-UNet++.",
    "- Bottleneck LiteSK gives the best c3 tradeoff among the lightweight variants while keeping CAS strong.",
    "- Bottleneck-LiteASK adds only about 0.6M parameters over U-Net++ and keeps nearly identical GFLOPs and throughput.",
    "- Bottleneck-LiteASK was therefore promoted to a five-seed robustness comparison; full-depth and deep-only variants remain one-seed screening evidence only.",
    "",
    "## Files",
    "",
    "- Screening table: D:\\landslide_unet_project\\reports\\liteask_variant_screening.csv",
    "- Main comparison: D:\\landslide_unet_project\\reports\\group_extended_experiment_report.md",
    "- Integrated assessment: D:\\landslide_unet_project\\reports\\combined_model_assessment.md",
    "- Five-seed robustness: D:\\landslide_unet_project\\reports\\five_seed_key_models_report.md",
]
(REPORTS / "liteask_ablation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("wrote", REPORTS / "liteask_ablation_report.md")
