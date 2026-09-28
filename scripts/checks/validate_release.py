"""Validate the public benchmark package without requiring GPU dependencies."""

from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "data_manifests" / "manifest_benchmark_v2.csv"
EXPECTED_REGIONS = {
    "hokkaido_iburi_tobu": 1484,
    "jiuzhai_valley": 5925,
    "lombok": 436,
    "longxi_river": 2504,
    "moxitaidi": 980,
    "palu": 817,
    "wenchuan": 178,
}
FORBIDDEN_SUFFIXES = {
    ".bmp",
    ".ckpt",
    ".jpeg",
    ".jpg",
    ".pth",
    ".pt",
    ".tif",
    ".tiff",
    ".png",
}
REQUIRED = [
    "README.md",
    "REPRODUCIBILITY.md",
    "PROVENANCE.md",
    "FULL_REPRODUCTION.md",
    "INPUTS.md",
    "DATA_LICENSES.md",
    "environment-windows.yml",
    "CHECKPOINT_INDEX.csv",
    "configs/RUN_INDEX.csv",
    "scripts/setup/download_inputs.ps1",
    "scripts/setup/prepare_primary_data.ps1",
    "scripts/checks/check_inputs.py",
    "scripts/checks/compare_checkpoint_tensors.py",
    "requirements.txt",
    "src/models.py",
    "src/datasets.py",
    "src/metrics.py",
    "scripts/train_unified.py",
    "scripts/train_exposure_matched.py",
    "scripts/evaluate_q2_checkpoint.py",
    "scripts/q2_x3_spatial_bootstrap.py",
    "scripts/q2_x3_seed_stratified_reanalysis.py",
    "metrics/q2_x3_seed_stratified_bootstrap/q2_x3_seed_stratified_t_interval.csv",
    "metrics/figure_source_data/README.md",
    "metrics/figure_source_data/fig4_exposure_matched_effects.csv",
    "metrics/route1_bootstrap/README.md",
    "metrics/q2_x2_augmentation/primary_paired_summary.csv",
    "metrics/q2_x1_summary/grouped.csv",
    "metrics/q2_x3_bootstrap/x3_spatial_cluster_bootstrap.csv",
    "reports/selected/Q2_final_manuscript_revision_summary_20260928.md",
]


def close(a: float, b: float, tolerance: float = 1e-9) -> bool:
    return abs(a - b) <= tolerance


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    errors: list[str] = []
    for relative in REQUIRED:
        if not (ROOT / relative).is_file():
            errors.append(f"missing required file: {relative}")

    if not MANIFEST.is_file():
        errors.append(f"missing manifest: {MANIFEST}")
    else:
        rows = read_csv(MANIFEST)
        counts = Counter(row["region"] for row in rows)
        if len(rows) != sum(EXPECTED_REGIONS.values()):
            errors.append(f"manifest row count is {len(rows)}, expected {sum(EXPECTED_REGIONS.values())}")
        for region, expected in EXPECTED_REGIONS.items():
            if counts[region] != expected:
                errors.append(f"{region}: {counts[region]} rows, expected {expected}")

    forbidden = [
        path.relative_to(ROOT).as_posix()
        for path in ROOT.rglob("*")
        if path.is_file() and ".git" not in path.parts and path.suffix.lower() in FORBIDDEN_SUFFIXES
    ]
    if forbidden:
        errors.extend(f"forbidden raw/binary artifact: {path}" for path in forbidden[:20])
        if len(forbidden) > 20:
            errors.append(f"... and {len(forbidden) - 20} more forbidden artifacts")

    x2_path = ROOT / "metrics/q2_x2_augmentation/primary_paired_summary.csv"
    if x2_path.is_file():
        row = next(item for item in read_csv(x2_path) if item["comparison"] == "none-current")
        if not close(float(row["source_val_iou_delta_mean"]), 0.04315053338028054):
            errors.append("X2 source-validation anchor mismatch")

    x1_path = ROOT / "metrics/q2_x1_summary/grouped.csv"
    if x1_path.is_file():
        rows = {row["model"]: row for row in read_csv(x1_path)}
        expected = {
            "ResUNet": -0.024314342604730504,
            "SegFormer-B0": 0.002866039792453312,
        }
        for model, value in expected.items():
            if model not in rows or not close(float(rows[model]["delta_256_minus_128_mean"]), value):
                errors.append(f"X1 anchor mismatch for {model}")

    x3_path = ROOT / "metrics/q2_x3_bootstrap/x3_spatial_cluster_bootstrap.csv"
    if x3_path.is_file():
        rows = read_csv(x3_path)
        expected = {
            "ResUNet": 0.0772718653904376,
            "Bottleneck-LiteASK": 0.045587568996210436,
        }
        for architecture, value in expected.items():
            match = next(
                (
                    row
                    for row in rows
                    if row["architecture"] == architecture
                    and row["region"] == "target_macro"
                    and row["scheme"] == "component"
                    and row["endpoint"] == "iou"
                ),
                None,
            )
            if match is None or not close(float(match["mean_delta"]), value):
                errors.append(f"X3 component-bootstrap anchor mismatch for {architecture}")

    seed_path = ROOT / "metrics/q2_x3_seed_stratified_bootstrap/q2_x3_seed_stratified_t_interval.csv"
    if seed_path.is_file():
        rows = read_csv(seed_path)
        expected = {
            ("ResUNet", "iou"): (0.0772718653904376, 0.051415226404182, 0.103128504376693),
            ("Bottleneck-LiteASK", "iou"): (0.0455875689962104, -0.005622849379496, 0.096797987371917),
            ("Bottleneck-LiteASK", "mcc"): (0.0989679028499328, 0.023974142002776, 0.17396166369709),
        }
        for (architecture, endpoint), (mean, low, high) in expected.items():
            row = next((item for item in rows if item["architecture"] == architecture and item["endpoint"] == endpoint), None)
            if row is None or not close(float(row["mean_delta"]), mean) or not close(float(row["ci95_low"]), low) or not close(float(row["ci95_high"]), high):
                errors.append(f"seed-stratified anchor mismatch for {architecture} {endpoint}")

    if errors:
        for error in errors:
            print(f"FAIL: {error}")
        return 1
    print("PASS: manifest counts, required files, artifact exclusions and metric anchors validated")
    return 0


if __name__ == "__main__":
    sys.exit(main())
