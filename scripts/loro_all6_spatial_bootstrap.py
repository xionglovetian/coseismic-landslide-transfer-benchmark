"""Spatial-cluster bootstrap for the all-six-region LORO extension."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

import os as _repo_os
PROJECT = Path(_repo_os.environ.get("LANDSLIDE_PROJECT_ROOT", Path(__file__).resolve().parents[1])).resolve()
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "scripts"))

import route1_spatial_cluster_bootstrap as spatial
from route1_tile_bootstrap import metrics

SIX_REGIONS = [
    "hokkaido_iburi_tobu",
    "lombok",
    "palu",
    "wenchuan",
    "longxi_river",
    "jiuzhai_valley",
]
SEEDS = [42, 2026, 777]
ENDPOINTS = ["iou", "balanced_iou", "mcc", "precision", "recall"]


def read_rows(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        yield from csv.DictReader(handle)


def load_counts(path: Path) -> dict[tuple[str, str], np.ndarray]:
    result = {}
    for row in read_rows(path):
        if float(row["threshold"]) != 0.5:
            continue
        key = (row["region"], row["id"])
        result[key] = np.asarray([float(row[k]) for k in ("tp", "fp", "fn", "tn")], dtype=float)
    return result


def build_pairs(tile_root: Path, control_regime: str, targets: list[str], seeds: list[int]):
    pairs = {}
    for seed in seeds:
        control_path = tile_root / f"e1__{control_regime}__seed{seed}.tiles.csv"
        if not control_path.exists():
            raise FileNotFoundError(control_path)
        control = load_counts(control_path)
        for target in targets:
            intervention_path = tile_root / f"q2_x3_loro_{target}_seed{seed}.tiles.csv"
            if not intervention_path.exists():
                raise FileNotFoundError(intervention_path)
            intervention = load_counts(intervention_path)
            ids = sorted(identifier for region, identifier in intervention if region == target and (region, identifier) in control)
            pairs[(target, seed)] = {
                "ids": ids,
                "a_counts": np.asarray([intervention[(target, identifier)] for identifier in ids], dtype=float),
                "b_counts": np.asarray([control[(target, identifier)] for identifier in ids], dtype=float),
            }
    return pairs


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
    parser.add_argument("--tile-root", default=str(PROJECT / "reports" / "q2_x3_loro_tiles"))
    parser.add_argument("--spatial-index", default=str(PROJECT / "data" / "processed" / "benchmark_v2_regions_512" / "target_spatial_index_all6.csv"))
    parser.add_argument("--output-dir", default=str(PROJECT / "reports" / "q2_x3_loro_bootstrap"))
    parser.add_argument("--draws", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260927)
    args = parser.parse_args()

    tile_root = Path(args.tile_root)
    spatial_index = spatial.load_spatial_index(Path(args.spatial_index))
    spatial.REGIONS = SIX_REGIONS
    spatial.SEEDS = SEEDS
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for comparison, control_regime in (("pooled-placebo", "cas-multistream"), ("pooled-single", "cas-single")):
        pairs = build_pairs(tile_root, control_regime, SIX_REGIONS, SEEDS)
        for scheme_index, scheme in enumerate(("tile", "component", "merged30")):
            for region_index, region in enumerate(["target_macro"] + SIX_REGIONS):
                rng = np.random.default_rng(args.seed + 10000 * (comparison == "pooled-single") + 100 * scheme_index + region_index)
                rows = spatial.bootstrap_contrast(pairs, region, scheme, spatial_index, rng, args.draws)
                for row in rows:
                    row["comparison"] = comparison
                all_rows.extend(rows)
    write_csv(output_dir / "loro_spatial_cluster_bootstrap.csv", all_rows)

    lines = [
        "# All-six-region LORO Spatial Bootstrap",
        "",
        f"Bootstrap draws: {args.draws}. Primary contrast: pooled LORO minus CAS multi-stream placebo.",
        "",
        "| Region | Scheme | Endpoint | Delta | 95% CI |",
        "|---|---:|---|---:|---:|",
    ]
    for row in all_rows:
        if row["comparison"] != "pooled-placebo":
            continue
        lines.append(
            f"| {row['region']} | {row['scheme']} | {row['endpoint']} | "
            f"{row['mean_delta']:+.4f} | [{row['ci95_low']:+.4f}, {row['ci95_high']:+.4f}] |"
        )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output_dir / "run_metadata.json").write_text(json.dumps({"draws": args.draws, "seed": args.seed, "tile_root": str(tile_root), "spatial_index": str(Path(args.spatial_index))}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"rows": len(all_rows), "output": str(output_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
