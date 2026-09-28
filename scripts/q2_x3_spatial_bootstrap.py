"""Spatial-cluster bootstrap for the two-architecture X3 exposure-matched contrast."""

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

REGIONS = ["hokkaido_iburi_tobu", "lombok", "palu"]
SEEDS = [42, 2026, 777]
ARCHITECTURES = {"resunet": "ResUNet", "bottleneckliteask": "Bottleneck-LiteASK"}


def read_counts(path: Path) -> dict[tuple[str, str], np.ndarray]:
    counts = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if float(row["threshold"]) != 0.5:
                continue
            key = (row["region"], row["id"])
            counts[key] = np.asarray([float(row[k]) for k in ("tp", "fp", "fn", "tn")], dtype=float)
    return counts


def build_pairs(tile_root: Path, slug: str):
    pairs = {}
    for seed in SEEDS:
        control_path = tile_root / f"q2_x3_{slug}_cas-multistream_seed{seed}.tiles.csv"
        if not control_path.exists():
            raise FileNotFoundError(control_path)
        control = read_counts(control_path)
        for region in REGIONS:
            intervention_path = tile_root / f"q2_x3_{slug}_pooled_seed{seed}.tiles.csv"
            if not intervention_path.exists():
                raise FileNotFoundError(intervention_path)
            intervention = read_counts(intervention_path)
            ids = sorted(identifier for reg, identifier in intervention if reg == region and (region, identifier) in control)
            pairs[(region, seed)] = {
                "ids": ids,
                "a_counts": np.asarray([intervention[(region, identifier)] for identifier in ids], dtype=float),
                "b_counts": np.asarray([control[(region, identifier)] for identifier in ids], dtype=float),
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
    parser.add_argument("--tile-root", default=str(PROJECT / "reports" / "q2_x3_tiles"))
    parser.add_argument("--spatial-index", default=str(PROJECT / "data" / "processed" / "benchmark_v2_regions_512" / "target_spatial_index.csv"))
    parser.add_argument("--output-dir", default=str(PROJECT / "reports" / "q2_x3_bootstrap"))
    parser.add_argument("--draws", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260927)
    args = parser.parse_args()

    tile_root = Path(args.tile_root)
    spatial_index = spatial.load_spatial_index(Path(args.spatial_index))
    spatial.REGIONS = REGIONS
    spatial.SEEDS = SEEDS
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for arch_index, (slug, label) in enumerate(ARCHITECTURES.items()):
        pairs = build_pairs(tile_root, slug)
        for scheme_index, scheme in enumerate(("tile", "component", "merged30")):
            for region_index, region in enumerate(["target_macro"] + REGIONS):
                rng = np.random.default_rng(args.seed + 10000 * arch_index + 100 * scheme_index + region_index)
                rows = spatial.bootstrap_contrast(pairs, region, scheme, spatial_index, rng, args.draws)
                for row in rows:
                    row["architecture"] = label
                all_rows.extend(rows)
    write_csv(output_dir / "x3_spatial_cluster_bootstrap.csv", all_rows)

    lines = [
        "# Two-architecture X3 Spatial Bootstrap",
        "",
        f"Bootstrap draws: {args.draws}. Contrast: pooled-source minus CAS multi-stream placebo.",
        "",
        "| Architecture | Region | Scheme | Endpoint | Delta | 95% CI |",
        "|---|---:|---:|---|---:|---:|",
    ]
    for row in all_rows:
        lines.append(
            f"| {row['architecture']} | {row['region']} | {row['scheme']} | {row['endpoint']} | "
            f"{row['mean_delta']:+.4f} | [{row['ci95_low']:+.4f}, {row['ci95_high']:+.4f}] |"
        )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output_dir / "run_metadata.json").write_text(json.dumps({"draws": args.draws, "seed": args.seed, "tile_root": str(tile_root), "spatial_index": str(Path(args.spatial_index))}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"rows": len(all_rows), "output": str(output_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
