"""Spatial cluster bootstrap sensitivity for the E1 exposure-matched contrast.

This script reuses the frozen per-tile E1 outputs and target spatial index. It
compares three resampling schemes:

1. tile: current analysis unit.
2. component: two-stage bootstrap over connected spatial components and tiles.
3. block6/block12: two-stage bootstrap over deterministic 6x6/12x12 grid blocks
   within a spatial component and tiles within blocks.

The reported contrast is pooled-source minus CAS multi-stream replay placebo.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

import os as _repo_os
PROJECT = Path(_repo_os.environ.get("LANDSLIDE_PROJECT_ROOT", Path(__file__).resolve().parents[1])).resolve()
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "scripts"))

from route1_tile_bootstrap import collect_pair_rows, load_tiles, metrics

REGIONS = ["hokkaido_iburi_tobu", "lombok", "palu"]
SEEDS = [42, 2026, 777]
ENDPOINTS = ["iou", "balanced_iou", "mcc", "precision", "recall"]
DEFAULT_EVAL_DIR = PROJECT / "reports" / "route1_evalonly"
DEFAULT_INDEX = PROJECT / "data" / "processed" / "benchmark_v2_regions_512" / "target_spatial_index.csv"
DEFAULT_OUTPUT_DIR = PROJECT / "reports" / "route1_spatial_cluster_bootstrap"


def read_csv(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        yield from csv.DictReader(handle)


def load_spatial_index(path: Path) -> dict[str, dict[str, dict[str, int]]]:
    result: dict[str, dict[str, dict[str, int]]] = defaultdict(dict)
    for row in read_csv(path):
        result[row["region"]][row["id"]] = {
            "component_id": int(row["component_id"]),
            "x": int(row["x"]),
            "y": int(row["y"]),
        }
    return result


def block_key(meta: dict[str, int], block_size: int, min_x: int, min_y: int) -> tuple[int, int, int]:
    return (
        int(meta["component_id"]),
        (int(meta["x"]) - min_x) // block_size,
        (int(meta["y"]) - min_y) // block_size,
    )


def cluster_maps(pair, spatial_index, region: str, scheme: str) -> dict[tuple, list[int]]:
    ids = pair["ids"]
    metadata = spatial_index[region]
    missing = [identifier for identifier in ids if identifier not in metadata]
    if missing:
        raise RuntimeError(f"Missing spatial metadata for {len(missing)} tiles in {region}; examples={missing[:5]}")
    if scheme == "tile":
        return {("tile", index): [index] for index, _ in enumerate(ids)}

    assignments: dict[tuple, list[int]] = defaultdict(list)
    if scheme == "component":
        for index, identifier in enumerate(ids):
            assignments[(int(metadata[identifier]["component_id"]),)].append(index)
        return assignments

    if scheme == "merged30":
        min_size = 30
        min_x = min(meta["x"] for meta in metadata.values())
        min_y = min(meta["y"] for meta in metadata.values())
        by_component = defaultdict(list)
        for index, identifier in enumerate(ids):
            by_component[int(metadata[identifier]["component_id"])].append(index)
        merged = {}
        for component_id, members in by_component.items():
            cells = defaultdict(list)
            for index in members:
                meta = metadata[ids[index]]
                cells[block_key(meta, 6, min_x, min_y)].append(index)
            clusters = []
            for cell_members in cells.values():
                xs = [metadata[ids[index]]["x"] for index in cell_members]
                ys = [metadata[ids[index]]["y"] for index in cell_members]
                clusters.append({"members": list(cell_members), "cx": float(np.mean(xs)), "cy": float(np.mean(ys))})
            while len(clusters) > 1:
                small_index = min(range(len(clusters)), key=lambda pos: len(clusters[pos]["members"]))
                if len(clusters[small_index]["members"]) >= min_size:
                    break
                candidates = [pos for pos in range(len(clusters)) if pos != small_index]
                nearest = min(candidates, key=lambda pos: math.hypot(clusters[pos]["cx"] - clusters[small_index]["cx"], clusters[pos]["cy"] - clusters[small_index]["cy"]))
                moved_count = len(clusters[small_index]["members"])
                total = len(clusters[nearest]["members"]) + moved_count
                clusters[nearest]["cx"] = (clusters[nearest]["cx"] * len(clusters[nearest]["members"]) + clusters[small_index]["cx"] * moved_count) / total
                clusters[nearest]["cy"] = (clusters[nearest]["cy"] * len(clusters[nearest]["members"]) + clusters[small_index]["cy"] * moved_count) / total
                clusters[nearest]["members"].extend(clusters[small_index]["members"])
                del clusters[small_index]
            for pos, cluster in enumerate(clusters):
                merged[("merged30", component_id, pos)] = cluster["members"]
        return merged

    if not scheme.startswith("block"):
        raise ValueError(scheme)
    block_size = int(scheme.replace("block", ""))
    for index, identifier in enumerate(ids):
        key = block_key(metadata[identifier], block_size, min(meta["x"] for meta in metadata.values()), min(meta["y"] for meta in metadata.values()))
        assignments[key].append(index)
    return assignments


def draw_indices(pair, assignments: dict[tuple, list[int]], rng: np.random.Generator) -> np.ndarray:
    keys = list(assignments)
    sampled_keys = rng.choice(len(keys), size=len(keys), replace=True)
    draws: list[np.ndarray] = []
    for key_index in sampled_keys:
        members = np.asarray(assignments[keys[int(key_index)]], dtype=np.int64)
        sampled_members = rng.choice(members, size=len(members), replace=True)
        draws.append(sampled_members)
    return np.concatenate(draws)


def metric_delta(pair, indices: np.ndarray) -> dict[str, float]:
    a_counts = pair["a_counts"][indices].sum(axis=0)
    b_counts = pair["b_counts"][indices].sum(axis=0)
    a = metrics(a_counts)
    b = metrics(b_counts)
    return {endpoint: float(a[endpoint] - b[endpoint]) for endpoint in ENDPOINTS}


def point_delta(pair) -> dict[str, float]:
    return metric_delta(pair, np.arange(len(pair["ids"]), dtype=np.int64))


def bootstrap_contrast(pairs, region: str, scheme: str, spatial_index, rng: np.random.Generator, draws_count: int):
    maps = {
        key: cluster_maps(pair, spatial_index, key[0], scheme)
        for key, pair in pairs.items()
    }
    draws: list[dict[str, float]] = []
    for _ in range(draws_count):
        if region == "target_macro":
            sampled_regions = [REGIONS[index] for index in rng.choice(len(REGIONS), size=len(REGIONS), replace=True)]
        else:
            sampled_regions = [region]
        strata = []
        for sampled_region in sampled_regions:
            sampled_seeds = [SEEDS[index] for index in rng.choice(len(SEEDS), size=len(SEEDS), replace=True)]
            for seed in sampled_seeds:
                pair = pairs[(sampled_region, seed)]
                indices = draw_indices(pair, maps[(sampled_region, seed)], rng)
                strata.append(metric_delta(pair, indices))
        draws.append({endpoint: float(np.mean([row[endpoint] for row in strata])) for endpoint in ENDPOINTS})

    point_regions = REGIONS if region == "target_macro" else [region]
    strata = []
    for sampled_region in point_regions:
        for seed in SEEDS:
            strata.append(point_delta(pairs[(sampled_region, seed)]))
    point = {endpoint: float(np.mean([row[endpoint] for row in strata])) for endpoint in ENDPOINTS}

    rows = []
    for endpoint in ENDPOINTS:
        values = np.asarray([row[endpoint] for row in draws], dtype=float)
        rows.append(
            {
                "region": region,
                "scheme": scheme,
                "endpoint": endpoint,
                "mean_delta": point[endpoint],
                "ci95_low": float(np.percentile(values, 2.5)),
                "ci95_high": float(np.percentile(values, 97.5)),
                "probability_positive": float(np.mean(values > 0)),
                "probability_negative": float(np.mean(values < 0)),
                "n_bootstrap": draws_count,
            }
        )
    return rows


def cluster_summary(pairs, spatial_index) -> list[dict]:
    rows = []
    for region in REGIONS:
        summaries = {}
        for scheme in ("component", "block6", "block12"):
            sizes = []
            for seed in SEEDS:
                maps = cluster_maps(pairs[(region, seed)], spatial_index, region, scheme)
                sizes.extend(len(members) for members in maps.values())
            summaries[scheme] = {
                "clusters": len(sizes),
                "min": min(sizes),
                "median": float(np.median(sizes)),
                "max": max(sizes),
            }
        rows.append({"region": region, **summaries})
    return rows


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
    parser.add_argument("--eval-dir", default=str(DEFAULT_EVAL_DIR))
    parser.add_argument("--spatial-index", default=str(DEFAULT_INDEX))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--draws", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260927)
    args = parser.parse_args()

    global REGIONS
    EVAL = Path(args.eval_dir)
    # Reuse the canonical loader so the tile-level baseline is identical.
    import route1_tile_bootstrap as frozen
    frozen.EVAL = EVAL
    tiles = frozen.load_tiles()
    pairs = frozen.collect_pair_rows(tiles, "E1", "pooled", "cas-multistream")
    spatial_index = load_spatial_index(Path(args.spatial_index))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for scheme_index, scheme in enumerate(("tile", "component", "block6", "block12")):
        for region_index, region in enumerate(["target_macro"] + REGIONS):
            rng = np.random.default_rng(args.seed + 100 * scheme_index + region_index)
            all_rows.extend(bootstrap_contrast(pairs, region, scheme, spatial_index, rng, args.draws))
    write_csv(output_dir / "spatial_cluster_bootstrap.csv", all_rows)
    summary = cluster_summary(pairs, spatial_index)
    write_csv(output_dir / "cluster_size_summary.csv", summary)
    (output_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                "eval_dir": str(EVAL),
                "spatial_index": str(Path(args.spatial_index)),
                "draws": args.draws,
                "seed": args.seed,
                "regions": REGIONS,
                "seeds": SEEDS,
                "endpoints": ENDPOINTS,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"rows": len(all_rows), "output": str(output_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
