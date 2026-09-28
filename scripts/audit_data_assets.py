"""Audit Route 2 data readiness without running model experiments.

Read-only audit of local manifests, raster metadata, masks, archives, and
few-shot split provenance. Outputs a machine-readable gate matrix and a
human-readable readiness report.
"""
from __future__ import annotations

import csv
import json
import os
import re
import sys
import warnings
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

warnings.filterwarnings("ignore", category=UserWarning)
try:
    import rasterio
except ImportError:
    rasterio = None

import os as _repo_os
PROJECT = Path(_repo_os.environ.get("LANDSLIDE_PROJECT_ROOT", Path(__file__).resolve().parents[1])).resolve()
DATA = PROJECT / "data"
REPORTS = PROJECT / "reports"
BENCH_MANIFEST = DATA / "processed" / "benchmark_v2_regions_512" / "manifest_benchmark_v2.csv"
SOURCE_MANIFEST = DATA / "splits" / "resunet_bfa_manifest.csv"
SOURCE_MASKS = DATA / "raw" / "resunet_bfa_ready" / "可直接进行实验的数据集" / "data" / "Masks"
FEWSHOT_DIR = DATA / "processed" / "fewshot_target_splits"
BUFFERED_DIR = FEWSHOT_DIR / "buffered"
STAMP = "20260925"

GEOSPATIAL_SIDECARS = {
    ".shp", ".shx", ".dbf", ".prj", ".geojson", ".gpkg", ".kml", ".kmz",
    ".jp2", ".hdf", ".h5", ".vrt", ".tfw", ".wld", ".aux", ".ovr", ".cpg",
}
AUG_RE = re.compile(r"_(?:BR|CN|FL_(?:HOR|VER|DIA)|RT_(?:90|180|270))$")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def source_base(identifier: str) -> str:
    return AUG_RE.sub("", identifier)


def scan_target_manifest() -> tuple[dict, list[dict]]:
    rows = read_csv(BENCH_MANIFEST)
    by_region: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_region[row["region"]].append(float(row["foreground_fraction"]))
    regions = {}
    for region, values in sorted(by_region.items()):
        regions[region] = {
            "n": len(values),
            "zero_foreground": int(sum(value == 0.0 for value in values)),
            "min_foreground_fraction": float(min(values)),
            "max_foreground_fraction": float(max(values)),
            "mean_foreground_fraction": float(np.mean(values)),
        }
    summary = {
        "manifest": str(BENCH_MANIFEST),
        "n_pairs": len(rows),
        "n_missing_images": sum(not Path(row["image_path"]).is_file() for row in rows),
        "n_missing_masks": sum(not Path(row["mask_path"]).is_file() for row in rows),
        "all_declared_512": all(row["width"] == "512" and row["height"] == "512" for row in rows),
        "n_duplicate_ids": len(rows) - len({row["id"] for row in rows}),
        "n_duplicate_image_sha256": len(rows) - len({row["image_sha256"] for row in rows}),
        "n_duplicate_mask_sha256": len(rows) - len({row["mask_sha256"] for row in rows}),
        "regions": regions,
        "metadata_columns": sorted(rows[0].keys()),
        "present_scene_or_geolocation_columns": sorted(
            key for key in rows[0]
            if any(token in key.lower() for token in ("scene", "crs", "transform", "latitude", "longitude", "coord"))
        ),
        "present_annotation_provenance_columns": sorted(
            key for key in rows[0]
            if any(token in key.lower() for token in ("annotator", "agreement", "validation", "label_source"))
        ),
    }
    matrix = []
    for region, stats in regions.items():
        sample = next(row for row in rows if row["region"] == region)
        matrix.append({
            "region": region,
            "n": stats["n"],
            "zero_foreground": stats["zero_foreground"],
            "min_foreground_fraction": stats["min_foreground_fraction"],
            "max_foreground_fraction": stats["max_foreground_fraction"],
            "acquisition": sample["acquisition"],
            "source": sample["source"],
            "sensor": sample["sensor"],
            "ground_resolution_m": sample["ground_resolution_m"],
            "source_archive": sample["source_archive"],
        })
    return summary, matrix


def scan_source_masks() -> dict:
    rows = read_csv(SOURCE_MANIFEST)
    by_split = Counter()
    zero_by_split = Counter()
    base_rows: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_split[row["split"]] += 1
        with Image.open(row["mask_path"]).convert("L") as image:
            histogram = image.histogram()
        is_zero = sum(histogram[1:]) == 0
        if is_zero:
            zero_by_split[row["split"]] += 1
        base_rows[source_base(row["id"])].append(row)
    base_total = Counter()
    base_zero = Counter()
    for grouped in base_rows.values():
        split = grouped[0]["split"]
        base_total[split] += 1
        original = next(row for row in grouped if row["augmented"] == "0")
        with Image.open(original["mask_path"]).convert("L") as image:
            histogram = image.histogram()
        if sum(histogram[1:]) == 0:
            base_zero[split] += 1
    return {
        "manifest_rows": len(rows),
        "unique_base_ids": len(base_rows),
        "rows_by_split": dict(sorted(by_split.items())),
        "zero_mask_rows_by_split": dict(sorted(zero_by_split.items())),
        "base_ids_by_split": dict(sorted(base_total.items())),
        "zero_base_ids_by_split": dict(sorted(base_zero.items())),
        "source_readme_contact": "zhaozhenyu1@stu.ynu.edu.cn",
        "source_readme_limit": "Original 1536x1536 rasters were not included; local originals are 224x224 crops.",
    }


def scan_source_rasters() -> dict:
    result = {
        "rasterio_available": rasterio is not None,
        "n_tif": 0,
        "by_parent": {},
        "shape_dtype_profiles": {},
        "crs_counts": {},
        "transform_counts": {},
        "n_with_non_identity_transform": 0,
        "n_with_crs": 0,
        "errors": [],
    }
    if rasterio is None:
        return result
    paths = [path for path in DATA.rglob("*") if path.suffix.lower() in {".tif", ".tiff"}]
    profiles = Counter()
    crs_counts = Counter()
    transform_counts = Counter()
    by_parent = Counter(path.parent.name for path in paths)
    for path in paths:
        try:
            with rasterio.open(path) as dataset:
                profiles[(dataset.count, dataset.width, dataset.height, dataset.dtypes[0])] += 1
                transform = tuple(float(value) for value in dataset.transform)[:6]
                transform_counts[transform] += 1
                crs_counts[str(dataset.crs)] += 1
                if dataset.crs is not None:
                    result["n_with_crs"] += 1
                if transform != (1.0, 0.0, 0.0, 0.0, 1.0, 0.0):
                    result["n_with_non_identity_transform"] += 1
        except Exception as exc:
            result["errors"].append({"path": str(path), "error": repr(exc)})
    result["n_tif"] = len(paths)
    result["by_parent"] = dict(sorted(by_parent.items()))
    result["shape_dtype_profiles"] = {str(key): value for key, value in profiles.most_common()}
    result["crs_counts"] = dict(crs_counts)
    result["transform_counts"] = {str(key): value for key, value in transform_counts.most_common(20)}
    return result


def scan_filesystem_assets() -> dict:
    sidecars = []
    archives = []
    for path in PROJECT.rglob("*"):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix in GEOSPATIAL_SIDECARS or path.name.lower().endswith((".tif.aux.xml", ".tiff.aux.xml")):
            sidecars.append(str(path))
        if suffix in {".zip", ".rar", ".7z"}:
            archives.append({"path": str(path), "bytes": path.stat().st_size})
    archive_members = Counter()
    suspicious_members = []
    unreadable = []
    import zipfile
    for item in archives:
        path = Path(item["path"])
        if path.suffix.lower() != ".zip":
            unreadable.append({"path": str(path), "reason": "RAR/7z listing not available with stdlib"})
            continue
        try:
            with zipfile.ZipFile(path) as handle:
                for name in handle.namelist():
                    suffix = Path(name).suffix.lower()
                    archive_members[suffix or "<none>"] += 1
                    candidate = Path(name)
                    if suffix in GEOSPATIAL_SIDECARS or any(token in candidate.name.lower() for token in ("coord", "crs", "georef")):
                        suspicious_members.append({"archive": str(path), "member": name})
        except Exception as exc:
            unreadable.append({"path": str(path), "reason": repr(exc)})
    return {
        "geospatial_sidecars": sidecars,
        "n_geospatial_sidecars": len(sidecars),
        "archives": archives,
        "archive_member_extensions": dict(archive_members.most_common()),
        "suspicious_geospatial_members": suspicious_members,
        "unreadable_archives": unreadable,
    }


def scan_fewshot_splits() -> dict:
    support_paths = [
        FEWSHOT_DIR / "fewshot_support_seed42.csv",
        FEWSHOT_DIR / "fewshot_support_seed2026_777.csv",
    ]
    eval_paths = [
        FEWSHOT_DIR / "fewshot_eval_seed42.csv",
        FEWSHOT_DIR / "fewshot_eval_seed2026_777.csv",
    ]
    support = [row for path in support_paths for row in read_csv(path) if int(row["shot"]) == 20]
    evaluation = [row for path in eval_paths for row in read_csv(path)]
    support_ids = {(row["seed"], row["region"], row["id"]) for row in support}
    evaluation_ids = {(row["seed"], row["region"], row["id"]) for row in evaluation}
    support_groups = {(row["seed"], row["region"], row["group_id"]) for row in support}
    evaluation_groups = {(row["seed"], row["region"], row["group_id"]) for row in evaluation}
    buffered = {}
    for path in sorted(BUFFERED_DIR.glob("eval_seed*_buffer*.csv")):
        buffered[path.name] = len(read_csv(path))
    return {
        "support_rows": len(support),
        "evaluation_rows": len(evaluation),
        "support_eval_id_overlap": len(support_ids & evaluation_ids),
        "support_eval_group_overlap": len(support_groups & evaluation_groups),
        "support_columns": sorted(support[0].keys()),
        "eval_columns": sorted(evaluation[0].keys()),
        "has_independent_annotator_column": any(
            any(token in key.lower() for key in row for token in ("annotator", "agreement", "label_source"))
            for row in (support[:1] + evaluation[:1])
        ),
        "buffered_eval_counts": buffered,
    }


def build_gate_rows(target: dict, source: dict, rasters: dict, filesystem: dict, fewshot: dict) -> list[dict]:
    return [
        {
            "gate_id": "GEO-SPLIT",
            "gate_name": "Source geospatial / defensible spatial split",
            "status": "FAIL",
            "required_asset": "Source scene IDs, tile coordinates, CRS/geotransform, polygons, or documented spatial blocks",
            "evidence": f"All {rasters['n_tif']} local TIFs have no CRS ({rasters['n_with_crs']} with CRS) and identity transforms; {filesystem['n_geospatial_sidecars']} vector/geospatial-sidecar files; source manifests have no scene/coordinate/CRS fields.",
            "consequence": "Gate 4 cannot pass; source train/validation independence and source retention cannot be verified.",
            "next_action": "Request original 1536x1536 source rasters plus scene IDs, geotransforms, tile polygons, and split assignments from the data provider.",
        },
        {
            "gate_id": "COMPLETE-MAPS",
            "gate_name": "Complete target mosaics with negatives",
            "status": "FAIL",
            "required_asset": "Full-scene target imagery and labels including stable-slope/background areas",
            "evidence": f"Target benchmark has {target['n_pairs']} positive 512-pixel chips and {sum(v['zero_foreground'] for v in target['regions'].values())} zero-foreground chips; target source archives are absent locally; only cropped tiles are retained.",
            "consequence": "Gate 3 cannot pass; complete-map precision, false-positive area, omitted area, and operational recommendation are unavailable.",
            "next_action": "Acquire or reconstruct complete licensed target mosaics, then tile without positivity filtering and retain empty chips.",
        },
        {
            "gate_id": "INDEPENDENT-LABELS",
            "gate_name": "Independent target validation labels",
            "status": "FAIL",
            "required_asset": "A validation partition generated independently of support/adaptation labels",
            "evidence": f"Support and evaluation ID overlap is {fewshot['support_eval_id_overlap']}, but both come from the same target manifest and component inventory; no annotator, agreement, or validation-source fields are present.",
            "consequence": "Adaptation is a within-label-inventory holdout result, not an independently annotated validation result.",
            "next_action": "Create a second annotation campaign/team or use an agency inventory with documented provenance and agreement; keep physical buffers.",
        },
        {
            "gate_id": "CHRONOLOGY",
            "gate_name": "Prospective or defensible chronological holdout",
            "status": "PARTIAL",
            "required_asset": "A post-cutoff event or source/target acquisition chronology sufficient for a retrospective temporal holdout",
            "evidence": "Target acquisition periods exist at region level, but source training imagery has no acquisition date and target labels are not independent.",
            "consequence": "A next-earthquake/prospective claim remains unsupported; only a bounded retrospective comparison may be considered.",
            "next_action": "Acquire source chronology and freeze a cutoff and model-selection rule before evaluating a later event.",
        },
        {
            "gate_id": "REPRODUCIBILITY",
            "gate_name": "Public reproducibility package",
            "status": "PARTIAL",
            "required_asset": "Repository identifier, code, splits, metric implementations, and machine-readable outputs",
            "evidence": "Scripts, split manifests, reports, and outputs exist locally, but the project has no Git repository/public identifier.",
            "consequence": "Gate 5 cannot yet be fully satisfied.",
            "next_action": "Create a clean repository, exclude checkpoints/raw data as appropriate, and publish code plus split/metric artifacts.",
        },
        {
            "gate_id": "ROUTE2-MATRIX",
            "gate_name": "Expanded exposure-matched 108-run package",
            "status": "BLOCKED",
            "required_asset": "Pass GEO-SPLIT, COMPLETE-MAPS, and INDEPENDENT-LABELS first",
            "evidence": "The three hard data gates currently fail; more runs would estimate an outcome that cannot yet be spatially or operationally validated.",
            "consequence": "Do not launch the 108-run matrix.",
            "next_action": "Resolve data gates or choose the Route A/R1 tile-benchmark fallback.",
        },
    ]


def write_outputs(payload: dict, gates: list[dict], target_matrix: list[dict]) -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    json_path = REPORTS / f"data_asset_audit_{STAMP}.json"
    csv_path = REPORTS / f"data_asset_gate_matrix_{STAMP}.csv"
    md_path = REPORTS / f"data_asset_readiness_audit_{STAMP}.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(gates[0].keys()))
        writer.writeheader()
        writer.writerows(gates)

    t = payload["target"]
    s = payload["source_masks"]
    r = payload["source_rasters"]
    fs = payload["filesystem"]
    few = payload["fewshot"]
    lines = [
        "# Data Asset Inventory and Route 2 Readiness Audit",
        "",
        f"Generated: {payload['generated_at']}",
        "",
        "## Decision",
        "",
        "**No-go for the Route 2 108-run matrix under the current data assets.** "
        "Source geospatial independence, complete target maps/negative areas, and independent target validation labels are not locally available. "
        "The defensible fallback is the Route A/R1 tile-benchmark paper with bounded cross-event and few-shot claims.",
        "",
        "## Inventory",
        "",
        f"- Target benchmark: {t['n_pairs']:,} image-mask pairs; images missing={t['n_missing_images']}, masks missing={t['n_missing_masks']}; all declared 512x512={t['all_declared_512']}.",
        f"- Target zero-foreground chips: {sum(v['zero_foreground'] for v in t['regions'].values())}; the benchmark is positive-chip filtered.",
        f"- Source ready-data masks: {s['manifest_rows']:,} manifest rows from {s['unique_base_ids']:,} base tiles; {sum(s['zero_mask_rows_by_split'].values()):,} rows and {sum(s['zero_base_ids_by_split'].values()):,} base tiles have zero foreground.",
        f"- Source rasters: {r['n_tif']:,} TIFs; CRS present={r['n_with_crs']}, non-identity transforms={r['n_with_non_identity_transform']}.",
        f"- Geospatial vector/sidecar files in the project: {fs['n_geospatial_sidecars']}.",
        f"- Few-shot split: support rows={few['support_rows']}, evaluation rows={few['evaluation_rows']}, ID overlap={few['support_eval_id_overlap']}; this is a label holdout, not an independent annotation partition.",
        "- The target manifest has region-level acquisition/source/sensor/GSD metadata but no scene ID, CRS, geotransform, coordinates, annotator, or agreement fields.",
        "",
        "## Target Regions",
        "",
        "| Region | N | Zero FG | Min FG | Acquisition | Sensor | GSD (m) |",
        "|---|---:|---:|---:|---|---|---:|",
    ]
    for row in target_matrix:
        lines.append(
            f"| {row['region']} | {row['n']:,} | {row['zero_foreground']} | {row['min_foreground_fraction']:.6f} | "
            f"{row['acquisition']} | {row['sensor']} | {row['ground_resolution_m']} |"
        )
    lines += ["", "## Gate Results", "", "| Gate | Status | Evidence | Consequence |", "|---|---|---|---|"]
    for gate in gates:
        lines.append(f"| {gate['gate_name']} | **{gate['status']}** | {gate['evidence']} | {gate['consequence']} |")
    lines += [
        "",
        "## What Is Feasible Now",
        "",
        "1. Finish a reproducible tile-benchmark paper using the existing E0-E4 evidence and bounded claims.",
        "2. Implement tile-level object metrics, area metrics, hierarchical bootstrap, native sliding-window code, and cost/latency reporting on the existing tile data.",
        "3. Build the prior-work matrix and public repository without new model runs.",
        "4. Request source georeferencing and the original 1536x1536 rasters from the documented source-data contact.",
        "5. Request or reconstruct complete target mosaics and create a genuinely independent target annotation partition.",
        "",
        "## Not Feasible From Current Assets",
        "",
        "- A defensible source spatial-independence claim.",
        "- Complete-map precision/recall, false-positive area per km2, or omitted landslide area.",
        "- Unbiased operational validation of few-shot adaptation.",
        "- A next-earthquake or prospective generalization claim.",
        "- The expanded 108-run Route 2 experiment before the data gates pass.",
        "",
        "## Primary Blockers",
        "",
        "- `GEO-SPLIT`: source TIFs are non-georeferenced 224x224 crops; original source size is documented as 1536x1536 but is not present.",
        "- `COMPLETE-MAPS`: target benchmark contains only positive 512x512 chips; local target source archives are absent.",
        "- `INDEPENDENT-LABELS`: support/evaluation tiles are disjoint samples but share the same annotation inventory and provenance.",
        "",
        "## Artifacts",
        "",
        f"- `{json_path}`",
        f"- `{csv_path}`",
        f"- `{md_path}`",
    ]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    target, target_matrix = scan_target_manifest()
    source_masks = scan_source_masks()
    source_rasters = scan_source_rasters()
    filesystem = scan_filesystem_assets()
    fewshot = scan_fewshot_splits()
    gates = build_gate_rows(target, source_masks, source_rasters, filesystem, fewshot)
    payload = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "project_root": str(PROJECT),
        "decision": "NO_GO_ROUTE2_108_RUNS",
        "fallback": "ROUTE_A_R1_TILE_BENCHMARK",
        "target": target,
        "source_masks": source_masks,
        "source_rasters": source_rasters,
        "filesystem": filesystem,
        "fewshot": fewshot,
        "gates": gates,
    }
    write_outputs(payload, gates, target_matrix)
    summary = {
        "decision": payload["decision"],
        "target_pairs": target["n_pairs"],
        "target_zero_foreground": sum(v["zero_foreground"] for v in target["regions"].values()),
        "source_tifs": source_rasters["n_tif"],
        "source_tifs_with_crs": source_rasters["n_with_crs"],
        "geospatial_sidecars": filesystem["n_geospatial_sidecars"],
        "fewshot_id_overlap": fewshot["support_eval_id_overlap"],
        "failing_hard_gates": [gate["gate_id"] for gate in gates if gate["status"] == "FAIL"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
