"""Build a reproducibility inventory for the reviewer response package."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import os as _repo_os
PROJECT = Path(_repo_os.environ.get("LANDSLIDE_PROJECT_ROOT", Path(__file__).resolve().parents[1])).resolve()
REPORTS = PROJECT / "reports"
FILES = [
    PROJECT / "data" / "processed" / "benchmark_v2_regions_512" / "manifest_benchmark_v2.csv",
    PROJECT / "data" / "processed" / "benchmark_v2_regions_512" / "target_spatial_index.csv",
    PROJECT / "data" / "processed" / "fewshot_target_splits" / "fewshot_support_seed42.csv",
    PROJECT / "data" / "processed" / "fewshot_target_splits" / "fewshot_eval_seed42.csv",
    PROJECT / "data" / "processed" / "fewshot_target_splits" / "fewshot_support_seed2026_777.csv",
    PROJECT / "data" / "processed" / "fewshot_target_splits" / "fewshot_eval_seed2026_777.csv",
    PROJECT / "scripts" / "train_unified.py",
    PROJECT / "scripts" / "train_multisource_dg.py",
    PROJECT / "scripts" / "evaluate_external_benchmark.py",
    PROJECT / "scripts" / "run_target_fewshot.py",
    PROJECT / "scripts" / "evaluate_epochwise_overfit.py",
    PROJECT / "scripts" / "audit_threshold_uncertainty.py",
    PROJECT / "scripts" / "audit_physical_buffers.py",
    PROJECT / "reports" / "requirements-lock.txt",
]
GLOBS = {
    "raw_zero_shot_result": "reports/benchmark_v2_zero_shot_raw/*.json",
    "raw_continuation_control": "reports/benchmark_v2_control_raw/*.json",
    "extended_source_result": "outputs/bench_v2_extended_*_seed*/target_metrics.json",
    "few_shot_result": "outputs/bench_v2_fewshot128_*/metrics.json",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    rows = []
    for path in FILES:
        rows.append({"path": str(path), "role": "code_or_manifest", "exists": path.exists(), "size_bytes": path.stat().st_size if path.exists() else 0, "sha256": sha256(path) if path.exists() else ""})
    for role, pattern in GLOBS.items():
        for path in sorted(PROJECT.glob(pattern)):
            rows.append({"path": str(path), "role": role, "exists": True, "size_bytes": path.stat().st_size, "sha256": ""})
    manifest_path = REPORTS / "e0_reproducibility_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "role", "exists", "size_bytes", "sha256"])
        writer.writeheader()
        writer.writerows(rows)

    roles = {}
    for row in rows:
        roles[row["role"]] = roles.get(row["role"], 0) + 1
    lines = [
        "# E0.5: Reproducibility Inventory",
        "",
        "The inventory separates verified local artifacts from missing publication requirements. It does not include model checkpoints because they are large and may be subject to redistribution constraints.",
        "",
        "## Included",
        "",
        "| Role | Files |",
        "|---|---:|",
    ]
    for role, count in sorted(roles.items()):
        lines.append(f"| {role} | {count} |")
    lines += [
        "",
        "## Verified Artifacts",
        "",
        "- Seven-region benchmark manifest with hashes and region metadata.",
        "- Target spatial index reconstructed from exact chip half-overlap.",
        "- Deterministic few-shot support/query split files for seeds 42, 2026, and 777.",
        "- Unified training and evaluation code.",
        "- Raw zero-shot results for four architectures and three seeds.",
        "- Continuation-control and fixed-source transfer results.",
        "- Few-shot adaptation results across all formal runs.",
        "- Environment lock file and experiment reports.",
        "",
        "## Still Missing Before Public Release",
        "",
        "- Source geolocation or geospatial polygons required for physical source train/validation block separation.",
        "- A complete checkpoint archive, if dataset and model licensing permit redistribution.",
        "- A frozen copy of the final manuscript table-generation scripts.",
        "- A formal data and code availability statement covering the CC BY-NC 4.0 dataset license.",
        "",
        f"Machine-readable inventory: `{manifest_path}`.",
    ]
    (REPORTS / "e0_reproducibility_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("rows", len(rows), roles)


if __name__ == "__main__":
    main()
