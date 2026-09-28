"""Audit continuation-training exposure for CAS-only and pooled-source controls."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import os as _repo_os
PROJECT = Path(_repo_os.environ.get("LANDSLIDE_PROJECT_ROOT", Path(__file__).resolve().parents[1])).resolve()
LOGS = PROJECT / "logs"
REPORTS = PROJECT / "reports"
BATCH_SIZE = 32
EPOCHS = 20


def parse_train_count(log_path: Path) -> int:
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    match = re.search(r"train=(\d+)", text)
    if match:
        return int(match.group(1))
    match = re.search(r"train\s+(\d+)\s+val", text)
    if match:
        return int(match.group(1))
    raise RuntimeError(f"Cannot parse train count from {log_path}")


def make_record(name: str, log_name: str, regime: str) -> dict:
    train_count = parse_train_count(LOGS / log_name)
    steps_per_epoch = train_count // BATCH_SIZE
    total_steps = steps_per_epoch * EPOCHS
    return {
        "regime": regime,
        "run": name,
        "train_images": train_count,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "steps_per_epoch": steps_per_epoch,
        "optimizer_steps": total_steps,
        "image_exposures": total_steps * BATCH_SIZE,
        "patch_exposures": total_steps * BATCH_SIZE,
    }


def main() -> None:
    records = [
        make_record("bench_v2_cas_retrain20", "bench_v2_cas_retrain20_seed42.stdout.log", "CAS-only continuation"),
        make_record("bench_v2_extended_uniform", "bench_v2_extended_uniform_seed42.stdout.log", "pooled-source continuation"),
        make_record("bench_v2_extended_resunet", "bench_v2_extended_resunet_seed42.stdout.log", "pooled-source continuation"),
    ]
    baseline = records[0]
    for row in records:
        row["step_ratio_vs_cas"] = row["optimizer_steps"] / baseline["optimizer_steps"]
        row["exposure_ratio_vs_cas"] = row["image_exposures"] / baseline["image_exposures"]
        row["extra_exposures_vs_cas"] = row["image_exposures"] - baseline["image_exposures"]

    csv_path = REPORTS / "e0_exposure_ledger.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    lines = [
        "# E0.1: Continuation Exposure Ledger",
        "",
        "This audit uses the actual train counts parsed from the original launch logs. All runs use batch size 32 with `drop_last=True`.",
        "",
        "| Regime | Train images | Steps/epoch | Total optimizer steps | Image exposures | Step ratio vs CAS | Exposure ratio vs CAS |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in records:
        lines.append(
            f"| {row['regime']} | {row['train_images']} | {row['steps_per_epoch']} | {row['optimizer_steps']} | "
            f"{row['image_exposures']} | {row['step_ratio_vs_cas']:.2f}x | {row['exposure_ratio_vs_cas']:.2f}x |"
        )
    lines += [
        "",
        "## Conclusion",
        "",
        "The current CAS-only continuation and pooled-source continuation are epoch-matched, not optimizer-step matched and not exposure matched. The pooled-source regime receives approximately 6.34 times as many optimizer steps and image exposures.",
        "",
        "Therefore, the existing contrast cannot causally isolate the effect of adding source regions from the effect of additional optimization and sample exposure. It must be relabeled as an epoch-matched exploratory comparison until E1 is completed.",
    ]
    (REPORTS / "e0_exposure_ledger.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(records, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
