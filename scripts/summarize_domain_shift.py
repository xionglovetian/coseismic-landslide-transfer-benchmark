"""Write domain-shift perturbation report from CSV."""

import csv
from pathlib import Path

import os as _repo_os
PROJECT = Path(_repo_os.environ.get("LANDSLIDE_PROJECT_ROOT", Path(__file__).resolve().parents[1])).resolve()
rows = list(csv.DictReader((PROJECT / "reports" / "p4_domain_shift.csv").open(encoding="utf-8-sig")))
labels = {"hokkaido_iburi_tobu": "Hokkaido", "lombok": "Lombok", "palu": "Palu"}
lines = [
    "# P4-F: Lightweight Domain-Shift Perturbations",
    "",
    "Checkpoint: ResUNet seed 42, epoch 50, 128x128. Each perturbation is applied after resizing to 128; the target mask is unchanged.",
    "",
    "| Region | Perturbation | IoU | Delta IoU | Dice |",
    "|---|---|---:|---:|---:|",
]
for row in rows:
    lines.append(f"| {labels[row['region']]} | {row['mode']} | {float(row['iou']):.4f} | {float(row['delta_iou']):+.4f} | {float(row['dice']):.4f} |")
lines += [
    "",
    "Interpretation: large degradation indicates sensitivity to acquisition or preprocessing shifts that are not represented by the source training distribution. Small changes are not evidence of robustness beyond these tested transforms.",
]
(PROJECT / "reports" / "p4_domain_shift_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("wrote", PROJECT / "reports" / "p4_domain_shift_report.md")
