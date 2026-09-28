"""Write confidence calibration report from JSON results."""

import json
from pathlib import Path

import os as _repo_os
PROJECT = Path(_repo_os.environ.get("LANDSLIDE_PROJECT_ROOT", Path(__file__).resolve().parents[1])).resolve()
data = json.loads((PROJECT / "reports" / "p4_confidence_calibration.json").read_text(encoding="utf-8"))
labels = {"source_val": "CAS validation", "hokkaido_iburi_tobu": "Hokkaido", "lombok": "Lombok", "palu": "Palu"}
lines = [
    "# P4-D: Confidence Calibration and High-Confidence Errors",
    "",
    "Checkpoint: ResUNet seed 42, epoch 50, 128x128. Confidence is max(p, 1-p); high confidence is >= 0.90.",
    "",
    "| Domain | Error rate | Mean confidence | High-confidence fraction | High-confidence error rate |",
    "|---|---:|---:|---:|---:|",
]
for key in ["source_val", "hokkaido_iburi_tobu", "lombok", "palu"]:
    row = data[key]
    lines.append(f"| {labels[key]} | {row['error_rate']:.4f} | {row['mean_confidence']:.4f} | {row['high_confidence_fraction']:.4f} | {row['high_confidence_error_rate']:.4f} |")
lines += [
    "",
    "## Confidence Bins",
    "",
    "| Domain | Confidence bin | Pixel fraction | Error rate | Mean confidence |",
    "|---|---|---:|---:|---:|",
]
for key in ["source_val", "hokkaido_iburi_tobu", "lombok", "palu"]:
    for bin_row in data[key]["bins"]:
        lines.append(f"| {labels[key]} | [{bin_row['lower']:.2f}, {bin_row['upper']:.2f}) | {bin_row['pixel_fraction']:.4f} | {bin_row['error_rate']:.4f} | {bin_row['mean_confidence']:.4f} |")
lines += [
    "",
    "Interpretation: the source domain has a much lower high-confidence error rate than the target domains. Hokkaido is particularly overconfident, so confidence cannot be used as a deployment reliability signal without target-domain calibration.",
]
(PROJECT / "reports" / "p4_confidence_calibration_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("wrote", PROJECT / "reports" / "p4_confidence_calibration_report.md")
