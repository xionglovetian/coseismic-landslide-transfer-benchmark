"""Build a portable index of experiment checkpoint files."""

from __future__ import annotations

import argparse
import csv
import hashlib
from datetime import datetime, timezone
from pathlib import Path


def classify(path: Path) -> str:
    if path.name == "best_model.pth":
        return "best"
    if path.name == "final_model.pth":
        return "final"
    if path.name.startswith("epoch_"):
        return "epoch"
    if path.name.startswith("step_"):
        return "step"
    return "other"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hash-best-final", action="store_true")
    args = parser.parse_args()
    root = args.project_root.resolve()
    outputs = root / "outputs"
    rows = []
    for path in sorted(outputs.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".pth", ".pt", ".ckpt"}:
            continue
        relative = path.relative_to(root).as_posix()
        role = classify(path)
        digest = ""
        if args.hash_best_final and role in {"best", "final"}:
            hasher = hashlib.sha256()
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    hasher.update(block)
            digest = hasher.hexdigest()
        rows.append(
            {
                "relative_path": relative,
                "run_name": path.relative_to(outputs).parts[0],
                "role": role,
                "size_bytes": path.stat().st_size,
                "modified_utc": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
                "sha256": digest,
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"WROTE: {args.output} ({len(rows)} checkpoints)")


if __name__ == "__main__":
    main()