"""Audit generated LORO source manifests for target exclusion and completeness."""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path

import os as _repo_os
PROJECT = Path(_repo_os.environ.get("LANDSLIDE_PROJECT_ROOT", Path(__file__).resolve().parents[1])).resolve()
SPLITS = PROJECT / "data" / "splits"
MANIFEST = PROJECT / "data" / "processed" / "external_regions_512" / "manifest_external.csv"
REPORTS = PROJECT / "reports"
OUT_CSV = REPORTS / "loro_protocol_audit.csv"
OUT_MD = REPORTS / "loro_protocol_audit.md"
TARGETS = ["wenchuan", "jiuzhai_valley", "moxitaidi", "longxi_river"]
LABELS = {
    "wenchuan": "Wenchuan",
    "jiuzhai_valley": "Jiuzhai Valley",
    "moxitaidi": "Moxitaidi",
    "longxi_river": "Longxi River",
}


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main():
    external = [row for row in read_csv(MANIFEST) if int(row.get("eligible_external", 1)) == 1]
    audits = []
    for target in TARGETS:
        path = SPLITS / f"multisource_exclude_{target}.csv"
        if not path.exists():
            audits.append({
                "target_region": target,
                "target_label": LABELS[target],
                "status": "MISSING_MANIFEST",
                "manifest_path": str(path),
                "n_rows": 0,
                "n_target_rows": None,
                "n_expected_source_rows": sum(row["region"] != target for row in external),
                "n_missing_source_rows": None,
                "n_extra_rows": None,
                "n_duplicate_rows": None,
                "split_id_overlap": None,
            })
            continue

        rows = read_csv(path)
        expected_keys = {(row["region"], row["id"]) for row in external if row["region"] != target}
        actual_keys = [(row["region"], row["id"]) for row in rows if row["region"] != "cas"]
        actual_counter = Counter(actual_keys)
        actual_set = set(actual_keys)
        duplicate_rows = sum(count - 1 for count in actual_counter.values() if count > 1)
        target_rows = sum(row["region"] == target for row in rows)
        missing = len(expected_keys - actual_set)
        extra = len(actual_set - expected_keys)

        split_members = defaultdict(lambda: defaultdict(set))
        for row in rows:
            if row["region"] != "cas":
                split_members[row["region"]][row["split"]].add(row["id"])
        overlap = sum(
            len(split_members[region]["train"] & split_members[region]["val"])
            for region in split_members
        )
        passed = target_rows == 0 and missing == 0 and extra == 0 and duplicate_rows == 0 and overlap == 0
        audits.append({
            "target_region": target,
            "target_label": LABELS[target],
            "status": "PASS" if passed else "FAIL",
            "manifest_path": str(path),
            "n_rows": len(rows),
            "n_target_rows": target_rows,
            "n_expected_source_rows": len(expected_keys),
            "n_missing_source_rows": missing,
            "n_extra_rows": extra,
            "n_duplicate_rows": duplicate_rows,
            "split_id_overlap": overlap,
        })

    with OUT_CSV.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(audits[0].keys()))
        writer.writeheader()
        writer.writerows(audits)

    lines = [
        "# LORO Protocol Audit",
        "",
        "This audit checks generated source manifests after each LORO target is excluded.",
        "PASS requires zero target rows, no missing/extra eligible source rows, no duplicate source IDs, and disjoint source train/val IDs.",
        "",
        "| Target | Status | Rows | Target rows | Expected sources | Missing | Extra | Duplicates | Train/val overlap |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in audits:
        lines.append(
            f"| {row['target_label']} | {row['status']} | {row['n_rows']} | {row['n_target_rows']} | "
            f"{row['n_expected_source_rows']} | {row['n_missing_source_rows']} | {row['n_extra_rows']} | "
            f"{row['n_duplicate_rows']} | {row['split_id_overlap']} |"
        )
    lines += [
        "",
        "Missing manifests indicate that the corresponding LORO run has not started or has not yet written its split manifest.",
        "",
        "Files:",
        f"- {OUT_CSV}",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT_CSV}")
    print(f"wrote {OUT_MD}")


if __name__ == "__main__":
    main()
