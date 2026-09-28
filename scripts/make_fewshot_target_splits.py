"""Create leakage-guarded nested few-shot target-domain splits."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import os as _repo_os
PROJECT = Path(_repo_os.environ.get("LANDSLIDE_PROJECT_ROOT", Path(__file__).resolve().parents[1])).resolve()
DEFAULT_INDEX = PROJECT / "data" / "processed" / "benchmark_v2_regions_512" / "target_spatial_index.csv"
DEFAULT_OUTPUT = PROJECT / "data" / "processed" / "fewshot_target_splits"
TARGET_REGIONS = ["hokkaido_iburi_tobu", "lombok", "palu"]


def stable_score(seed: int, region: str, sample_id: str) -> int:
    digest = hashlib.sha256(f"{seed}:{region}:{sample_id}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def choose_support(rows: list[dict], seed: int, max_shot: int, min_separation: int) -> list[dict]:
    """Pick nested support chips with component coverage and a spatial gap."""
    selected = []
    by_component = defaultdict(list)
    for row in rows:
        by_component[int(row["component_id"])].append(row)
    component_order = sorted(by_component, key=lambda cid: (-len(by_component[cid]), cid))

    def far_enough(candidate: dict) -> bool:
        for current in selected:
            if int(candidate["component_id"]) != int(current["component_id"]):
                continue
            if max(
                abs(int(candidate["x"]) - int(current["x"])),
                abs(int(candidate["y"]) - int(current["y"])),
            ) < min_separation:
                return False
        return True

    # First pass: cover distinct spatial components, largest components first.
    for component_id in component_order:
        if len(selected) >= max_shot:
            break
        candidates = sorted(
            by_component[component_id],
            key=lambda row: (stable_score(seed, row["region"], row["id"]), row["id"]),
        )
        for candidate in candidates:
            if far_enough(candidate):
                selected.append(candidate)
                break

    # Second pass: fill remaining slots from the globally deterministic order.
    if len(selected) < max_shot:
        candidates = sorted(
            rows,
            key=lambda row: (stable_score(seed, row["region"], row["id"]), row["id"]),
        )
        for candidate in candidates:
            if len(selected) >= max_shot:
                break
            if candidate in selected or not far_enough(candidate):
                continue
            selected.append(candidate)
    if len(selected) != max_shot:
        raise RuntimeError(f"Only selected {len(selected)}/{max_shot} support chips for {rows[0]['region']}")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", default=str(DEFAULT_INDEX))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--shots", type=int, nargs="+", default=[5, 10, 20])
    parser.add_argument("--regions", nargs="+", default=TARGET_REGIONS)
    parser.add_argument("--min-separation", type=int, default=3)
    args = parser.parse_args()
    if max(args.shots) != 20:
        raise ValueError("This protocol expects the maximum shot to be 20")

    rows = list(csv.DictReader(Path(args.index).open(encoding="utf-8-sig")))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_support_rows = []
    all_eval_rows = []
    summaries = {}

    for seed in args.seeds:
        for region in args.regions:
            region_rows = [row for row in rows if row["region"] == region]
            selected = choose_support(region_rows, seed, max(args.shots), args.min_separation)
            support_rank = {row["id"]: rank + 1 for rank, row in enumerate(selected)}
            selected_ids = set(support_rank)
            guard_keys = set()
            for row in selected:
                component_id, x, y = int(row["component_id"]), int(row["x"]), int(row["y"])
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        guard_keys.add((component_id, x + dx, y + dy))

            for shot in args.shots:
                for row in selected[:shot]:
                    record = dict(row)
                    record.update({"seed": seed, "shot": shot, "support_rank": support_rank[row["id"]]})
                    all_support_rows.append(record)

            eval_rows = []
            for row in region_rows:
                key = (int(row["component_id"]), int(row["x"]), int(row["y"]))
                if row["id"] in selected_ids or key in guard_keys:
                    continue
                record = dict(row)
                record.update({"seed": seed, "shot_scope": "common_20shot_guard"})
                eval_rows.append(record)
            all_eval_rows.extend(eval_rows)
            summaries[f"{region}_seed{seed}"] = {
                "n_region": len(region_rows),
                "n_support_20": len(selected),
                "n_guarded": len(region_rows) - len(eval_rows) - len(selected_ids),
                "n_eval_common": len(eval_rows),
                "support_by_component": dict(Counter(int(row["component_id"]) for row in selected)),
                "support_ids": [row["id"] for row in selected],
                "min_separation": args.min_separation,
            }
            print(region, "seed", seed, summaries[f"{region}_seed{seed}"], flush=True)

    support_path = output_dir / f"fewshot_support_seed{'_'.join(map(str, args.seeds))}.csv"
    eval_path = output_dir / f"fewshot_eval_seed{'_'.join(map(str, args.seeds))}.csv"
    summary_path = output_dir / f"fewshot_split_summary_seed{'_'.join(map(str, args.seeds))}.json"
    with support_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_support_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_support_rows)
    with eval_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_eval_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_eval_rows)
    summary_path.write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {support_path}")
    print(f"wrote {eval_path}")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
